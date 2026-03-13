from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding import MultiScaleTokenEmbedding
from transformer import TransformerModule 


class ConvStem(nn.Module):
    """
    (Search by "the uppermost section starts with a convolution stem..." in paper).
    Initial convolution stem for low-level feature extraction
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 32, num_layers: int = 2):
        super().__init__()
        layers = []

        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            layers += [
                nn.Conv2d(cin, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.stem = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)  # B, out_channels, H, W


class TokenMerger(nn.Module):
    """
    (Search by "the uppermost section starts with a convolution stem..." in paper)
    Cross-branch token merging
    One adapter conv per each source to change the resolution and channels to the target branch
    and another 1x1 fusion conf to fuse the output of the adapters with the target branch output 
    """
    def __init__(self, target_dim: int, source_dims: List[int]):
        super().__init__()
        # One adapter conv per source branch
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sd, target_dim, 1, bias=False),
                nn.BatchNorm2d(target_dim),
                nn.ReLU(inplace=True),
            )
            for sd in source_dims
        ])
        # 1x1 fusion conv (target + all sources concatenated)
        total = target_dim * (1 + len(source_dims))
        self.fuse = nn.Sequential(
            nn.Conv2d(total, target_dim, 1, bias=False),
            nn.BatchNorm2d(target_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        target: torch.Tensor, tH: int, tW: int,
        sources: List[torch.Tensor],
        sHs: List[int], sWs: List[int],
    ) -> torch.Tensor:
        B, _, C = target.shape
        t_sp = target.transpose(1, 2).reshape(B, C, tH, tW)
        parts = [t_sp] # Initialize the "parts" list with the current branch tensor, 
        # and then append to it the other branches tensor
        for adapter, src, sH, sW in zip(self.adapters, sources, sHs, sWs):
            sC = src.shape[-1]
            s_sp = src.transpose(1, 2).reshape(B, sC, sH, sW)
            s_sp = adapter(s_sp) # To adjust the channels of the source to the target
            s_sp = F.interpolate(s_sp, size=(tH, tW), mode='bilinear', align_corners=False) # To adjust the resolution of the source to the target
            parts.append(s_sp)

        fused = torch.cat(parts, dim=1)         # B, total, tH, tW
        fused = self.fuse(fused)                # B, target_dim, tH, tW
        return fused.flatten(2).transpose(1, 2) # B, tH*tW, target_dim


class HRViTStream(nn.Module):
    """
    ("the uppermost section starts with a convolution stem..." in paper)
    Single High-Resolution Vision Transformer stream Search by
    ConvStem  →  MultiScaleTokenEmbedding  →  5-level HR-ViT
    Branch spatial scales  : 1/2, 1/4, 1/8, 1/16  (relative to stem output)
    Branch channel sizes   : C, 2C, 4C, 8C
    """
    def __init__(
        self,
        in_channels: int     = 3,
        base_dim: int        = 32,
        stem_layers: int     = 2,
        num_levels: int      = 5,
        num_branches: int    = 4,
        blocks_per_branch: List[int] = [2, 2, 2, 2],   # N1, N2, N3, N4
        num_heads: int       = 4,
        window_size: int     = 7,
        num_sparse: int      = 64,
        mlp_ratio: float     = 4.0,
        dropout: float       = 0.0,
        patch_sizes: List[int] = [1, 3, 5, 7],

        patch_stride: int    = 2,
    ):
        super().__init__()
        self.num_levels   = num_levels
        self.num_branches = num_branches
        self.branch_dims  = [base_dim * (2 ** b) for b in range(num_branches)] # [32, 64, 128, 256] for base_dim = 32

        # Active branches at each level
        self._active = [min(1 + level, num_branches) for level in range(num_levels)] # [1, 2, 3, 4, 4] for num_levels, num_branches = 5, 4

        # Stem & Token Embedding in the first level
        self.stem       = ConvStem(in_channels, base_dim, stem_layers)
        self.tok_embed  = MultiScaleTokenEmbedding(base_dim, base_dim, patch_sizes, patch_stride)

        # Branch expansion convolutions to create new branch at each level
        self.expanders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.branch_dims[b], self.branch_dims[b + 1],
                          kernel_size=3, stride=2, padding=1, bias=False), # stride=2 to create a new branch with half the resolution
                nn.BatchNorm2d(self.branch_dims[b + 1]),
                nn.ReLU(inplace=True),
            ) # 32 --> 64 --> 128 --> 256 
            for b in range(num_branches - 1)
        ])

        # Transformer stages: stages[level][branch] = nn.ModuleList of blocks
        # Each level has some branches, and each branch has N blocks
        self.stages = nn.ModuleList()
        for level in range(num_levels):
            level_mods = nn.ModuleList()
            for active_branches in range(self._active[level]):
                branch_blocks = blocks_per_branch[active_branches] 
                level_mods.append(nn.ModuleList([
                    TransformerModule(
                        dim=self.branch_dims[active_branches],
                        num_heads=num_heads,
                        window_size=window_size,
                        num_sparse=num_sparse,
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                    )
                    for _ in range(branch_blocks)
                ]))
            self.stages.append(level_mods)

        # Token mergers: mergers[level][branch]
        # Each level has some branches, and each branch has one merger
        self.mergers = nn.ModuleList()
        for level in range(num_levels):
            if level == 0: # No merging at the first level
                self.mergers.append(None)
                continue
            active_branches = self._active[level]
            level_mergers = nn.ModuleList([
                TokenMerger(
                    target_dim=self.branch_dims[b],
                    source_dims=[self.branch_dims[s] for s in range(active_branches) if s != b], 
                    # No need to send current branch in the source_dims as it's already handled inside the ToikenMerger
                )
                for b in range(active_branches) # One merger for each branch, that merges the output of the previous stage
                # b: at level 2: [0, 1], 
                #    at level 3: [0, 1, 2], 
                #    at level 4: [0, 1, 2, 3], 
                #    at level 5: [0, 1, 2, 3]
            ])
            self.mergers.append(level_mergers)

        # Output aggregation: transform 1-3 channels to branch-0 channel
        self.up_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.branch_dims[b], base_dim, 1, bias=False),
                nn.BatchNorm2d(base_dim),
            ) # [64 --> 32], [128 --> 32], [256 --> 32], 
            for b in range(1, num_branches)
        ])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #TODO: Add dimension before and after each layer
        B = x.shape[0]
        # print(x.shape)
        x = self.stem(x)                          # B, C, H, W
        # print(x.shape)
        tokens, H0, W0 = self.tok_embed(x)        # B, H'W', C

        # Store tokens, H, and W for each branch
        # Initially, we have only one branch
        # tok_list shall hold the output tensor of each branch, that get's updated at each level
        # H_list, W_list shall hold the H, W of each branch created, 
        #   which are halves of the previous branch and don't get updated at each level as they're constants
        tok_list: List[torch.Tensor] = [tokens]
        H_list: List[int]            = [H0]
        W_list: List[int]            = [W0]

        for level in range(self.num_levels):
            active_branches = self._active[level] # [1, 2, 3, 4, 4] for num_levels, num_branches = 5, 4

            # Expand branches if needed
            while len(tok_list) < active_branches:
                last_branch_idx = len(tok_list) - 1

                last_branch_C = tok_list[last_branch_idx].shape[-1]
                last_branch_tensor  = tok_list[last_branch_idx].transpose(1, 2).reshape(B, last_branch_C, H_list[last_branch_idx], W_list[last_branch_idx])

                created_branch_tensor  = self.expanders[last_branch_idx](last_branch_tensor)
                nH, nW = created_branch_tensor.shape[2], created_branch_tensor.shape[3]

                tok_list.append(created_branch_tensor.flatten(2).transpose(1, 2))
                H_list.append(nH)
                W_list.append(nW)

            # Cross-branch token merging (levels 1+) 
            if level > 0:
                merged_list = []
                level_mergers = self.mergers[level]

                for b in range(active_branches):
                    srcs   = [tok_list[s] for s in range(active_branches) if s != b]
                    src_Hs = [H_list[s]   for s in range(active_branches) if s != b]
                    src_Ws = [W_list[s]   for s in range(active_branches) if s != b]
                    merged = level_mergers[b](
                        tok_list[b], H_list[b], W_list[b],
                        srcs, src_Hs, src_Ws
                    )
                    merged_list.append(merged)
                    
                tok_list = merged_list

            # Transformer blocks per branch
            for b in range(active_branches):
                for blk in self.stages[level][b]:
                    tok_list[b] = blk(tok_list[b], H_list[b], W_list[b])


        # Aggregate all branches to branch-0 dimensions
        H_top, W_top = H_list[0], W_list[0]
        C0 = self.branch_dims[0]
        out = tok_list[0].transpose(1, 2).reshape(B, C0, H_top, W_top)

        for b in range(1, self.num_branches):
            Cb = tok_list[b].shape[-1]
            branch_tensor = tok_list[b].transpose(1, 2).reshape(B, Cb, H_list[b], W_list[b])
            # Adapt to branch-0 resolution
            branch_tensor = F.interpolate(branch_tensor, size=(H_top, W_top), mode='bilinear', align_corners=False)   
            # Adapt to branch-0 channels
            branch_tensor = self.up_convs[b - 1](branch_tensor)                                                      
            out = out + branch_tensor

        return out   # B, base_dim, H_top, W_top


class SpoofFormerNet(nn.Module):
    def __init__(
        self,
        rgb_in_channels: int   = 3,
        depth_in_channels: int = 1,
        base_dim: int          = 32,
        stem_layers: int       = 2,
        num_levels: int        = 5,
        num_branches: int      = 4,
        blocks_per_branch: List[int] = [2, 2, 2, 2],
        num_heads: int         = 4,
        window_size: int       = 7,
        num_sparse: int        = 64,
        mlp_ratio: float       = 4.0,
        dropout: float         = 0.1,
        num_classes: int       = 2,
    ):
        super().__init__()

        stream_kwargs = dict(
            base_dim        = base_dim,
            stem_layers     = stem_layers,
            num_levels      = num_levels,
            num_branches    = num_branches,
            blocks_per_branch = blocks_per_branch,
            num_heads       = num_heads,
            window_size     = window_size,
            num_sparse      = num_sparse,
            mlp_ratio       = mlp_ratio,
            dropout         = dropout,
        )
        self.rgb_stream   = HRViTStream(in_channels=rgb_in_channels,   **stream_kwargs)
        self.depth_stream = HRViTStream(in_channels=depth_in_channels, **stream_kwargs)

        fused_dim = base_dim * 2   # concat of both stream outputs
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        
            nn.Flatten(),
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),    # logits; Softmax applied in predict()
        )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        rgb_feat   = self.rgb_stream(rgb)           # B, C, H', W'
        depth_feat = self.depth_stream(depth)       # B, C, H', W'

        fused  = torch.cat([rgb_feat, depth_feat], dim=1)  # B, 2C, H', W'
        logits = self.classifier(fused)                     # B, num_classes
        return logits


    @torch.no_grad()
    def predict(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
    ) -> torch.Tensor:
        """Returns class probabilities (softmax of logits)."""
        return F.softmax(self.forward(rgb, depth), dim=-1)


def build_spoof_former_net(variant: str = "base") -> SpoofFormerNet:
    """
    Build SpoofFormerNet with predefined configurations.

    Variants:
      'tiny'    Fast testing                        [0.81 M params]
      'base'    Similar to the paper but smaller    [5.83 M params]
    """
    configs = {
        "tiny": dict(
            base_dim=16, stem_layers=2, num_levels=4,
            num_branches=3, blocks_per_branch=[1, 1, 1, 1],
            num_heads=4, window_size=7, dropout=0.1,
        ),
        "base": dict(
            base_dim=16, stem_layers=2, num_levels=5,
            num_branches=4, blocks_per_branch=[2, 2, 2, 2],
            num_heads=4, window_size=7, dropout=0.1,
        ),
    }

    if variant not in configs:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(configs)}")
    
    return SpoofFormerNet(**configs[variant])