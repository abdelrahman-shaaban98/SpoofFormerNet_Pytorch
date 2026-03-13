from typing import List, Tuple
import torch
import torch.nn as nn


class MultiScaleTokenEmbedding(nn.Module):
    """
    (Search by "A token embedding module is used to generate..." in paper).
    Multi-scale patch token embedding.
    Splits the feature map into 4 groups of patches with varying sizes
    [1x1, 3x3, 5x5, 7x7], applies linear embedding to each group,
    then concatenates sub-tokens from smaller to larger patches.

    Useful for capturing both local (small) and global (large) features.
    """
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_sizes: List[int] = [1, 3, 5, 7],
        stride: int = 2, # As paper, to reduce the extent of the feature maps by 50%
    ):
        super().__init__()
        assert embed_dim % len(patch_sizes) == 0, \
            "embed_dim must be divisible by the number of patch sizes"

        self.patch_sizes = patch_sizes
        self.stride = stride
        self.sub_dim = embed_dim // len(patch_sizes)

        # One conv per patch size; all share the same stride so output H/W is identical
        self.sub_embeds = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels, self.sub_dim,
                    kernel_size=ps, stride=stride,
                    padding=ps // 2, bias=False
                ),
                nn.BatchNorm2d(self.sub_dim),
            )
            for ps in patch_sizes
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # Tokens goes through four embedding layers, then concatenate them.
        sub_tokens = [embed(x) for embed in self.sub_embeds]  # each: B, sub_dim, H', W'
        # for s in sub_tokens:
        #     print(s.shape)
        out = torch.cat(sub_tokens, dim=1)                     # B, embed_dim, H', W'
        B, C, H, W = out.shape
        tokens = out.flatten(2).transpose(1, 2)                # B, H'W', embed_dim
        tokens = self.norm(tokens)
        return tokens, H, W # return H, W as those will be the resolution of the first branch