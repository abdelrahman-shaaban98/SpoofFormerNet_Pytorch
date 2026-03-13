from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFFN(nn.Module):
    """
    (Search by "There are two point-wise convolutional  (PConv)..." in paper).
    Convolutional Feed-Forward Network used inside both transformer blocks.
    Structure: PConv(1x1) → DConv(3x3) → GELU → PConv(1x1)
    Operates on spatial feature maps reconstructed from token sequences.
    """
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.pconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False) # pointwise convolution
        self.dconv  = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                                padding=1, groups=hidden_dim, bias=False) # depthwise convolution
        self.act    = nn.GELU()
        self.pconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False) # pointwise convolution
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        # Reconstruct spatial layout
        s = x.transpose(1, 2).reshape(B, C, H, W)
        s = self.pconv1(s)
        s = self.dconv(s)
        s = self.act(s)
        s = self.pconv2(s)
        s = self.drop(s)
        return s.flatten(2).transpose(1, 2)  # B, N, C
    

class WeightedMSA(nn.Module):
    """
    (Search by "we propose the implementation of a weighted multi-head..." in paper).
    Weighted Multi-Head Self-Attention.
    Each attention head is assigned a learnable scalar priority weight.

    Equations (9-11)
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv       = nn.Linear(dim, dim * 3, bias=False)
        self.proj      = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        # Learnable per-head priority weights
        self.head_w    = nn.Parameter(torch.ones(num_heads))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)   # B, N, 3, H, D
        qkv = qkv.permute(2, 0, 3, 1, 4)                                    # 3, B, H, N, D
        q, k, v = qkv.unbind(0)                                             # each: B, H, N, D

        attn = (q @ k.transpose(-2, -1)) * self.scale                       # Eq. 9 (B, H, N, D) x (B, H, D, N) --> B, H, N, N
        attn = attn.softmax(dim=-1)                                         # B, H, N, N
        attn = self.attn_drop(attn)

        out = attn @ v  # Eq. 10 (B, H, N, N) x (B, H, N, D) --> B, H, N, D
        # Apply normalised per-head weights
        hw = F.softmax(self.head_w, dim=0).view(1, self.num_heads, 1, 1)
        out = out * hw # Eq. 11

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out



class LocalWindowAttention(nn.Module):
    """
    (Search by "the Window-Local transformer block is intended to capture..." in paper).
    Local window weighted multi-head self-attention (LW-WMSA).
    Partitions tokens into non-overlapping windows of size W_L x W_L,
    applies WMSA within each window (Section 3.3).
    """
    def __init__(self, dim: int, num_heads: int = 8,
                 window_size: int = 7, dropout: float = 0.0):
        super().__init__()
        self.ws   = window_size
        self.wmsa = WeightedMSA(dim, num_heads, dropout)

    def _partition(self, x: torch.Tensor, H: int, W: int):
        """x: B, H, W, C  →  num_wins*B, ws*ws, C"""
        B, _, _, C = x.shape
        ws = self.ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws

        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            
        Hp, Wp = H + pad_h, W + pad_w
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)
        return x, Hp, Wp

    def _reverse(self, wins: torch.Tensor, H: int, W: int,
                 Hp: int, Wp: int, B: int):
        """num_wins*B, ws*ws, C  →  B, H, W, C"""
        ws = self.ws
        C = wins.shape[-1]
        x = wins.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, C)
        return x[:, :H, :W, :].contiguous()

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x_2d = x.view(B, H, W, C)
        wins, Hp, Wp = self._partition(x_2d, H, W)
        wins = self.wmsa(wins)
        x_2d = self._reverse(wins, H, W, Hp, Wp, B)
        return x_2d.view(B, N, C)



class SparseGlobalAttention(nn.Module):
    """
    (Search by "The SWindow-Global transformer block, on the other hand..." in paper).
    Sparse-window global weighted multi-head self-attention (GW-WMSA).
    Uniformly samples tokens from the entire feature space, groups them
    into sparse windows, and applies WMSA to capture long-range context
    """
    def __init__(self, dim: int, num_heads: int = 8,
                 num_sparse: int = 64, dropout: float = 0.0):
        super().__init__()
        self.num_sparse = num_sparse
        self.wmsa       = WeightedMSA(dim, num_heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        k = min(self.num_sparse, N)

        # Uniform sampling across the spatial sequence
        idx = torch.linspace(0, N - 1, k, dtype=torch.long, device=x.device)
        sparse = x[:, idx, :]        # B, k, C – sparse window tokens

        sparse_out = self.wmsa(sparse).to(torch.float32)

        # Scatter refined sparse tokens back
        out = x.clone()
        # print(out.shape)
        # print(out.dtype)
        # print(sparse_out.shape)
        # print(sparse_out.dtype)
        out[:, idx, :] = sparse_out
        return out

class WindowLocalBlock(nn.Module):
    """
    Window-Local Transformer Block (Figure 4, Equations 1-4):
      LN → LW-WMSA → residual → LN → FFN → residual
    """
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 7,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = LocalWindowAttention(dim, num_heads, window_size, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = ConvFFN(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), H, W)   # Eq. 1-2
        x = x + self.ffn(self.norm2(x), H, W)    # Eq. 3-4
        return x



class SGlobalBlock(nn.Module):
    """
    Sparse-Global Transformer Block (Figure 4, Equations 5-8):
      LN → GW-WMSA → residual → LN → FFN → residual
    """
    def __init__(self, dim: int, num_heads: int = 8, num_sparse: int = 64,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = SparseGlobalAttention(dim, num_heads, num_sparse, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn   = ConvFFN(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # Eq. 5-6
        x = x + self.ffn(self.norm2(x), H, W)    # Eq. 7-8
        return x



class TransformerModule(nn.Module):
    """
    (Figure 2)
    One transformer module = WindowLocalBlock + SGlobalBlock.
    """
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 7,
                 num_sparse: int = 64, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.local_blk  = WindowLocalBlock(dim, num_heads, window_size, mlp_ratio, dropout)
        self.global_blk = SGlobalBlock(dim, num_heads, num_sparse, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.local_blk(x, H, W)
        x = self.global_blk(x, H, W)
        return x
