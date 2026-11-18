"""Building blocks of the TacInno tokenizer: multi-head attention, the spatio-temporal block and field (un)patching."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Attention", "SpatioTemporalBlock", "unpatchify"]

# The fused attention kernels fail ("invalid configuration argument") above ~65k sequences x heads, e.g. batch 512 x 137
# tokens, so the temporal attention (one short sequence per token position) runs in chunks of sequences.
TEMPORAL_CHUNK = 16384


class Attention(nn.Module):
    """Multi-head self-attention, x (B, N, C) -> (B, N, C); ``causal`` applies a lower-triangular mask."""

    def __init__(self, dim: int = 384, num_heads: int = 6):
        super().__init__()
        assert dim % num_heads == 0, (dim, num_heads)
        self.dim, self.num_heads = dim, num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        B, N, _ = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.proj(y.transpose(1, 2).reshape(B, N, self.dim))


class SpatioTemporalBlock(nn.Module):
    """Pre-LN block on x (B, frames, tokens, C):

    1. spatial self-attention among the tokens of each frame;
    2. temporal self-attention across frames at each token position, causal (frame f attends to frames <= f);
    3. MLP (C -> 4C -> C, GELU).
    Each sub-layer is residual.
    """

    def __init__(self, dim: int = 384, num_heads: int = 6, mlp_ratio: int = 4):
        super().__init__()
        self.dim = dim
        self.ln_s, self.ln_t, self.ln_m = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.s_attn, self.t_attn = Attention(dim, num_heads), Attention(dim, num_heads)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Fr, N, _ = x.shape
        D = self.dim
        x = x + self.s_attn(self.ln_s(x).reshape(B * Fr, N, D)).reshape(B, Fr, N, D)
        xt = self.ln_t(x).transpose(1, 2).reshape(B * N, Fr, D)
        yt = torch.cat([self.t_attn(c, causal=True) for c in xt.split(TEMPORAL_CHUNK)], 0)
        x = x + yt.reshape(B, N, Fr, D).transpose(1, 2)
        return x + self.mlp(self.ln_m(x))


def unpatchify(p: torch.Tensor, num_gels: int = 2, channels: int = 3, grid_hw=(6, 8), patch: int = 16) -> torch.Tensor:
    """(B, gels * rows * cols, patch * patch * channels) -> (B, gels, channels, rows * patch, cols * patch).

    Inverse of the patch embedding's token order (gel, row, col); the outputs of one token are ordered
    (channel, patch_row, patch_col)."""
    B = p.shape[0]
    rows, cols = grid_hw
    p = p.reshape(B, num_gels, rows, cols, channels, patch, patch).permute(0, 1, 4, 2, 5, 3, 6)
    return p.reshape(B, num_gels, channels, rows * patch, cols * patch)
