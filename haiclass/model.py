"""Voxel transformer: full self-attention within spatial blocks of voxels.

Device-agnostic: plain nn.MultiheadAttention-style SDPA blocks, no CUDA-only ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding: torch.Tensor | None) -> torch.Tensor:
        # x: (B, N, D); key_padding: (B, N) True where padded
        h = self.norm1(x)
        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, D // self.heads)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))  # (B, H, N, d)
        attn_mask = None
        if key_padding is not None:
            attn_mask = ~key_padding[:, None, None, :]  # True = attend
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        h = h.transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.proj(h))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class VoxelTransformer(nn.Module):
    """Encoder over per-voxel features + block-relative coordinates."""

    def __init__(self, cfg: Config, num_features: int):
        super().__init__()
        self.cfg = cfg
        in_dim = num_features + 3  # + block-relative xyz
        self.embed = nn.Sequential(
            nn.Linear(in_dim, cfg.dim),
            nn.GELU(),
            nn.Linear(cfg.dim, cfg.dim),
        )
        self.blocks = nn.ModuleList(
            [Block(cfg.dim, cfg.heads, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.num_classes)

    def forward(
        self,
        feats: torch.Tensor,      # (B, N, F)
        coords: torch.Tensor,     # (B, N, 3) block-relative, metres
        key_padding: torch.Tensor | None = None,  # (B, N) True where padded
    ) -> torch.Tensor:
        # scale coords to O(1): blocks are typically a few tens of metres
        x = torch.cat([feats, coords * 0.05], dim=-1)
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x, key_padding)
        return self.head(self.norm(x))  # (B, N, C)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
