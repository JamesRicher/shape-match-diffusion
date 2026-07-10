import torch
import torch.nn as nn
from typing import Tuple

from networks.layers import MultiHeadAttention, FeedForward


class IntraShapeBlock(nn.Module):
    """
    MHA block with FFN and norm used to updated features within a shape.
    This uses a prenorm formulation with LN
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm_attn(x)
        x = x + self.attn(h, h, bias=bias)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class InterShapeBlock(nn.Module):
    """
    MHA block with FFN and norm for cross attending shape features
    This uses prenorm formulation with LN
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        hx, hy = self.norm(x), self.norm(y)                 # same norm -> symmetric
        bias_t = bias.transpose(-1, -2) if bias is not None else None

        # cross-attention; both updates read the pre-update normed Tensors
        x = x + self.attn(hx, hy, bias=bias)  # X <- Y
        y = y + self.attn(hy, hx, bias=bias_t)  # Y <- X

        x = x + self.mlp(self.norm_mlp(x))
        y = y + self.mlp(self.norm_mlp(y))
        return x, y
