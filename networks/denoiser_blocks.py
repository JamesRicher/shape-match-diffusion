"""AdaLN-Zero variants of the intra/inter shape blocks, for the matrix denoiser.

These mirror networks/baseline_blocks.py (same pre-norm intra self-attn / inter cross-attn
trunk, same weight-sharing symmetries) but replace the LayerNorms with DiT-style
AdaLN-Zero modulation driven by a conditioning vector c (the timestep pathway, from
networks/denoiser_conditioning.py). All gates are zero-init, so every block is exactly
identity at init and the timestep conditioning fades in — the validated supervised
trunk is the starting point.

The blocks accept an attention bias that is either shared across heads (B, Lq, Lk) or
per-head (B, H, Lq, Lk); the per-head form is where the learned per-head/per-layer P_t
and geodesic-kernel scales (denoiser_conditioning.py) plug in. The blocks themselves
stay agnostic to the bias content.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from networks.layers import MultiHeadAttention, FeedForward


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN affine modulation: x * (1 + scale) + shift. shift/scale are (B, 1, D),
    broadcast over tokens."""
    return x * (1 + scale) + shift


class AdaLNModulation(nn.Module):
    """Maps conditioning c -> (shift, scale, gate) for each of a block's sublayers.

    Zero-init (AdaLN-Zero): at init shift=scale=0 (norm passes through) and gate=0 (the
    sublayer's residual contributes nothing), so the block starts as the identity.
    """
    def __init__(self, dim: int, n_sublayers: int = 2):
        super().__init__()
        self.n_sublayers = n_sublayers
        self.proj = nn.Linear(dim, 3 * n_sublayers * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c: torch.Tensor):
        # SiLU then Linear (DiT convention). (B, 3*n_sublayers*D) -> tokens broadcast.
        params = self.proj(F.silu(c)).unsqueeze(1)          # (B, 1, 3*n_sublayers*D)
        return params.chunk(3 * self.n_sublayers, dim=-1)   # 3*n_sublayers x (B, 1, D)


class AdaLNIntraShapeBlock(nn.Module):
    """Intra-shape self-attention block with AdaLN-Zero timestep conditioning.

    One instance is shared across both shapes (call it once per shape), preserving the
    intra weight-sharing symmetry. Identity at init.
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model, elementwise_affine=False) # no LN learning here
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_mlp = nn.LayerNorm(d_model, elementwise_affine=False) # no LN learning herer
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)
        self.mod = AdaLNModulation(d_model, n_sublayers=2)

    def forward(self, x: torch.Tensor, c: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.mod(c)
        h = modulate(self.norm_attn(x), sh_a, sc_a)
        x = x + g_a * self.attn(h, h, bias=bias)
        h = modulate(self.norm_mlp(x), sh_m, sc_m)
        x = x + g_m * self.mlp(h)
        return x


class AdaLNInterShapeBlock(nn.Module):
    """Inter-shape cross-attention block with AdaLN-Zero timestep conditioning.

    Shared attn/mlp weights and a single norm applied to both shapes, modulated by the
    same c; the reverse direction uses the transposed bias. Together with mirrored
    inputs this gives exact pair-swap symmetry f(P^T, Y<->X) = f(P)^T. Identity at init.
    """
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_mlp = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp = FeedForward(d_model, mlp_ratio, dropout)
        self.mod = AdaLNModulation(d_model, n_sublayers=2)

    def forward(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor,
                bias: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = self.mod(c)
        # same modulation + norm for both shapes -> symmetric
        hx = modulate(self.norm(x), sh_a, sc_a)
        hy = modulate(self.norm(y), sh_a, sc_a)
        bias_t = bias.transpose(-1, -2) if bias is not None else None  # works for 3D and 4D

        x = x + g_a * self.attn(hx, hy, bias=bias)     # X <- Y
        y = y + g_a * self.attn(hy, hx, bias=bias_t)   # Y <- X

        x = x + g_m * self.mlp(modulate(self.norm_mlp(x), sh_m, sc_m))
        y = y + g_m * self.mlp(modulate(self.norm_mlp(y), sh_m, sc_m))
        return x, y
