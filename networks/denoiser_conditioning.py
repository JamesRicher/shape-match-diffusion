"""Conditioning for the matrix denoiser: the timestep spine and the per-head biases.

Everything here injects a signal INTO the trunk's attention: the timestep pathway
c = MLP_t(sinusoid(t)) drives AdaLN-Zero in every block, and the two bias constructors
turn P_t and geodesic distance into the per-head additive attention biases the blocks
consume. See notes/2026-07-08_denoiser_architecture.md.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sinkhorn import safe_log


# --------------------------------------------------------------------------- #
# conditioning spine: t -> c
# --------------------------------------------------------------------------- #
def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Standard sinusoidal embedding of a scalar per sample. t: (B,) -> (B, dim).

    Frequencies are geometric in (1/max_period, 1]; scale t into a range that spans
    them before calling (see ConditioningSpine.time_scale) or the embedding barely
    varies over t in [0, 1].
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # pad to dim when odd
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class ConditioningSpine(nn.Module):
    """Timestep pathway c = MLP_t(sinusoid(t)) driving AdaLN-Zero in every block.

    forward returns a SUM of contributions (currently just the t pathway); the cascade's
    s and level embeddings add into that same sum later, zero-init, with no surgery here.

    Args:
        dim: conditioning width (matches the token dim the AdaLN blocks expect).
        embed_dim: sinusoidal embedding width before the MLP (defaults to dim).
        time_scale: t is multiplied by this before embedding so continuous t in [0, 1]
            spans the sinusoidal frequencies (1000 mirrors 1000-step diffusion).
        max_period: sinusoidal max period.
    """
    def __init__(self, dim: int, embed_dim: int | None = None,
                 time_scale: float = 1000.0, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim or dim
        self.time_scale = time_scale
        self.max_period = max_period
        self.t_mlp = nn.Sequential(nn.Linear(self.embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32, device=self.t_mlp[0].weight.device)
        if t.dim() == 0:
            t = t[None]
        emb = sinusoidal_embedding(t * self.time_scale, self.embed_dim, self.max_period)
        c = self.t_mlp(emb)
        # cascade extras add into this sum later: c = c + self.s_mlp(sinusoid(s)) + level_emb
        return c


# --------------------------------------------------------------------------- #
# per-head attention bias construction
# --------------------------------------------------------------------------- #
class LogAssignmentBias(nn.Module):
    """Inter cross-attention bias (Route 1): per-head gamma_h * log(clamp(P_t, eps)).

    gamma is unconstrained and zero-init, so the P_t prior fades in (AdaLN-Zero
    philosophy) and useful ranges cover ignore (0), tempered prior (0<g<1), product of
    experts (1), near-hard routing (>1), even anti-prior (<0). One instance per inter
    layer. The eps-clamp is mandatory: a -inf bias is a dead, zero-gradient edge.

    P_t is (B, n_y, n_x). Returns the X<-Y per-head bias (B, H, n_x, n_y) that the inter
    block consumes as its `bias` arg; the block transposes it internally for Y<-X, so
    Y token j attends X token i with the correct prior log P_t[j, i] in both directions.
    """
    def __init__(self, n_heads: int, eps: float = 1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(n_heads))
        self.eps = eps

    def forward(self, P_t: torch.Tensor) -> torch.Tensor:
        logP = safe_log(P_t, self.eps)                       # (B, n_y, n_x)
        base = logP.transpose(-1, -2).unsqueeze(1)           # (B, 1, n_x, n_y)  [X<-Y]
        return self.gamma.view(1, -1, 1, 1) * base           # (B, H, n_x, n_y)


class GeodesicKernelBias(nn.Module):
    """Intra self-attention bias: per-head -gamma_h * D^2 (log of a Gaussian kernel on
    geodesic distance; Graphormer/ALiBi-style spatial encoding). Baseline, not an add-on
    -- the trunk's only relative-position signal.

    gamma_h > 0 is a per-head inverse bandwidth (softplus), initialised spread from
    global (small) to local (large) so heads cover a range of scales. One instance per
    intra layer, shared across both shapes (call once per shape with that shape's D).
    D is symmetric with zero diagonal, so the bias is too (self-distance -> 0 bias).
    """
    def __init__(self, n_heads: int, gamma_min: float = 0.1, gamma_max: float = 10.0):
        super().__init__()
        gammas = torch.logspace(math.log10(gamma_min), math.log10(gamma_max), n_heads)
        self.raw_gamma = nn.Parameter(torch.log(torch.expm1(gammas)))  # inverse-softplus

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        gamma = F.softplus(self.raw_gamma)                   # (H,) > 0
        D2 = (D ** 2).unsqueeze(1)                           # (B, 1, n, n)
        return -gamma.view(1, -1, 1, 1) * D2                 # (B, H, n, n)
