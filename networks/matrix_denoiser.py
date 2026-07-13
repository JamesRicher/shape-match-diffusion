"""The matrix denoiser: predicts the clean logit matrix u0 from a noised assignment P_t.

Assembles the primitives into one network:

    TokenConstructor  ->  L x [ intra(x) | intra(y) | inter(x, y) ]  ->  LogitReadout

- Conditioning: c = ConditioningSpine(t) drives AdaLN-Zero in every block.
- Intra bias: one GeodesicKernelBias per layer, shared across both shapes (called
  once per shape with that shape's geodesic matrix D).
- Inter bias: one LogAssignmentBias per layer, the per-head log(P_t) prior.
- P_t is the projected read-in of the diffusion state (P_t = Π_S(u_t), computed upstream
  in the logit-space scheme -- notes/noising.md); it enters the tokens, every inter bias,
  and the readout skip. The denoiser never sees the raw logits u_t, only P_t.

Predict-x0 in logit space: forward returns u0_hat, an unconstrained logit matrix.
Projection to the Birkhoff polytope (Sinkhorn) is external -- at the loss (row-CE on
Π_S(u0_hat)) and in the sampler.

Identity at init: AdaLN gates zero, bilinear W zero, alpha=1, bias gammas zero, so the
trunk is inert and u0_hat = log P_t; projected, Π_S(log P_t) = P_t. Symmetries
(permutation equivariance, size agnosticism, pair-swap) hold by construction -- see
networks/matrix_denoiser_tests.py.
"""
import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY
from networks.denoiser_blocks import AdaLNIntraShapeBlock, AdaLNInterShapeBlock
from networks.denoiser_conditioning import ConditioningSpine, LogAssignmentBias, GeodesicKernelBias
from networks.denoiser_io import TokenConstructor, LogitReadout


@NETWORK_REGISTRY.register()
class MatrixDenoiser(nn.Module):
    """Denoise a doubly-stochastic assignment P_t -> P0_pred on the Birkhoff polytope.

    Args:
        feat_dim: per-point input feature dimension (F_x, F_y).
        dim: transformer latent width.
        heads: attention heads (must divide dim).
        depth: number of [intra | intra | inter] layers.
        n_anchors: number of geodesic anchor coordinates in the tokens.
        mlp_ratio: FFN hidden expansion.
        dropout: dropout probability.
        time_scale: t scaling into the sinusoidal spine (see ConditioningSpine).
    """
    def __init__(self, feat_dim: int, dim: int, heads: int, depth: int = 4,
                 n_anchors: int = 16, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 time_scale: float = 1000.0):
        super().__init__()
        self.spine = ConditioningSpine(dim, time_scale=time_scale)
        self.tokens = TokenConstructor(feat_dim, dim, n_anchors=n_anchors)

        # one intra block per layer, reused for both shapes => intra weight sharing
        self.intra = nn.ModuleList(
            AdaLNIntraShapeBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth))
        self.inter = nn.ModuleList(
            AdaLNInterShapeBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth))
        # per-layer bias generators (intra shared across shapes, inter per layer)
        self.intra_bias = nn.ModuleList(GeodesicKernelBias(heads) for _ in range(depth))
        self.inter_bias = nn.ModuleList(LogAssignmentBias(heads) for _ in range(depth))

        self.out_norm = nn.LayerNorm(dim)  # shared across shapes => symmetric
        self.readout = LogitReadout(dim)

    def forward(self, P_t: torch.Tensor, feat_x: torch.Tensor, feat_y: torch.Tensor,
                D_x: torch.Tensor, D_y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        P_t:    (B, n_y, n_x) projected diffusion state Π_S(u_t), doubly stochastic.
        feat_*: (B, n, feat_dim) frozen per-point features.
        D_*:    (B, n, n) intrinsic geodesic distance matrices.
        t:      (B,) or scalar diffusion time (step/T in [0, 1], or integer step).
        returns u0_hat (B, n_y, n_x), the predicted clean logit matrix. Project externally
        (Π_S) for the loss / sampler.
        """
        c = self.spine(t)                                       # (B, dim)
        x, y = self.tokens(feat_x, feat_y, D_x, D_y, P_t)       # (B, n, dim)

        for intra, inter, ib, pb in zip(self.intra, self.inter, self.intra_bias, self.inter_bias):
            bx, by = ib(D_x), ib(D_y)                           # (B, H, n, n) per shape
            xy_bias = pb(P_t)                                   # (B, H, n_x, n_y) X<-Y
            x = intra(x, c, bias=bx)
            y = intra(y, c, bias=by)                            # same instance
            x, y = inter(x, y, c, bias=xy_bias)

        x, y = self.out_norm(x), self.out_norm(y)
        return self.readout(x, y, P_t)
