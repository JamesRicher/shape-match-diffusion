"""Input/output ends of the matrix denoiser: token construction and the Sinkhorn readout.

TokenConstructor builds the trunk's input tokens from features, anchor coords, P_t
pull-backs and entropy; SinkhornReadout turns the trunk's output embeddings back into a
doubly-stochastic assignment. See notes/2026-07-08_denoiser_architecture.md.
"""
import torch
import torch.nn as nn

from utils.sinkhorn import log_sinkhorn, safe_log


class TokenConstructor(nn.Module):
    """Build the initial (n, dim) tokens for both shapes, once before the trunk.

    Structured input per shape (same W_tok for both -> pair-swap symmetry):

        y0 = W_tok [ F_Y, A_Y, P_t F_X,  P_t A_X,  rowent(P_t) ]
        x0 = W_tok [ F_X, A_X, P_tᵀF_Y,  P_tᵀA_Y, colent(P_t) ]

    - F: frozen per-point features (n, d_f).
    - A: anchor coordinates = geodesic distances to the first `a` FPS points, sliced
      from the sparse dist submatrix D[:, :a] (intrinsic absolute position). All inputs
      are intrinsic; raw xyz never enters.
    - Pull-backs P_t F_X / P_t A_X: the expected features / anchor-position of each Y
      token's current match; the first Linear can subtract F_Y from P_t F_X to form a
      per-point error signal (denoising = estimate and remove the current error).
    - Entropy: how decided each token's current assignment is (row entropy for Y tokens,
      column entropy for X tokens).

    Auto-anneals with t: near-uniform P_t -> pull-backs ~ global mean (uninformative);
    sharp P_t -> the match's actual features/position.
    """
    def __init__(self, feat_dim: int, dim: int, n_anchors: int = 16, eps: float = 1e-8):
        super().__init__()
        self.n_anchors = n_anchors
        self.eps = eps
        self.in_dim = 2 * feat_dim + 2 * n_anchors + 1
        self.proj = nn.Linear(self.in_dim, dim)              # W_tok, shared across shapes

    def forward(self, F_x: torch.Tensor, F_y: torch.Tensor,
                D_x: torch.Tensor, D_y: torch.Tensor, P_t: torch.Tensor):
        """F_*: (B, n, d_f); D_*: (B, n, n); P_t: (B, n_y, n_x). Returns x0, y0 (B, n, dim)."""
        a = self.n_anchors
        A_x, A_y = D_x[..., :a], D_y[..., :a]                # anchor coords (B, n, a)
        P_tT = P_t.transpose(-1, -2)                         # (B, n_x, n_y)

        # per-token entropy: rows of P_t for Y, columns for X (both sum to 1 -> valid)
        ent = -(P_t * safe_log(P_t, self.eps))              # (B, n_y, n_x)
        row_ent = ent.sum(-1, keepdim=True)                 # (B, n_y, 1)
        col_ent = ent.sum(-2).unsqueeze(-1)                 # (B, n_x, 1)

        y_struct = torch.cat([F_y, A_y, P_t @ F_x,  P_t @ A_x,  row_ent], dim=-1)
        x_struct = torch.cat([F_x, A_x, P_tT @ F_y, P_tT @ A_y, col_ent], dim=-1)
        return self.proj(x_struct), self.proj(y_struct)


class SinkhornReadout(nn.Module):
    """Turn the trunk's output embeddings into a doubly-stochastic assignment.

        L0      = (E_Y W E_Xᵀ)/√d  +  α·log(clamp(P_t, ε))
        P0_pred = Sinkhorn(L0 / τ)

    - Bilinear W: learned symmetric compatibility metric between the two shapes'
      embeddings (generalises the dot product). Symmetric for pair-swap symmetry
      (property 6); zero-init so the matching score fades in.
    - Skip α·log P_t (Route 3): guarantees P_t-dependence (property 4), restores full
      rank (property 7; the bilinear is rank ≤ d), and gives the t→0 "return P_t"
      endpoint. α init 1 so at init L0 = log P_t and Sinkhorn returns P_t exactly
      (a doubly-stochastic matrix is a Sinkhorn fixed point) — identity at init.
    - τ: temperate at training time (Lipschitz control); annealed toward sharp only in
      the sampler, via the forward `tau` override. Compare only post-Sinkhorn
      quantities — the logits are gauge-ambiguous (property 9).
    """
    def __init__(self, dim: int, tau: float = 1.0, n_iters: int = 10, eps: float = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.W = nn.Parameter(torch.zeros(dim, dim))     # bilinear metric, zero-init
        self.alpha = nn.Parameter(torch.tensor(1.0))     # skip strength -> returns P_t at init
        self.tau = tau
        self.n_iters = n_iters
        self.eps = eps

    def logits(self, E_x: torch.Tensor, E_y: torch.Tensor, P_t: torch.Tensor) -> torch.Tensor:
        """Pre-Sinkhorn logits L0 (B, n_y, n_x)."""
        W = self.W + self.W.transpose(-1, -2)            # symmetric -> pair-swap symmetry
        bilinear = (E_y @ W) @ E_x.transpose(-1, -2) * self.scale
        return bilinear + self.alpha * safe_log(P_t, self.eps)

    def forward(self, E_x: torch.Tensor, E_y: torch.Tensor, P_t: torch.Tensor,
                tau: float | None = None) -> torch.Tensor:
        """Returns P0_pred (B, n_y, n_x), doubly stochastic. Pass tau to sharpen in the sampler."""
        tau = self.tau if tau is None else tau
        return log_sinkhorn(self.logits(E_x, E_y, P_t), n_iters=self.n_iters, tau=tau).exp()
