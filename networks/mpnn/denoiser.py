"""MPNNMatrixDenoiser: geodesic-MPNN + state-track denoiser (networks/mpnn design).

Same external contract as MatrixDenoiser — drop-in under MatrixDiffusionModel:

    forward(P_t, F_x, F_y, D_x, D_y, t) -> u0_hat   (B, n_y, n_x) logits

Per call (stateless across diffusion steps — only P_t carries the trajectory):

    c = spine(t);  precompute geodesic kNN graphs + distance RBFs (+ feature-kNN
    candidates for the mpnn cross variant);  u = log P_t;  tokens from
    [features, warp, stats].

    L x block:
        intra(X), intra(Y)      geodesic MPNN, shared instance      (intra_mpnn)
        cross(X<->Y)            'attn' or 'mpnn' by config          (cross_stage)
        u <- StateWrite(...)    feature/affinity write              (state_track)
        u <- soft_project_sym   cheap re-gauge, transpose-symmetric
        u <- u + g(c)*Δ_bp      belief propagation, at each block in bp.at_block
                                (one site unless a list), then re-gauged  (bp_block)
        rewarp(X), rewarp(Y)    refreshed belief back into tokens

    return u  (identity at init: all writes gated to zero -> u0_hat = log P_t).

Properties by construction (verified in networks/mpnn/tests.py): FULL permutation
equivariance (no anchor frame), pair-swap symmetry, size agnosticism, intrinsic
inputs only, identity at init.
"""
import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY
from utils.sinkhorn import safe_log, row_logprob, col_logprob
from networks.denoiser_conditioning import ConditioningSpine
from networks.mpnn.geometry import (RBFEmbed, feature_knn, knn_from_dist,
                                    normalise_knn_dist)
from networks.mpnn.intra_mpnn import IntraMPNNLayer
from networks.mpnn.cross_stage import build_cross_stage
from networks.mpnn.bp_block import BPStage
from networks.mpnn.state_track import (InputEmbed, Rewarp, StateWrite,
                                       soft_project_sym)

# Safety rail on the propagating state. NOT U_LO (-18): that is the feature/RBF grid
# boundary, and railing there would change the CE in healthy training.
U_RAIL = -100.0


@NETWORK_REGISTRY.register()
class MPNNMatrixDenoiser(nn.Module):
    """Denoise P_t -> clean logit matrix via interleaved intra-MPNN / cross / state writes.

    Args:
        feat_dim: per-point input feature dimension (autofilled from data when null).
        dim: token width.
        depth: number of [intra|intra|cross|write] blocks.
        heads: attention heads (attn cross variant only).
        cross_type: 'attn' (dense biased cross-attention) or 'mpnn' (candidate GNN).
        k_intra: geodesic-kNN neighbours for the intra graphs.
        k_feat: static feature-kNN candidates per receiver (mpnn cross only).
        k_state: dynamic state-top-k candidates per receiver (mpnn cross only).
        n_rbf: RBF basis size (distance and state embeddings).
        rbf_max: upper end of the normalised-distance RBF grid (distances are
            divided by the mean k-th-NN radius, so ~[0, 3] covers the graph).
        inner_iters: soft-Sinkhorn iterations per in-block re-gauge.
        mlp_ratio, dropout: FFN settings.
        time_scale: t scaling into the sinusoidal spine.
        bp: belief-propagation settings (networks/mpnn/bp_block.BPStage) or None to
            disable entirely — no parameters, no code path (the default). BP runs once
            per forward, at block `bp.at_block`.
    """

    def __init__(self, feat_dim: int, dim: int, depth: int = 5, heads: int = 4,
                 cross_type: str = "attn", k_intra: int = 12, k_feat: int = 10,
                 k_state: int = 10, n_rbf: int = 16, rbf_max: float = 3.0,
                 inner_iters: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 time_scale: float = 1000.0, bp: dict | None = None,
                 row_stochastic: bool = False):
        super().__init__()
        self.cross_type = cross_type
        self.k_intra = k_intra
        self.k_feat = k_feat
        self.inner_iters = inner_iters
        # Row-stochastic (non doubly-stochastic) variant. Stored so the owning model can read it
        # as the single source of truth; the in-trunk re-gauge and X-side warps switch on it in
        # forward (added in a follow-up). Default False keeps the doubly-stochastic path.
        self.row_stochastic = row_stochastic

        self.spine = ConditioningSpine(dim, time_scale=time_scale)
        self.embed = InputEmbed(feat_dim, dim)
        self.dist_rbf = RBFEmbed(n_rbf, 0.0, rbf_max)

        self.intra = nn.ModuleList(
            IntraMPNNLayer(dim, n_rbf, mlp_ratio, dropout) for _ in range(depth))
        self.cross = nn.ModuleList(
            build_cross_stage(cross_type, dim, heads, k_state, n_rbf, mlp_ratio, dropout)
            for _ in range(depth))
        self.write = nn.ModuleList(StateWrite(dim, n_rbf) for _ in range(depth))
        # no rewarp after the final block: its output tokens are never read again
        self.rewarp = nn.ModuleList(Rewarp(feat_dim, dim) for _ in range(depth - 1))

        # BP subsystem (notes/BP-loop-design.md). It owns its own gated additive write
        # rather than feeding StateWrite's reserved pair-MLP channel: routing it through
        # the pair-MLP would mix BP's contribution into du nonlinearly and destroy the
        # calibrated meaning of beta and g. That channel stays fed with zeros.
        self.bp = BPStage(depth, dim, **bp) if bp else None
        if self.row_stochastic and self.bp is not None:
            raise ValueError("BP is not supported with row_stochastic yet (its bidirectional / "
                             "cycle-gate machinery assumes a doubly-stochastic / bijective map)")

    def _geo(self, D: torch.Tensor):
        """Static per-call intra-graph tensors: kNN indices + RBF-embedded distances."""
        idx, dist = knn_from_dist(D, min(self.k_intra, D.shape[-1] - 1))
        return idx, self.dist_rbf(normalise_knn_dist(dist))

    def _regauge(self, u: torch.Tensor) -> torch.Tensor:
        """Re-gauge the state after a write. Row-stochastic: one row log-softmax, so each row is
        a log-distribution (the maintained invariant Z_j = 1). DS: the transpose-symmetric
        truncated Sinkhorn (pair-swap symmetric). Railed below: log-domain Sinkhorn only shifts
        rows/cols, so it re-centres a runaway dynamic range without bounding it."""
        u = row_logprob(u) if self.row_stochastic else soft_project_sym(u, self.inner_iters)
        return u.clamp_min(U_RAIL)

    def _warps(self, u: torch.Tensor):
        """The two warp matrices read off the current state, (P_y, P_x): P_y (B, n_y, n_x) rows=Y
        for the Y-token warp; P_x (B, n_x, n_y) rows=X for the X-token warp -- both row-summing
        to 1 so warp_stats sees genuine distributions. Row-stochastic: P_y = row_softmax(u),
        P_x = col_softmax(u)^T (the induced X->Y posterior; valid because u is row-normalised).
        DS: both are views of the single (near) doubly-stochastic P = u.exp()."""
        if self.row_stochastic:
            return row_logprob(u).exp(), col_logprob(u).exp().transpose(-1, -2)
        P = u.exp()
        return P, P.transpose(-1, -2)

    def forward(self, P_t: torch.Tensor, F_x: torch.Tensor, F_y: torch.Tensor,
                D_x: torch.Tensor, D_y: torch.Tensor, t: torch.Tensor,
                log_P_t: torch.Tensor | None = None) -> torch.Tensor:
        """P_t: (B, n_y, n_x) projected read-in Π_S(u_t); F_*: (B, n, feat_dim);
        D_*: (B, n, n) geodesics; t: (B,) or scalar. Returns u0_hat logits.

        log_P_t: the same read-in already in log space (pass P_t=None). Skips the exp/log
        round trip, whose safe_log has a 1/x derivative reaching ~1e8 at its 1e-8 floor.
        """
        c = self.spine(t)

        # ---- static per-call precompute ---- #
        idx_x, rbf_x = self._geo(D_x)
        idx_y, rbf_y = self._geo(D_y)
        cand_x = cand_y = None
        if self.cross_type == "mpnn":
            k_f = min(self.k_feat, F_y.shape[1])
            cand_x = feature_knn(F_x, F_y, k_f)          # X receivers <- Y senders
            cand_y = feature_knn(F_y, F_x, k_f)          # Y receivers <- X senders

        # ---- state + tokens ---- #
        u = safe_log(P_t) if log_P_t is None else log_P_t    # gauge-fixed read-in
        # Y warps through rows of P_t; X warps through P_x -- P_t^T under DS, the induced X->Y
        # posterior col_softmax(u)^T under row-stochastic (columns of P_t are not distributions).
        P_y = P_t if log_P_t is None else u.exp()
        P_x = self._warps(u)[1] if self.row_stochastic else P_y.transpose(-1, -2)
        hy = self.embed(F_y, P_y, F_x)                   # rows of P: Y pulls back X
        hx = self.embed(F_x, P_x, F_y)
        geo_x, geo_y = (idx_x, D_x), (idx_y, D_y)
        bp_cache = self.bp.init_cache(F_x, F_y, geo_x, geo_y) if self.bp else None

        # ---- blocks ---- #
        for i, (intra, cross, write) in enumerate(zip(self.intra, self.cross, self.write)):
            hx = intra(hx, idx_x, rbf_x, c)
            hy = intra(hy, idx_y, rbf_y, c)
            hx, hy = cross(hx, hy, u, c, cand_x=cand_x, cand_y=cand_y)

            u = write(hx, hy, u, c)
            # Re-gauge BEFORE BP, not after: BP's unary is beta*u with beta calibrated as a
            # scale on a LOG-PROBABILITY, but StateWrite returns u + gate(c)*du -- off-gauge by
            # a gate-driven, hence t-dependent, amount. Reading that directly made beta absorb
            # a moving gauge error on top of its actual job.
            u = self._regauge(u)
            if self.bp is not None and i in self.bp.at_blocks:
                u = self.bp(i, u, bp_cache, F_x, F_y, geo_x, geo_y, c)
                u = self._regauge(u)                     # BP's additive write, likewise

            if i < len(self.rewarp):                     # refreshed belief bridge
                P_y, P_x = self._warps(u)
                hy = self.rewarp[i](hy, P_y, F_x)
                hx = self.rewarp[i](hx, P_x, F_y)

        return u
