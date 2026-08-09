"""In-loop belief propagation: the learned wrapper around `belief_prop.bp_delta`.

Design in notes/BP-loop-design.md. BP runs ONCE PER BLOCK of the MPNN denoiser and
writes its own gated residual onto the assignment state:

    Δ  = ½ ( diag(w_y)·Δ_Y  +  Δ_Xᵀ·diag(w_x) )        bidirectional, cycle-gated
    u  ← u + g(c)·Δ
    u_bp ← ρ·u_bp + g(c)·Δ                             BP's own stream, tracked

Three learned quantities, all per block, all driven by the conditioning spine's c
(so they are functions of the diffusion timestep) and all initialised so the block is
EXACTLY the identity at init:

    β(c) = softplus(b_β + W_β c)      unary scale     W_β zero-init, β(0) = beta_init
    g(c) = g_scale·(W_g c + b_g)      output gate     zero-init  -> g = 0
    w_j  = 1 + tanh(½(v(c)·z_j + b))  agreement gate  zero-init  -> w ≡ 1

Everything else (sigma, delta, tau, alpha, slack, sweep count) is a FIXED scalar from
`diagnostics/bp_sigma_calibration.py` + `bp_postprocess_sweep.py` — the propagation rule
stays unlearned, which is what separates this from another MPNN stage and what makes the
test-time sweep-count scaling diagnostic meaningful.

The two exclusion mechanisms (notes/bp_implementation_and_exclusion.md):
  - cavity N(i)\\j, inside `bp_delta` — the within-run, across-edges exclusion;
  - residual unaries β·(u − u_bp), here — the across-blocks exclusion, so BP never
    re-consumes its own earlier writes as fresh evidence.

Orientation: u is (B, n_y, n_x), rows = Y (the repo's P_t convention). The "Y pass"
puts variables on Y (message graph and edge lengths from D_y, labels and label metric
from X); the "X pass" is the same call with every argument swapped, which is exactly
what makes the pair Δ swap-symmetric.
"""
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from networks.mpnn.belief_prop import bp_delta, build_candidate_sets
from networks.mpnn.geometry import feature_knn, knn_from_dist


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def _zero_head(dim: int, out: int) -> nn.Sequential:
    """SiLU + zero-init Linear, matching StateWrite's gate convention."""
    head = nn.Sequential(nn.SiLU(), nn.Linear(dim, out))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class BPBlock(nn.Module):
    """One block's worth of BP: two passes, an agreement gate, and the output gate.

    Args:
        dim: conditioning width (c).
        k_logit, k_feat: candidate set halves (top-k state ∪ feature-kNN).
        sigma, delta, tau, alpha, slack: the fixed BP scalars.
        n_sweeps: sweeps per call (mutable at eval — see BPStack.set_n_sweeps).
        beta_init: unary scale at init.
        g_scale: fixed multiplier on the output gate, so the learned head works in
            O(1) units against a post-process optimum of g ~ 4-16.
        bidirectional: run the X pass too and combine ½(Δ_Y + Δ_Xᵀ). Required for
            pair-swap symmetry and for the agreement gate (which needs both opinions).
        cycle_gate: enable the cycle-consistency gate w.
        checkpoint: recompute each BP pass in the backward instead of storing it
            (the sweeps' (B,n,k,Kc+1,Kc+1) transport tensors dominate activation memory).
    """

    def __init__(self, dim: int, *, k_logit: int = 10, k_feat: int = 10,
                 sigma: float = 0.05, delta: float = 4.0, tau: float = 1.0,
                 alpha: float = 0.5, slack: float = -4.0, n_sweeps: int = 3,
                 beta_init: float = 0.5, g_scale: float = 4.0,
                 bidirectional: bool = True, cycle_gate: bool = True,
                 checkpoint: bool = False):
        super().__init__()
        if cycle_gate and not bidirectional:
            raise ValueError("cycle_gate needs bidirectional=True (it compares the two "
                             "directions' beliefs)")
        self.k_logit, self.k_feat = k_logit, k_feat
        self.sigma, self.delta, self.tau = sigma, delta, tau
        self.alpha, self.slack = alpha, slack
        self.n_sweeps = n_sweeps
        self.g_scale = g_scale
        self.bidirectional, self.cycle_gate = bidirectional, cycle_gate
        self.checkpoint = checkpoint

        self.beta_head = _zero_head(dim, 1)
        self.beta_head[-1].bias.data.fill_(_inv_softplus(beta_init))
        self.g_head = _zero_head(dim, 1)
        # d = 1: one coefficient on the standardised round-trip mass, plus a bias.
        # Widening to the other per-vertex BP statistics is a change of `out` here and
        # of `_vertex_stats` below; nothing else moves.
        self.gate_head = _zero_head(dim, 2) if cycle_gate else None

    # ------------------------------------------------------------------ #
    # one directed pass
    # ------------------------------------------------------------------ #
    def _pass(self, theta, logits, feats_src, feats_tgt, D_src, D_tgt, nbr, feat_cand):
        """BP with variables on the shape `logits`' rows live on.

        theta: (B, n_src, n_tgt) residual unary field β·(u − u_bp) in this orientation;
        logits: (B, n_src, n_tgt) the CURRENT state (candidates come from the actual
        belief, only the unary is residual). Returns (Δ, dense belief), both
        (B, n_src, n_tgt). All-tensor in/out so it can go through `checkpoint`.
        """
        cand_idx, cand_mask = build_candidate_sets(
            logits, feats_src, feats_tgt, self.k_logit, self.k_feat, feat_cand=feat_cand)
        delta_mat, info = bp_delta(
            theta, cand_idx, cand_mask, nbr, D_src, D_tgt, sigma=self.sigma,
            delta=self.delta, tau=self.tau, alpha=self.alpha, s=self.slack,
            n_sweeps=self.n_sweeps, return_info=True)
        # dense belief for the agreement gate: scatter P(label | variable) onto the
        # candidates (slack dropped, invalid slots zeroed so duplicates add nothing)
        Kc = cand_idx.shape[-1]
        p = info["belief"][..., :Kc].exp().masked_fill(~cand_mask, 0.0)
        dense = torch.zeros_like(delta_mat)
        dense.scatter_add_(-1, cand_idx, p)
        return delta_mat, dense

    def _run(self, *args):
        if self.checkpoint and torch.is_grad_enabled():
            return checkpoint(self._pass, *args, use_reentrant=False)
        return self._pass(*args)

    # ------------------------------------------------------------------ #
    # cycle consistency
    # ------------------------------------------------------------------ #
    @staticmethod
    def _round_trip(A: torch.Tensor, C: torch.Tensor):
        """Per-vertex cycle consistency from the two directions' dense beliefs.

        A (B, n_y, n_x): Y's belief over X labels. C (B, n_x, n_y): X's over Y.
        M = A ⊙ Cᵀ is the soft mutual-nomination matrix; its row sum is
        diag(AC) — the probability that a walk Y -> X -> Y returns home — and its
        column sum is diag(CA). Forming M avoids ever materialising the (n,n)
        round-trip matmuls. Rows sum to 1 − slack, so abstention deflates the score.
        """
        M = A * C.transpose(-1, -2)                              # (B, n_y, n_x)
        return M.sum(-1), M.sum(-2)                              # (B, n_y), (B, n_x)

    def _gate(self, a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Per-vertex trust multiplier w (B, n) from the agreement score a (B, n).

        Standardised across vertices, so the gate is a scale-free REDISTRIBUTION of
        trust (which vertices) and never competes with g for the global magnitude
        (how much). Centred at 1: below-average vertices are suppressed toward 0,
        above-average amplified toward 2. Zero-init head -> w ≡ 1 exactly.
        """
        z = (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-6)
        vb = self.gate_head(c)                                   # (B, 2)
        return 1.0 + torch.tanh(0.5 * (vb[:, :1] * z + vb[:, 1:]))

    # ------------------------------------------------------------------ #
    def forward(self, u, u_bp, F_x, F_y, geo_x, geo_y, c, cand_x=None, cand_y=None):
        """u, u_bp: (B, n_y, n_x). geo_*: (nbr (B,n,k), D (B,n,n)) for that shape.
        cand_x/cand_y: cached feature-kNN halves (X->Y and Y->X label indices).
        Returns the GATED write g·Δ (B, n_y, n_x) — the caller adds it to both u and
        the u_bp accumulator, so what is tracked is exactly what was written."""
        nbr_x, D_x = geo_x
        nbr_y, D_y = geo_y
        beta = nn.functional.softplus(self.beta_head(c)).unsqueeze(-1)    # (B,1,1)
        theta = beta * (u - u_bp)                                        # residual unary

        d_y, A = self._run(theta, u, F_y, F_x, D_y, D_x, nbr_y, cand_y)
        if not self.bidirectional:
            delta = d_y
        else:
            tT = theta.transpose(-1, -2).contiguous()
            uT = u.transpose(-1, -2).contiguous()
            d_x, C = self._run(tT, uT, F_x, F_y, D_x, D_y, nbr_x, cand_x)
            if self.cycle_gate:
                a_y, a_x = self._round_trip(A, C)
                d_y = self._gate(a_y, c).unsqueeze(-1) * d_y     # scales rows
                d_x = self._gate(a_x, c).unsqueeze(-1) * d_x     # -> scales columns
            delta = 0.5 * (d_y + d_x.transpose(-1, -2))

        g = self.g_scale * self.g_head(c).unsqueeze(-1)          # (B,1,1)
        return g * delta


class BPStack(nn.Module):
    """The whole BP subsystem: per-block modules + the u_bp bookkeeping + caches.

    Kept as one self-contained module so the denoiser's integration is three lines and
    `bp: null` (the default) leaves zero parameters and zero code path behind.

    Usage inside a denoiser forward:

        state = bp.init_state(u, F_x, F_y, geo_x, geo_y)     # once
        ...
        u, state = bp(i, u, state, F_x, F_y, geo_x, geo_y, c)   # per block
    """

    def __init__(self, depth: int, dim: int, *, track_decay: float = 1.0,
                 k_graph: int | None = None, **block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(BPBlock(dim, **block_kwargs) for _ in range(depth))
        self.track_decay = track_decay
        self.k_graph = k_graph          # None -> reuse the trunk's k_intra graph
        self.k_feat = self.blocks[0].k_feat
        self.stats: dict[str, float] = {}

    def set_n_sweeps(self, n: int) -> None:
        """Test-time inference-effort knob: train at 3, evaluate at 1/2/4/8/16.
        Monotone improvement past the trained count is the evidence BP performs
        inference rather than smoothing (notes/BP-idea.md diagnostics)."""
        for b in self.blocks:
            b.n_sweeps = n

    @torch.no_grad()
    def _graph(self, nbr, D):
        if self.k_graph is None:
            return nbr
        return knn_from_dist(D, min(self.k_graph, D.shape[-1] - 1))[0]

    def init_state(self, u, F_x, F_y, geo_x, geo_y) -> dict:
        """Per-forward state: the u_bp accumulator plus everything static across blocks.

        u_bp starts at zero every call — the denoiser is stateless across diffusion
        steps, so BP's history is per-forward. (Across sampler steps BP's earlier write
        legitimately re-enters through P_t, exactly as in the post-process.)
        """
        k_f = min(self.k_feat, min(F_x.shape[1], F_y.shape[1]))
        return {"u_bp": torch.zeros_like(u),
                "cand_y": feature_knn(F_y, F_x, k_f),    # Y variables -> X labels
                "cand_x": feature_knn(F_x, F_y, k_f),    # X variables -> Y labels
                "nbr_x": self._graph(geo_x[0], geo_x[1]),
                "nbr_y": self._graph(geo_y[0], geo_y[1])}

    def forward(self, i: int, u, state: dict, F_x, F_y, geo_x, geo_y, c):
        """Run block i's BP, apply its write to u, and update the tracked stream."""
        d = self.blocks[i](u, state["u_bp"], F_x, F_y,
                           (state["nbr_x"], geo_x[1]), (state["nbr_y"], geo_y[1]), c,
                           cand_x=state["cand_x"], cand_y=state["cand_y"])
        state = dict(state, u_bp=self.track_decay * state["u_bp"] + d)
        if not self.training:
            self.stats[f"bp/write_abs_{i}"] = d.abs().mean().item()
        return u + d, state
