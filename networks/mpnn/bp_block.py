"""In-loop belief propagation: the learned wrapper around `belief_prop.bp_delta`.

Design in notes/BP-loop-design.md, revised after the gate audit of the 512/1024 runs.
BP runs ONCE PER DENOISER FORWARD — i.e. once per DDIM step — at block index
`at_block` of the trunk, and writes its own gated residual onto the assignment state:

    Δ  = ½ ( Δ_Y  +  Δ_Xᵀ )                  bidirectional (+ optional cycle gate)
    u  ← u + g(c)·Δ

WHY ONE CALL (was: one per block, i.e. 6). Profiling every trained BP checkpoint
showed the stack had already collapsed itself: measured on the EFFECTIVE write
g·(1 + tanh(b/2)) rather than g alone, the top block carried 48-97% of the total
(median ~78%) and was block 0 or 1 in 7 of 8 runs, while blocks 2-5 were switched off
(effective write < 0.05). Six calls were being paid for and roughly one was being
used. Collapsing to one also deletes the cross-block exclusion machinery outright:
with a single call the residual unary is β·u by construction, so the `u_bp`
accumulator, `track_decay`, and the Sinkhorn-gauge leak they were built to manage
all disappear. Across sampler steps BP's earlier write still re-enters through P_t,
exactly as in the post-process.

Two learned quantities, both driven by the conditioning spine's c (so both are
functions of the diffusion timestep) and both initialised so the block is EXACTLY
the identity at init:

    β(c) = β_min·(β_max/β_min)^σ(W_β c + b_β)   unary scale, BOUNDED, β(0) = beta_init
    g(c) = g_scale·(W_g c + b_g)                output gate, W_g zero-init -> g = 0

β IS BOUNDED because both ends of its range destroy the mixing of evidence that is
BP's entire function, and the unbounded softplus ran to both: trained checkpoints
held β = 0.000 (dead — flat unaries, so the belief is decided by geometry and slack
alone, ignoring the state) and β = 9.6 (saturated — the pairwise log-potential is
bounded in [-delta, 0] = [-4, 0], so a unary this large can never be overturned and BP
just re-emits its input). The map is a scaled/shifted sigmoid in LOG space: β is a
scale parameter, so 0.2->0.4 and 2->4 should be equal-sized moves, and log spacing
puts beta_init near the middle of the sigmoid instead of down in its tail. A hard
clamp is the wrong tool — zero gradient outside the range makes saturation permanent,
which is the failure being fixed.

Everything else (sigma, delta, tau, alpha, slack, sweep count) is a FIXED scalar from
`diagnostics/bp_sigma_calibration.py` + `bp_postprocess_sweep.py` — the propagation rule
stays unlearned, which is what separates this from another MPNN stage and what makes the
test-time sweep-count scaling diagnostic meaningful.

Orientation: u is (B, n_y, n_x), rows = Y (the repo's P_t convention). The "Y pass"
puts variables on Y (message graph and edge lengths from D_y, labels and label metric
from X); the "X pass" is the same call with every argument swapped, which is exactly
what makes the pair Δ swap-symmetric — on its own, with no gate involved.
"""
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from networks.mpnn.belief_prop import bp_delta, build_candidate_sets
from networks.mpnn.geometry import feature_knn, knn_from_dist


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _zero_head(dim: int, out: int) -> nn.Sequential:
    """SiLU + zero-init Linear, matching StateWrite's gate convention."""
    head = nn.Sequential(nn.SiLU(), nn.Linear(dim, out))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class BPBlock(nn.Module):
    """One BP call: two passes, an optional agreement gate, and the output gate.

    V_MAX caps the agreement gate's slope; see `_gate` for why it is needed once w is
    renormalised. 2.0 leaves the gate able to reach w ~ [0.1, 1.9] for a vertex 3 sigma
    from the mean round-trip score, which is the full useful range of a trust weight.

    Args:
        dim: conditioning width (c).
        k_logit, k_feat: candidate set halves (top-k state ∪ feature-kNN).
        sigma, delta, tau, alpha, slack: the fixed BP scalars.
        n_sweeps: sweeps per call (mutable at eval — see BPStage.set_n_sweeps).
        beta_init: unary scale at init; must lie strictly inside (beta_min, beta_max).
        beta_min, beta_max: the bounds of the learned unary scale (see module docstring).
        g_scale: initial slope of the output gate (dg/d(head) at init), so the learned
            head works in O(1) units against a post-process optimum of g ~ 4-16.
        g_max: smooth bound on the output gate, |g| < g_max via g = g_max·tanh(raw/g_max)
            (see `forward`). Default 20 leaves the 4-16 band in the near-linear region and
            caps only the runaway tail; the gate stays differentiable everywhere.
        bidirectional: run the X pass too and combine ½(Δ_Y + Δ_Xᵀ). Required for
            pair-swap symmetry.
        cycle_gate: enable the per-vertex round-trip agreement gate w. OFF by default:
            in every trained checkpoint its learned v was ~0 (|v| < 0.4 in 44/48 blocks),
            so it never used the round-trip signal to redistribute trust — it used its
            global bias as an off-switch instead. The bias is gone (see `_gate`), which
            leaves the mechanism unstudied rather than vindicated, so it must be opted in.
        checkpoint: recompute each BP pass in the backward instead of storing it
            (the sweeps' (B,n,k,Kc+1,Kc+1) transport tensors dominate activation memory).
            With one call instead of six this is ~6x less pressing than it was.
    """

    # Cap on the agreement gate's slope (see `_gate`). Renormalising to mean 1 makes an
    # exact [0, 2] bound impossible, but with |v| <= 2 the gate is bounded at max w ~ 2.4
    # even for adversarial score distributions and unbounded head output — versus no
    # bound at all without the cap (measured 4.3, growing with v).
    V_MAX = 2.0

    def __init__(self, dim: int, *, k_logit: int = 10, k_feat: int = 10,
                 sigma: float = 0.05, delta: float = 4.0, tau: float = 1.0,
                 alpha: float = 0.5, slack: float = -4.0, n_sweeps: int = 8,
                 beta_init: float = 0.5, beta_min: float = 0.1, beta_max: float = 4.0,
                 g_scale: float = 4.0, g_max: float = 20.0, bidirectional: bool = True,
                 cycle_gate: bool = False, checkpoint: bool = False):
        super().__init__()
        if cycle_gate and not bidirectional:
            raise ValueError("cycle_gate needs bidirectional=True (it compares the two "
                             "directions' beliefs)")
        if not 0.0 < beta_min < beta_init < beta_max:
            raise ValueError(f"need 0 < beta_min < beta_init < beta_max, got "
                             f"{beta_min} / {beta_init} / {beta_max}")
        self.k_logit, self.k_feat = k_logit, k_feat
        self.sigma, self.delta, self.tau = sigma, delta, tau
        self.alpha, self.slack = alpha, slack
        self.n_sweeps = n_sweeps
        self.beta_min, self.beta_max = beta_min, beta_max
        self.g_scale, self.g_max = g_scale, g_max
        self.bidirectional, self.cycle_gate = bidirectional, cycle_gate
        self.checkpoint = checkpoint

        # log-space sigmoid: beta = beta_min * (beta_max/beta_min)^sigmoid(head(c)).
        # The bias places beta_init exactly at init; the head weight is zero-init, so
        # beta is constant in t until training moves it.
        self.beta_head = _zero_head(dim, 1)
        frac = math.log(beta_init / beta_min) / math.log(beta_max / beta_min)
        self.beta_head[-1].bias.data.fill_(_logit(frac))
        self.g_head = _zero_head(dim, 1)
        # d = 1: one coefficient on the standardised round-trip mass. There is NO bias
        # term -- see `_gate`.
        self.gate_head = _zero_head(dim, 1) if cycle_gate else None

    def _beta(self, c: torch.Tensor) -> torch.Tensor:
        """Bounded unary scale (B, 1), geometrically spaced across [beta_min, beta_max]."""
        s = torch.sigmoid(self.beta_head(c))
        return self.beta_min * (self.beta_max / self.beta_min) ** s

    # ------------------------------------------------------------------ #
    # one directed pass
    # ------------------------------------------------------------------ #
    def _pass(self, theta, logits, feats_src, feats_tgt, D_src, D_tgt, nbr, feat_cand):
        """BP with variables on the shape `logits`' rows live on.

        theta: (B, n_src, n_tgt) unary field β·u in this orientation; logits:
        (B, n_src, n_tgt) the current state, from which candidates are drawn. Returns
        (Δ, msg_delta, dense belief). All-tensor in/out so it can go through
        `checkpoint`. `dense` is the (n_src, n_tgt) scatter the agreement gate needs and
        is built ONLY when that gate is on — it is two n^2 allocations per call that
        would otherwise be computed and discarded.
        """
        cand_idx, cand_mask = build_candidate_sets(
            logits, feats_src, feats_tgt, self.k_logit, self.k_feat, feat_cand=feat_cand)
        delta_mat, info = bp_delta(
            theta, cand_idx, cand_mask, nbr, D_src, D_tgt, sigma=self.sigma,
            delta=self.delta, tau=self.tau, alpha=self.alpha, s=self.slack,
            n_sweeps=self.n_sweeps, return_info=True)
        if not self.cycle_gate:
            # msg_delta is the convergence monitor: the max message change on the final
            # sweep. Free (already computed) and the only evidence that n_sweeps is set
            # anywhere near right.
            return delta_mat, info["msg_delta"], delta_mat.new_zeros(())
        # dense belief for the agreement gate: scatter P(label | variable) onto the
        # candidates (slack dropped, invalid slots zeroed so duplicates add nothing)
        Kc = cand_idx.shape[-1]
        p = info["belief"][..., :Kc].exp().masked_fill(~cand_mask, 0.0)
        dense = torch.zeros_like(delta_mat)
        dense.scatter_add_(-1, cand_idx, p)
        return delta_mat, info["msg_delta"], dense

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

            v = V_MAX·tanh(head(c));  w = 1 + tanh(0.5·v·z(a));  w /= mean(w)

        The gate is a REDISTRIBUTION of trust (which vertices) and must never compete
        with g for the global magnitude (how much). The previous parameterisation added
        a learned bias b inside the tanh; because b is vertex-independent it acted as a
        global multiplier, and that is exactly what every trained checkpoint used it for
        — b fell to [-8, -2], driving w -> 0 and removing 29-66% of the write while g
        stayed large. Two changes close the loophole: b is gone, and w is divided by its
        mean over vertices. Dropping b alone is not enough, because tanh is odd and the
        round-trip score is left-skewed (bounded above by 1 − slack), so a residual
        global component survives; the renormalisation makes "cannot compete with g" a
        property rather than an approximation. The removed scalar is absorbed by g,
        which is where global magnitude belongs.

        v IS BOUNDED, which the pre-renormalisation gate did not need. Unbounded v lets
        the tanh saturate for nearly every vertex at once (w -> 0 almost everywhere);
        dividing by that small mean then rescales the survivors without limit — measured
        max w = 4.3, i.e. the old [0, 2] guarantee was gone and the gate had recovered a
        magnitude channel through the back door. Capping |v| keeps saturation local, so
        mean(w) stays near 1 and the renormalisation is a small correction rather than a
        large rescale. Zero-init head -> v = 0 -> w ≡ 1 exactly.
        """
        z = (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-6)
        v = self.V_MAX * torch.tanh(self.gate_head(c))           # (B,1), |v| < V_MAX
        w = 1.0 + torch.tanh(0.5 * v * z)                        # (B,1)*(B,n) -> (B,n)
        return w / w.mean(1, keepdim=True).clamp_min(1e-3)

    # ------------------------------------------------------------------ #
    def forward(self, u, F_x, F_y, geo_x, geo_y, c, cand_x=None, cand_y=None):
        """u: (B, n_y, n_x). geo_*: (nbr (B,n,k), D (B,n,n)) for that shape.
        cand_x/cand_y: cached feature-kNN halves (X->Y and Y->X label indices).
        Returns (gated write g·Δ, stats). The caller adds the write to u."""
        nbr_x, D_x = geo_x
        nbr_y, D_y = geo_y
        beta = self._beta(c).unsqueeze(-1)                       # (B,1,1)
        theta = beta * u                                         # no u_bp: one call

        d_y, md_y, A = self._run(theta, u, F_y, F_x, D_y, D_x, nbr_y, cand_y)
        md = md_y
        if not self.bidirectional:
            delta = d_y
        else:
            tT = theta.transpose(-1, -2).contiguous()
            uT = u.transpose(-1, -2).contiguous()
            d_x, md_x, C = self._run(tT, uT, F_x, F_y, D_x, D_y, nbr_x, cand_x)
            md = 0.5 * (md_y.mean() + md_x.mean())
            if self.cycle_gate:
                a_y, a_x = self._round_trip(A, C)
                d_y = self._gate(a_y, c).unsqueeze(-1) * d_y     # scales rows
                d_x = self._gate(a_x, c).unsqueeze(-1) * d_x     # -> scales columns
            delta = 0.5 * (d_y + d_x.transpose(-1, -2))

        # Output gate, smoothly bounded to (-g_max, g_max). The write feeds u before the
        # Sinkhorn read-in, so an unbounded g (zero-init but free to grow) was a runaway
        # path into log_sinkhorn -> nan. tanh keeps a live gradient at every value (a hard
        # clamp would zero it past the bound and freeze g there), and the g_scale/g_max
        # framing preserves the old behaviour where it matters: at init g_head=0 -> g=0
        # (exact identity), and the initial slope dg/d(head) is still g_scale, so the O(1)
        # head units and the calibrated 4-16 operating band are unchanged in the near-
        # linear region -- g_max only caps the tail.
        raw = self.g_scale * self.g_head(c).unsqueeze(-1)        # (B,1,1)
        g = self.g_max * torch.tanh(raw / self.g_max)
        return g * delta, {"beta": beta.mean(), "g": g.mean(), "msg_delta": md.mean()}


class BPStage(nn.Module):
    """The BP subsystem: one BPBlock, its position in the trunk, and the static caches.

    Kept as one self-contained module so the denoiser's integration is three lines and
    `bp: null` (the default) leaves zero parameters and zero code path behind.

    Usage inside a denoiser forward:

        cache = bp.init_cache(F_x, F_y, geo_x, geo_y)        # once
        ...
        u = bp(i, u, cache, F_x, F_y, geo_x, geo_y, c)       # per block; no-op unless
                                                             # i == at_block
    """

    def __init__(self, depth: int, dim: int, *, at_block: int = 0,
                 k_graph: int | None = None, **block_kwargs):
        super().__init__()
        if not 0 <= at_block < depth:
            raise ValueError(f"at_block must be in [0, {depth}), got {at_block}")
        self.at_block = at_block
        self.block = BPBlock(dim, **block_kwargs)
        self.k_graph = k_graph          # None -> reuse the trunk's k_intra graph
        self.k_feat = self.block.k_feat
        self.stats: dict[str, float] = {}

    def set_n_sweeps(self, n: int) -> None:
        """Test-time inference-effort knob: train at n_sweeps, evaluate at 1/2/4/8/16.
        Monotone improvement past the trained count is the evidence BP performs
        inference rather than smoothing (notes/BP-idea.md diagnostics). With one call
        the reading is clean — previously six interacting calls made it uninterpretable.
        """
        self.block.n_sweeps = n

    @torch.no_grad()
    def _graph(self, nbr, D):
        if self.k_graph is None:
            return nbr
        return knn_from_dist(D, min(self.k_graph, D.shape[-1] - 1))[0]

    def init_cache(self, F_x, F_y, geo_x, geo_y) -> dict:
        """Per-forward statics: feature-kNN candidate halves and the BP message graphs."""
        k_f = min(self.k_feat, min(F_x.shape[1], F_y.shape[1]))
        return {"cand_y": feature_knn(F_y, F_x, k_f),    # Y variables -> X labels
                "cand_x": feature_knn(F_x, F_y, k_f),    # X variables -> Y labels
                "nbr_x": self._graph(geo_x[0], geo_x[1]),
                "nbr_y": self._graph(geo_y[0], geo_y[1])}

    def forward(self, i: int, u, cache: dict, F_x, F_y, geo_x, geo_y, c):
        """Run BP and apply its write, but only at block `at_block`; else pass u through."""
        if i != self.at_block:
            return u
        d, stats = self.block(u, F_x, F_y,
                              (cache["nbr_x"], geo_x[1]), (cache["nbr_y"], geo_y[1]), c,
                              cand_x=cache["cand_x"], cand_y=cache["cand_y"])
        if not self.training:
            # .item() syncs, so this is eval-only; validation runs every epoch, which is
            # the cadence these need to be tracked at anyway.
            self.stats = {"bp/write_abs": d.abs().mean().item(),
                          **{f"bp/{k}": v.item() for k, v in stats.items()}}
        return u + d
