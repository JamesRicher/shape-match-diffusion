"""Sinkhorn + Birkhoff-polytope utilities for the sparse matrix diffusion matcher.

Soft assignment matrices P have shape (..., R, C) with R = n_y rows, C = n_x cols.

Noising is DDPM in the logit domain (notes/noising.md): the diffusion variable is an
unconstrained logit matrix u; a doubly-stochastic P = Π_S(u) = log_sinkhorn(u).exp() is
only ever the projected view. logit_target embeds the GT permutation as a finite logit u0,
q_sample runs the variance-preserving forward marginal on u0, and cosine_alpha_bar is the
schedule. Time convention: t=0 is clean data, t=1 is noise.
"""
import math
from typing import Optional

import torch


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """log with a floor so a zero entry gives a large finite value, not -inf.

    Used wherever a log of an assignment matrix feeds a bias/logit (a -inf bias is a
    dead, zero-gradient edge)."""
    return torch.log(x.clamp_min(eps))


def log_sinkhorn(
    log_alpha: torch.Tensor,
    n_iters: int = 10,
    tau: float = 1.0,
    log_mu: Optional[torch.Tensor] = None,
    log_nu: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project a score matrix onto the Birkhoff polytope in the log domain.

    Alternating row/col log-normalisation on log_alpha / tau, differentiable by
    unrolling and stable at small tau (logsumexp, not exponentiated values).

    Args:
        log_alpha: (..., R, C) logits (log of the unnormalised assignment).
        n_iters: Sinkhorn iterations (each = one row + one col pass).
        tau: temperature; smaller sharpens toward a hard permutation.
        log_mu: (..., R) target log row-marginals; default 0 (unit row sums).
        log_nu: (..., C) target log col-marginals; default 0 (unit col sums).
            Requires exp(log_mu).sum() == exp(log_nu).sum(); the square default is
            the matcher's case.

    Returns log P (..., R, C); the last pass normalises columns, so col marginals are
    exact and row marginals hold to tolerance. Take .exp() for the assignment.
    """
    log_p = log_alpha / tau
    for _ in range(n_iters):
        # rows -> mu
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        if log_mu is not None:
            log_p = log_p + log_mu.unsqueeze(-1)
        # cols -> nu
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
        if log_nu is not None:
            log_p = log_p + log_nu.unsqueeze(-2)
    return log_p


def sample_gumbel(shape, device=None, dtype=torch.float32, generator=None) -> torch.Tensor:
    """i.i.d. standard Gumbel noise -log(-log(U)), U ~ Uniform(0, 1)."""
    u = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    # clamp guards log(0) at the tails of the uniform draw
    u = u.clamp_(min=torch.finfo(dtype).tiny)
    return -torch.log(-torch.log(u))


def sample_doubly_stochastic(
    n_rows: int,
    n_cols: int,
    tau: float = 1.0,
    n_iters: int = 20,
    batch_shape=(),
    device=None,
    dtype=torch.float32,
    generator=None,
) -> torch.Tensor:
    """Random doubly-stochastic matrix via Sinkhorn(Gumbel noise).

    Maximum-entropy sample on the polytope; as tau -> 0 it concentrates on a uniformly
    random permutation (Gumbel-Sinkhorn). A convenient DS prior / test fixture; not the
    forward noise (that is Gaussian in the logit chart — see q_sample).

    Args:
        n_rows, n_cols: matrix shape (n_y, n_x).
        tau: sampling temperature (higher = closer to uniform, lower = sharper).
        n_iters: Sinkhorn iterations.
        batch_shape: leading batch dims, e.g. (B,) or () for a single matrix.

    Returns P_noise (*batch_shape, n_rows, n_cols) in the probability domain.
    """
    g = sample_gumbel((*batch_shape, n_rows, n_cols), device=device, dtype=dtype, generator=generator)
    return log_sinkhorn(g, n_iters=n_iters, tau=tau).exp()


def logit_target(P0: torch.Tensor, eta: float = 0.1, eps: float = 1e-8) -> torch.Tensor:
    """Clean logit target u0 = log((1 - eta)·P0 + eta/m·1) for the logit-space diffusion.

    log(P0) has -inf on the zeros of a (near-)permutation. Label-smoothing toward the
    row-barycenter (a convex mix of two row-stochastic matrices stays row-stochastic, so
    the log is finite) gives a bounded target whose magnitude sets the near-t=0 difficulty.

    Args:
        P0: (..., R, C) ground-truth assignment, rows sum to 1 (a permutation / relaxation).
        eta: smoothing weight in [0.05, 0.2]; smaller = sharper target, larger logits.
            A real hyperparameter — expose in config and tune.
    Returns u0 (..., R, C), an unconstrained logit matrix (Π_S(u0) recovers ~P0).
    """
    m = P0.shape[-1]
    P_tilde = (1.0 - eta) * P0 + eta / m
    return safe_log(P_tilde, eps)


def cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cosine VP schedule ᾱ(t) (Nichol & Dhariwal). t in [0, 1]: ᾱ(0)=1 (clean), ᾱ(1)=0.

    ᾱ(t) = f(t)/f(0), f(u) = cos((u + s)/(1 + s)·π/2)². The small offset s keeps ᾱ from
    dropping too fast near t=0. Accepts a scalar or a batch tensor of times.
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=torch.float32)
    f = lambda u: torch.cos((u + s) / (1.0 + s) * math.pi / 2.0) ** 2
    return f(t) / f(torch.zeros_like(t))


def q_sample(
    u0: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    s: float = 0.008,
) -> torch.Tensor:
    """Forward marginal in logit space: u_t = √ᾱ(t)·u0 + √(1-ᾱ(t))·ε, ε ~ N(0, I).

    Variance-preserving DDPM noising applied entirely in the unconstrained logit chart —
    what makes the DDPM/DDIM scaffold legitimate over the polytope. Project with Π_S
    (log_sinkhorn(u_t).exp()) to get the doubly-stochastic P_t the denoiser reads.

    Args:
        u0: (..., R, C) clean logit target (from logit_target).
        t: scalar, or a batch-shaped tensor reshaped to broadcast over R, C.
        noise: optional ε (same shape as u0); sampled standard normal if None.
    Returns u_t (..., R, C), an unconstrained logit matrix.
    """
    if noise is None:
        noise = torch.randn_like(u0)
    ab = cosine_alpha_bar(t, s)
    if ab.dim() > 0 and ab.dim() == u0.dim() - 2:
        ab = ab.reshape(*ab.shape, 1, 1)
    return ab.sqrt() * u0 + (1.0 - ab).clamp_min(0.0).sqrt() * noise


def tau_schedule(
    t: torch.Tensor,
    tau_min: float,
    tau_max: float,
    mode: str = "geometric",
) -> torch.Tensor:
    """Sampler-time temperature vs diffusion time t (sampler only; training stays temperate).

    Anneals from tau_max at the noisy end (t=1) to tau_min at the clean end (t=0), so
    the projection Π_S sharpens toward a permutation as sampling approaches the data.

    Args:
        t: diffusion time in [0, 1] (scalar or tensor).
        tau_min: temperature at t=0 (clean, sharpest).
        tau_max: temperature at t=1 (noisy, smoothest).
        mode: "geometric" (log-linear) or "linear".
    """
    if mode == "geometric":
        return tau_min ** (1.0 - t) * tau_max ** t
    if mode == "linear":
        return (1.0 - t) * tau_min + t * tau_max
    raise ValueError(f"unknown tau_schedule mode: {mode!r}")


# --------------------------------------------------------------------------- #
# unit tests (Step 1 "done when"): marginals in tolerance, tau->0 recovers a
# permutation, gradients flow.  Run: python -m utils.sinkhorn
# --------------------------------------------------------------------------- #
def _run_tests() -> None:
    torch.manual_seed(0)
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f'[{"PASS" if ok else "FAIL"}] {name:42s}' + (f'  {detail}' if detail else ""))

    B, n = 4, 32

    # --- marginals within tolerance (square, unit marginals) --------------- #
    logits = torch.randn(B, n, n)
    P = log_sinkhorn(logits, n_iters=30, tau=1.0).exp()
    row_err = (P.sum(-1) - 1).abs().max().item()
    col_err = (P.sum(-2) - 1).abs().max().item()
    check("square marginals -> 1", max(row_err, col_err) < 1e-4,
          f"row_err={row_err:.1e} col_err={col_err:.1e}")

    # --- rectangular unit-row / scaled-col marginals ----------------------- #
    R, C = 24, 32
    lr = torch.randn(B, R, C)
    log_nu = torch.full((B, C), torch.log(torch.tensor(R / C)).item())
    Pr = log_sinkhorn(lr, n_iters=50, tau=1.0, log_nu=log_nu).exp()
    r_err = (Pr.sum(-1) - 1).abs().max().item()          # rows -> 1
    c_err = (Pr.sum(-2) - R / C).abs().max().item()      # cols -> R/C
    check("rect marginals (rows->1, cols->R/C)", max(r_err, c_err) < 1e-3,
          f"row_err={r_err:.1e} col_err={c_err:.1e}")

    # --- tau -> 0 recovers a permutation ----------------------------------- #
    # Plant a permutation with a separated cost; entropic Sinkhorn must converge to
    # it (a near one-hot matrix). Random logits are avoided here: a row with two
    # near-equal-cost columns keeps ~0.5/0.5 mass at any finite tau even though its
    # argmax is a valid permutation, so per-entry sharpness would test the cost, not
    # the solver.
    perm = torch.stack([torch.randperm(n) for _ in range(B)])
    planted = torch.zeros(B, n, n)
    planted.scatter_(-1, perm.unsqueeze(-1), 1.0)
    planted = 5.0 * planted + 0.1 * torch.randn(B, n, n)
    Phard = log_sinkhorn(planted, n_iters=200, tau=0.02).exp()
    row_max = Phard.max(-1).values.min().item()          # every row nearly one-hot
    recovered = torch.equal(Phard.argmax(-1), perm)
    check("tau->0 recovers a permutation", row_max > 0.99 and recovered,
          f"min row-max={row_max:.3f} recovered={recovered}")

    # --- random doubly stochastic sampler ---------------------------------- #
    Pn = sample_doubly_stochastic(n, n, tau=1.0, n_iters=30, batch_shape=(B,))
    ds_err = max((Pn.sum(-1) - 1).abs().max().item(), (Pn.sum(-2) - 1).abs().max().item())
    check("sample_doubly_stochastic is DS", ds_err < 1e-4, f"marg_err={ds_err:.1e}")

    # --- logit_target: row-stochastic, log finite, recovers the permutation - #
    perm2 = torch.stack([torch.randperm(n) for _ in range(B)])
    P0 = torch.zeros(B, n, n).scatter_(-1, perm2.unsqueeze(-1), 1.0)
    u0 = logit_target(P0, eta=0.1)
    P_tilde = u0.exp()
    lt_row = (P_tilde.sum(-1) - 1).abs().max().item()               # rows sum to 1
    lt_finite = torch.isfinite(u0).all().item()
    lt_recover = torch.equal(u0.argmax(-1), perm2)                  # argmax = GT match
    check("logit_target row-stochastic + finite + recovers perm",
          lt_row < 1e-5 and bool(lt_finite) and lt_recover, f"row_err={lt_row:.1e}")

    # --- cosine_alpha_bar endpoints + monotone decreasing ------------------ #
    ts = torch.linspace(0, 1, 11)
    ab = cosine_alpha_bar(ts)
    ab_ends = abs(ab[0].item() - 1.0) < 1e-6 and ab[-1].item() < 1e-6
    ab_mono = bool((ab[1:] <= ab[:-1]).all())
    check("cosine_alpha_bar endpoints + monotone", ab_ends and ab_mono,
          f"abar(0)={ab[0]:.3f} abar(1)={ab[-1]:.1e}")

    # --- q_sample: VP forward marginal; endpoints + projection is DS -------- #
    t = torch.rand(B)
    eps = torch.randn(B, n, n)
    u_t = q_sample(u0, t, noise=eps)
    q0_err = (q_sample(u0, torch.zeros(B), noise=eps) - u0).abs().max().item()  # t=0 -> u0
    q1_err = (q_sample(u0, torch.ones(B), noise=eps) - eps).abs().max().item()  # t=1 -> noise
    Pt = log_sinkhorn(u_t, n_iters=30).exp()                        # Π_S(u_t) is DS
    q_ds = max((Pt.sum(-1) - 1).abs().max().item(), (Pt.sum(-2) - 1).abs().max().item())
    check("q_sample endpoints + Π_S doubly stochastic",
          q0_err < 1e-6 and q1_err < 1e-6 and q_ds < 1e-4,
          f"t0={q0_err:.1e} t1={q1_err:.1e} marg={q_ds:.1e}")

    # --- tau_schedule endpoints and monotonicity --------------------------- #
    ts = torch.linspace(0, 1, 11)
    tg = tau_schedule(ts, 0.05, 1.0, "geometric")
    tl = tau_schedule(ts, 0.05, 1.0, "linear")
    mono = bool((tg[1:] >= tg[:-1]).all() and (tl[1:] >= tl[:-1]).all())
    ends = abs(tg[0] - 0.05) < 1e-6 and abs(tg[-1] - 1.0) < 1e-6
    check("tau_schedule endpoints + monotone", mono and ends,
          f"tau(0)={tg[0]:.3f} tau(1)={tg[-1]:.3f}")

    # --- gradients flow through the unrolled Sinkhorn ---------------------- #
    x = torch.randn(B, n, n, requires_grad=True)
    loss = log_sinkhorn(x, n_iters=10, tau=0.5).exp().sum()
    loss.backward()
    g_ok = x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    check("gradients flow", g_ok, f"grad_norm={x.grad.norm().item():.3e}")

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_tests()
