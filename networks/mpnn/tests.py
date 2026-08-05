"""Property-test battery for MPNNMatrixDenoiser — the same hard requirements as
networks/matrix_denoiser_tests.py, run for BOTH cross variants, plus two upgrades
the MPNN design buys:

    1. FULL permutation equivariance (no anchor frame -> not just the FPS tail)
    2. pair-swap symmetry (exact on logits, incl. the symmetrised inner Sinkhorn)
    3. identity at init (Π_S(u0_hat) == P_t; structural, no alpha-skip)
    4. polytope compliance, size agnosticism, intrinsic-only inputs
    5. attn/mpnn variant interface parity

Run: python -m networks.mpnn.tests
"""
import torch

from networks import build_network
from utils.sinkhorn import sample_doubly_stochastic, log_sinkhorn


def project(u: torch.Tensor, n_iters: int = 50) -> torch.Tensor:
    return log_sinkhorn(u, n_iters=n_iters).exp()


def _make_D(B: int, n: int) -> torch.Tensor:
    pts = torch.randn(B, n, 3)
    D = torch.cdist(pts, pts)
    return 0.5 * (D + D.transpose(-1, -2))


def _make_inputs(B: int, n: int, feat_dim: int):
    fx = torch.randn(B, n, feat_dim)
    fy = torch.randn(B, n, feat_dim)
    Dx, Dy = _make_D(B, n), _make_D(B, n)
    P_t = sample_doubly_stochastic(n, n, batch_shape=(B,))
    t = torch.rand(B)
    return fx, fy, Dx, Dy, P_t, t


def _build(cross_type, feat_dim=8, dim=32, heads=4, depth=3, k_intra=5,
           k_feat=4, k_state=4, n_rbf=8):
    return build_network({
        "type": "MPNNMatrixDenoiser",
        "feat_dim": feat_dim, "dim": dim, "heads": heads, "depth": depth,
        "cross_type": cross_type, "k_intra": k_intra, "k_feat": k_feat,
        "k_state": k_state, "n_rbf": n_rbf,
    }).eval()


def _perturb(net, scale=0.1):
    with torch.no_grad():
        for p in net.parameters():
            p.add_(scale * torch.randn_like(p))


def _report(name, err, tol=1e-4):
    ok = err < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<44} err={err:.2e} (tol {tol:.0e})")
    return ok


def test_identity_at_init(cross_type):
    torch.manual_seed(0)
    net = _build(cross_type)
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    with torch.no_grad():
        out = net(P_t, fx, fy, Dx, Dy, t)
    return _report(f"[{cross_type}] identity at init", (project(out) - P_t).abs().max().item(),
                   tol=1e-3)   # safe_log's eps floor bounds the round trip


def test_polytope_compliance(cross_type):
    torch.manual_seed(1)
    net = _build(cross_type)
    _perturb(net)
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    with torch.no_grad():
        out = project(net(P_t, fx, fy, Dx, Dy, t))
    ok = _report(f"[{cross_type}] polytope: row marginals", (out.sum(-1) - 1).abs().max().item())
    ok &= _report(f"[{cross_type}] polytope: col marginals", (out.sum(-2) - 1).abs().max().item())
    nonneg = out.min().item()
    print(f"[{'PASS' if nonneg >= 0 else 'FAIL'}] [{cross_type}] polytope: nonnegativity"
          f"{'':<14} min={nonneg:.2e}")
    return ok and nonneg >= 0.0


def test_full_permutation_equivariance(cross_type):
    """FULL relabeling of both shapes (not just an anchor-preserving tail)."""
    torch.manual_seed(2)
    net = _build(cross_type)
    _perturb(net, 0.05)
    B, n, feat_dim = 2, 16, 8
    fx, fy, Dx, Dy, P_t, t = _make_inputs(B, n, feat_dim)
    px, py = torch.randperm(n), torch.randperm(n)

    with torch.no_grad():
        out = net(P_t, fx, fy, Dx, Dy, t)
        out_perm = net(P_t[:, py][:, :, px], fx[:, px], fy[:, py],
                       Dx[:, px][:, :, px], Dy[:, py][:, :, py], t)
    expected = out[:, py][:, :, px]
    return _report(f"[{cross_type}] FULL permutation equivariance",
                   (out_perm - expected).abs().max().item())


def test_size_agnosticism(cross_type):
    torch.manual_seed(3)
    net = _build(cross_type)
    ok = True
    for n in (12, 24):
        fx, fy, Dx, Dy, P_t, t = _make_inputs(2, n, 8)
        with torch.no_grad():
            out = net(P_t, fx, fy, Dx, Dy, t)
        shape_ok = tuple(out.shape) == (2, n, n)
        marg_ok = (project(out, n_iters=100).sum(-1) - 1).abs().max().item() < 1e-4
        print(f"[{'PASS' if shape_ok and marg_ok else 'FAIL'}] [{cross_type}] size agnosticism "
              f"n={n:<2}{'':<21} shape={tuple(out.shape)} marg_ok={marg_ok}")
        ok &= shape_ok and marg_ok
    return ok


def test_pair_swap_symmetry(cross_type):
    torch.manual_seed(4)
    net = _build(cross_type)
    _perturb(net)
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    with torch.no_grad():
        out = net(P_t, fx, fy, Dx, Dy, t)
        out_swap = net(P_t.transpose(-1, -2), fy, fx, Dy, Dx, t)
    return _report(f"[{cross_type}] pair-swap symmetry (logits)",
                   (out_swap - out.transpose(-1, -2)).abs().max().item())


def test_rigid_motion_invariance(cross_type):
    import inspect
    params = inspect.signature(_build(cross_type).forward).parameters
    banned = {"coords", "xyz", "verts", "pos", "X", "Y"}
    clean = banned.isdisjoint(params)
    print(f"[{'PASS' if clean else 'FAIL'}] [{cross_type}] rigid-motion: intrinsic-only "
          f"args={list(params)}")
    return clean


def test_gradients_flow(cross_type):
    torch.manual_seed(5)
    net = _build(cross_type).train()
    _perturb(net, 0.05)
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    out = net(P_t, fx, fy, Dx, Dy, t)
    project(out, n_iters=10).sum().backward()
    n_with, n_tot = 0, 0
    for name, p in net.named_parameters():
        n_tot += 1
        if p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0:
            n_with += 1
    # everything trainable must receive gradient (gates are perturbed off zero,
    # so every pathway is live; dead params indicate dead compute)
    ok = n_with == n_tot
    print(f"[{'PASS' if ok else 'FAIL'}] [{cross_type}] gradients flow"
          f"{'':<30} {n_with}/{n_tot} params")
    return ok


if __name__ == "__main__":
    tests = [
        test_identity_at_init,
        test_polytope_compliance,
        test_full_permutation_equivariance,
        test_size_agnosticism,
        test_pair_swap_symmetry,
        test_rigid_motion_invariance,
        test_gradients_flow,
    ]
    results = []
    for cross_type in ("attn", "mpnn"):
        print(f"\n=== cross_type = {cross_type} ===")
        results += [t(cross_type) for t in tests]
    print(f"\n{sum(results)}/{len(results)} property groups passed")
    raise SystemExit(0 if all(results) else 1)
