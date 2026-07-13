"""Property-test battery for MatrixDenoiser (theory note 2026-07-08_theoretical_properties).

The denoiser predicts a clean LOGIT matrix u0_hat (predict-x0, logit-space scheme,
notes/noising.md); Sinkhorn projection Π_S is external, applied here before any
doubly-stochastic assertion. Mechanically-testable hard requirements, plus id-at-init:

    1. permutation equivariance   f(Pi_Y P Pi_X^T, ...) = Pi_Y f(P) Pi_X^T (on u0_hat)
    2. polytope compliance        Π_S(u0_hat) marginals == 1 (uniform convention)
    3. size agnosticism           two values of n through the same weights
    5. rigid-motion invariance    inputs are intrinsic (D, F, P_t) -- no coords enter
    6. pair-swap symmetry         f(P^T, Y<->X) = f(P)^T (exact on u0_hat logits)
    +  identity at init           Π_S(u0_hat) == P_t at init

Note on (1): the token anchor coordinates are D[:, :a], the geodesic distances to the
first `a` FPS points -- an intrinsic frame fixed by the FPS ordering. Equivariance
therefore holds for relabelings that preserve that anchor set, i.e. permutations of the
arbitrary FPS tail (indices >= a). The test permutes the tail. (Making it hold for ALL
relabelings needs gather-by-anchor-index in TokenConstructor -- open decision.)

Run: python -m networks.matrix_denoiser_tests
"""
import torch

from networks import build_network
from utils.sinkhorn import sample_doubly_stochastic, log_sinkhorn


def project(u: torch.Tensor, n_iters: int = 50) -> torch.Tensor:
    """External Sinkhorn projection Π_S to the Birkhoff polytope (near-converged for
    decisive assertions)."""
    return log_sinkhorn(u, n_iters=n_iters).exp()


def _make_D(B: int, n: int) -> torch.Tensor:
    """Symmetric, zero-diagonal distance matrix (euclidean proxy for a geodesic one)."""
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


def _build(feat_dim=8, dim=32, heads=4, depth=3, n_anchors=4):
    return build_network({
        "type": "MatrixDenoiser",
        "feat_dim": feat_dim, "dim": dim, "heads": heads,
        "depth": depth, "n_anchors": n_anchors,
    }).eval()


def _report(name, err, tol=1e-4):
    ok = err < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<32} err={err:.2e} (tol {tol:.0e})")
    return ok


def test_identity_at_init():
    torch.manual_seed(0)
    net = _build()
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    with torch.no_grad():
        out = net(P_t, fx, fy, Dx, Dy, t)        # logits u0_hat
    return _report("identity at init (Π_S(u0_hat) == P_t)", (project(out) - P_t).abs().max().item())


def test_polytope_compliance():
    torch.manual_seed(1)
    net = _build()
    # perturb params off the identity init so the readout does real work
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    with torch.no_grad():
        out = project(net(P_t, fx, fy, Dx, Dy, t))   # Π_S(u0_hat)
    row_err = (out.sum(-1) - 1.0).abs().max().item()   # rows (per Y token) sum to 1
    col_err = (out.sum(-2) - 1.0).abs().max().item()   # cols (per X token) sum to 1
    nonneg = out.min().item()
    ok = _report("polytope: row marginals", row_err)
    ok &= _report("polytope: col marginals", col_err)
    ok &= (nonneg >= 0.0)
    print(f"[{'PASS' if nonneg >= 0 else 'FAIL'}] polytope: nonnegativity        min={nonneg:.2e}")
    return ok


def test_permutation_equivariance():
    torch.manual_seed(2)
    a = 4
    net = _build(n_anchors=a)
    B, n, feat_dim = 2, 16, 8
    fx, fy, Dx, Dy, P_t, t = _make_inputs(B, n, feat_dim)

    # permute only the FPS tail (>= a) so the anchor frame is preserved
    def tail_perm():
        p = torch.arange(n)
        p[a:] = a + torch.randperm(n - a)
        return p
    px, py = tail_perm(), tail_perm()

    fx_p = fx[:, px]
    fy_p = fy[:, py]
    Dx_p = Dx[:, px][:, :, px]
    Dy_p = Dy[:, py][:, :, py]
    P_p = P_t[:, py][:, :, px]          # rows=Y (py), cols=X (px)

    with torch.no_grad():
        out = net(P_t, fx, fy, Dx, Dy, t)
        out_perm = net(P_p, fx_p, fy_p, Dx_p, Dy_p, t)
    expected = out[:, py][:, :, px]
    return _report("permutation equivariance (tail)", (out_perm - expected).abs().max().item())


def test_size_agnosticism():
    torch.manual_seed(3)
    net = _build()
    ok = True
    for n in (12, 24):
        fx, fy, Dx, Dy, P_t, t = _make_inputs(2, n, 8)
        with torch.no_grad():
            out = net(P_t, fx, fy, Dx, Dy, t)
        shape_ok = tuple(out.shape) == (2, n, n)
        marg_ok = (project(out).sum(-1) - 1.0).abs().max().item() < 1e-4
        print(f"[{'PASS' if shape_ok and marg_ok else 'FAIL'}] size agnosticism n={n:<2}          "
              f"shape={tuple(out.shape)} marg_ok={marg_ok}")
        ok &= shape_ok and marg_ok
    return ok


def test_pair_swap_symmetry():
    torch.manual_seed(4)
    net = _build()
    # perturb off identity: the symmetry must hold with real (non-inert) weights
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.1 * torch.randn_like(p))
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    with torch.no_grad():
        out = net(P_t, fx, fy, Dx, Dy, t)
        out_swap = net(P_t.transpose(-1, -2), fy, fx, Dy, Dx, t)   # P^T, Y<->X
    # exact on the logit output now (projection is external): f(P^T, Y<->X) = f(P)^T
    return _report("pair-swap symmetry (logits)", (out_swap - out.transpose(-1, -2)).abs().max().item())


def test_rigid_motion_invariance():
    # Property 5 holds by construction: forward consumes only intrinsic inputs
    # (geodesic D, features F, assignment P_t); no extrinsic coordinates enter, so
    # rotating either shape leaves every input -- and thus the output -- unchanged.
    import inspect
    params = inspect.signature(_build().forward).parameters
    banned = {"coords", "xyz", "verts", "pos", "X", "Y"}
    clean = banned.isdisjoint(params)
    print(f"[{'PASS' if clean else 'FAIL'}] rigid-motion: intrinsic-only    args={list(params)}")
    return clean


if __name__ == "__main__":
    tests = [
        test_identity_at_init,
        test_polytope_compliance,
        test_permutation_equivariance,
        test_size_agnosticism,
        test_pair_swap_symmetry,
        test_rigid_motion_invariance,
    ]
    results = [t() for t in tests]
    print(f"\n{sum(results)}/{len(results)} property groups passed")
