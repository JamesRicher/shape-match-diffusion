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
from utils.sinkhorn import sample_doubly_stochastic, log_sinkhorn, safe_log


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


# BP settings for the third pass of the battery: every structural property the trunk
# guarantees must survive with belief propagation writing to the state.
BP_CFG = {"k_logit": 4, "k_feat": 4, "sigma": 0.15, "n_sweeps": 2}


def _tag(cross_type, bp):
    return cross_type + ("+bp" if bp else "")


def _build(cross_type, feat_dim=8, dim=32, heads=4, depth=3, k_intra=5,
           k_feat=4, k_state=4, n_rbf=8, bp=None):
    return build_network({
        "type": "MPNNMatrixDenoiser",
        "feat_dim": feat_dim, "dim": dim, "heads": heads, "depth": depth,
        "cross_type": cross_type, "k_intra": k_intra, "k_feat": k_feat,
        "k_state": k_state, "n_rbf": n_rbf, "bp": bp,
    }).eval()


def _perturb(net, scale=0.1):
    with torch.no_grad():
        for p in net.parameters():
            p.add_(scale * torch.randn_like(p))


def _report(name, err, tol=1e-4):
    ok = err < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<44} err={err:.2e} (tol {tol:.0e})")
    return ok


def test_identity_at_init(cross_type, bp=None):
    torch.manual_seed(0)
    net = _build(cross_type, bp=bp)
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    with torch.no_grad():
        out = net(P_t, fx, fy, Dx, Dy, t)
    return _report(f"[{_tag(cross_type, bp)}] identity at init", (project(out) - P_t).abs().max().item(),
                   tol=1e-3)   # safe_log's eps floor bounds the round trip


def test_polytope_compliance(cross_type, bp=None):
    torch.manual_seed(1)
    net = _build(cross_type, bp=bp)
    _perturb(net)
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    with torch.no_grad():
        out = project(net(P_t, fx, fy, Dx, Dy, t))
    ok = _report(f"[{_tag(cross_type, bp)}] polytope: row marginals", (out.sum(-1) - 1).abs().max().item())
    ok &= _report(f"[{_tag(cross_type, bp)}] polytope: col marginals", (out.sum(-2) - 1).abs().max().item())
    nonneg = out.min().item()
    print(f"[{'PASS' if nonneg >= 0 else 'FAIL'}] [{_tag(cross_type, bp)}] polytope: nonnegativity"
          f"{'':<14} min={nonneg:.2e}")
    return ok and nonneg >= 0.0


def test_full_permutation_equivariance(cross_type, bp=None):
    """FULL relabeling of both shapes (not just an anchor-preserving tail)."""
    torch.manual_seed(2)
    net = _build(cross_type, bp=bp)
    _perturb(net, 0.05)
    B, n, feat_dim = 2, 16, 8
    fx, fy, Dx, Dy, P_t, t = _make_inputs(B, n, feat_dim)
    px, py = torch.randperm(n), torch.randperm(n)

    with torch.no_grad():
        out = net(P_t, fx, fy, Dx, Dy, t)
        out_perm = net(P_t[:, py][:, :, px], fx[:, px], fy[:, py],
                       Dx[:, px][:, :, px], Dy[:, py][:, :, py], t)
    expected = out[:, py][:, :, px]
    return _report(f"[{_tag(cross_type, bp)}] FULL permutation equivariance",
                   (out_perm - expected).abs().max().item())


def test_size_agnosticism(cross_type, bp=None):
    torch.manual_seed(3)
    net = _build(cross_type, bp=bp)
    ok = True
    for n in (12, 24):
        fx, fy, Dx, Dy, P_t, t = _make_inputs(2, n, 8)
        with torch.no_grad():
            out = net(P_t, fx, fy, Dx, Dy, t)
        shape_ok = tuple(out.shape) == (2, n, n)
        marg_ok = (project(out, n_iters=100).sum(-1) - 1).abs().max().item() < 1e-4
        print(f"[{'PASS' if shape_ok and marg_ok else 'FAIL'}] [{_tag(cross_type, bp)}] size agnosticism "
              f"n={n:<2}{'':<21} shape={tuple(out.shape)} marg_ok={marg_ok}")
        ok &= shape_ok and marg_ok
    return ok


def test_pair_swap_symmetry(cross_type, bp=None):
    torch.manual_seed(4)
    net = _build(cross_type, bp=bp)
    _perturb(net)
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    with torch.no_grad():
        out = net(P_t, fx, fy, Dx, Dy, t)
        out_swap = net(P_t.transpose(-1, -2), fy, fx, Dy, Dx, t)
    return _report(f"[{_tag(cross_type, bp)}] pair-swap symmetry (logits)",
                   (out_swap - out.transpose(-1, -2)).abs().max().item())


def test_rigid_motion_invariance(cross_type, bp=None):
    import inspect
    params = inspect.signature(_build(cross_type, bp=bp).forward).parameters
    banned = {"coords", "xyz", "verts", "pos", "X", "Y"}
    clean = banned.isdisjoint(params)
    print(f"[{'PASS' if clean else 'FAIL'}] [{_tag(cross_type, bp)}] rigid-motion: intrinsic-only "
          f"args={list(params)}")
    return clean


def test_gradients_flow(cross_type, bp=None):
    torch.manual_seed(5)
    net = _build(cross_type, bp=bp).train()
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
    print(f"[{'PASS' if ok else 'FAIL'}] [{_tag(cross_type, bp)}] gradients flow"
          f"{'':<30} {n_with}/{n_tot} params")
    return ok


# --------------------------------------------------------------------------- #
# BP-specific properties (notes/BP-loop-design.md)
# --------------------------------------------------------------------------- #
def test_bp_disabled_by_default():
    """`bp: null` must leave no parameters and no code path behind."""
    off, on = _build("mpnn"), _build("mpnn", bp=BP_CFG)
    n_off = sum(p.numel() for p in off.parameters())
    n_on = sum(p.numel() for p in on.parameters())
    ok = off.bp is None and on.bp is not None and n_on > n_off
    print(f"[{'PASS' if ok else 'FAIL'}] [bp] disabled by default"
          f"{'':<25} params {n_off} -> {n_on} (+{n_on - n_off})")
    return ok


def test_bp_gates_at_init():
    """g = 0, beta = beta_init, and (when enabled) w = 1 exactly at init."""
    net = _build("mpnn", bp=BP_CFG)
    c = torch.randn(4, 32)
    blk = net.bp.sites[0]
    with torch.no_grad():
        g = blk.g_scale * blk.g_head(c)
        beta = blk._beta(c)
    ok = _report("[bp] output gate g = 0 at init", g.abs().max().item())
    ok &= _report("[bp] beta = beta_init, constant in t", (beta - 0.5).abs().max().item())

    gated = _build("mpnn", bp={**BP_CFG, "cycle_gate": True}).bp.sites[0]
    with torch.no_grad():
        w = gated._gate(torch.rand(4, 16), c)
    return ok and _report("[bp] agreement gate w = 1 at init", (w - 1).abs().max().item())


def test_bp_beta_bounds():
    """beta stays inside (beta_min, beta_max) for any head output, and cannot saturate
    to a dead gradient at either end (notes: the softplus version reached 0.000 and 9.6).
    """
    blk = _build("mpnn", bp={**BP_CFG, "beta_min": 0.1, "beta_max": 4.0}).bp.sites[0]
    with torch.no_grad():                       # drive the head hard in both directions
        blk.beta_head[-1].bias.fill_(-50.0)
        lo = blk._beta(torch.randn(8, 32))
        blk.beta_head[-1].bias.fill_(50.0)
        hi = blk._beta(torch.randn(8, 32))
    inside = bool((lo >= 0.1).all() and (hi <= 4.0).all())
    print(f"[{'PASS' if inside else 'FAIL'}] [bp] beta bounded"
          f"{'':<34} min={lo.min():.4f} max={hi.max():.4f}")

    blk.beta_head[-1].bias.data.fill_(3.0)      # well up the sigmoid, not at init
    beta = blk._beta(torch.randn(4, 32))
    beta.sum().backward()
    grad = blk.beta_head[-1].bias.grad.abs().max().item()
    live = grad > 0.0
    print(f"[{'PASS' if live else 'FAIL'}] [bp] beta gradient alive off-centre"
          f"{'':<20} |grad|={grad:.2e}")
    return inside and live


def test_bp_writes_when_gated_on():
    """With g forced non-zero the state must actually move (the block is not inert)."""
    torch.manual_seed(6)
    net = _build("mpnn", bp=BP_CFG)
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    with torch.no_grad():
        base = net(P_t, fx, fy, Dx, Dy, t)
        net.bp.sites[0].g_head[-1].bias.fill_(0.5)
        moved = net(P_t, fx, fy, Dx, Dy, t)
    d = (moved - base).abs().max().item()
    ok = d > 1e-4 and torch.isfinite(moved).all()
    print(f"[{'PASS' if ok else 'FAIL'}] [bp] writes when gated on"
          f"{'':<27} max|Δu|={d:.3e}")
    return ok


def test_bp_runs_once_at_configured_block():
    """BP fires exactly once per forward, at `at_block` and nowhere else.

    Drives the stage by hand over every block index: exactly one must move u, and it
    must be the configured one. This is the invariant the whole rewrite rests on — six
    calls per forward was what the trained gates had already collapsed to ~one.
    """
    torch.manual_seed(7)
    depth, at = 3, 1
    net = _build("mpnn", depth=depth, bp={**BP_CFG, "at_block": at})
    with torch.no_grad():
        net.bp.sites[0].g_head[-1].bias.fill_(0.5)
    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    idx_x, _ = net._geo(Dx)
    idx_y, _ = net._geo(Dy)
    geo_x, geo_y = (idx_x, Dx), (idx_y, Dy)
    with torch.no_grad():
        c = net.spine(t)
        u = safe_log(P_t)
        cache = net.bp.init_cache(fx, fy, geo_x, geo_y)
        fired = [(net.bp(i, u, cache, fx, fy, geo_x, geo_y, c) - u).abs().max().item()
                 for i in range(depth)]
    ok = fired[at] > 1e-4 and all(v == 0.0 for i, v in enumerate(fired) if i != at)
    print(f"[{'PASS' if ok else 'FAIL'}] [bp] runs once, at at_block={at}"
          f"{'':<20} writes={['%.1e' % v for v in fired]}")

    bad = 0
    for cfg in ({"at_block": depth},          # out of range
                {"at_block": []},             # no site at all
                {"at_block": [0, 0]},         # duplicate site
                {"at_block": [0, depth]}):    # one entry out of range
        try:
            _build("mpnn", depth=depth, bp={**BP_CFG, **cfg})
        except ValueError:
            bad += 1
    print(f"[{'PASS' if bad == 4 else 'FAIL'}] [bp] bad at_block rejected"
          f"{'':<25} {bad}/4")
    return ok and bad == 4


def test_bp_multi_site():
    """`at_block: [a, b]` must fire at exactly those blocks, with INDEPENDENT gates.

    Weight-shared sites would write identically at every site (beta and g are functions of
    c, which is fixed within a forward), which is precisely the signal the two-site run is
    meant to measure. Per-site stats must stay separate for the same reason.
    """
    torch.manual_seed(7)
    depth, sites = 6, [1, 5]
    net = _build("mpnn", depth=depth, bp={**BP_CFG, "at_block": sites})
    ok = net.bp.at_blocks == tuple(sites) and len(net.bp.sites) == 2

    # distinct parameters, and distinct writes: gate site 1 twice as hard as site 0
    with torch.no_grad():
        for s, b in enumerate(net.bp.sites):
            b.g_head[-1].bias.fill_(0.5 * (s + 1))
    shared = net.bp.sites[0].g_head[-1].bias is net.bp.sites[1].g_head[-1].bias
    ok &= not shared

    fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
    idx_x, _ = net._geo(Dx)
    idx_y, _ = net._geo(Dy)
    geo_x, geo_y = (idx_x, Dx), (idx_y, Dy)
    net.eval()
    with torch.no_grad():
        c = net.spine(t)
        u = safe_log(P_t)
        cache = net.bp.init_cache(fx, fy, geo_x, geo_y)
        fired = [(net.bp(i, u, cache, fx, fy, geo_x, geo_y, c) - u).abs().max().item()
                 for i in range(depth)]
    ok &= all((v > 1e-4) == (i in sites) for i, v in enumerate(fired))
    ok &= fired[5] > fired[1]                       # harder gate -> bigger write
    # stats namespaced per trunk block, so the sites cannot overwrite each other
    keys = {k.split('/')[0] for k in net.bp.stats}
    ok &= keys == {"bp1", "bp5"}
    print(f"[{'PASS' if ok else 'FAIL'}] [bp] multi-site at_block={sites}"
          f"{'':<19} writes={['%.1e' % v for v in fired]} stats={sorted(keys)}")

    single = _build("mpnn", depth=depth, bp={**BP_CFG, "at_block": 3})
    compat = single.bp.at_blocks == (3,) and single.bp.at_block == 3
    print(f"[{'PASS' if compat else 'FAIL'}] [bp] int at_block still supported")
    return ok and compat


def test_bp_fixed_beta_g():
    """beta_fixed / g_fixed must pin the value exactly, drop the head, and stay constant in t.

    NOTE g_fixed != 0 deliberately breaks identity-at-init: BP writes from step 0. That is the
    point -- a zero-init gate needs a head of ~5.5 to reach the swept optimum g=16.
    """
    net = _build("mpnn", bp={**BP_CFG, "beta_fixed": 0.25, "g_fixed": 16.0}).eval()
    blk = net.bp.sites[0]
    ok = blk.beta_head is None and blk.g_head is None
    n_bp = sum(p.numel() for n, p in net.named_parameters() if n.startswith('bp.'))
    ok &= n_bp == 0

    fx, fy, Dx, Dy, P_t, _ = _make_inputs(2, 16, 8)
    vals = []
    for t in (0.05, 0.5, 0.95):                       # beta/g must not vary with t
        with torch.no_grad():
            c = net.spine(torch.full((2,), t))
            vals.append((blk._beta(c).mean().item(), blk._g(c).mean().item()))
    ok &= all(abs(b - 0.25) < 1e-6 and abs(g - 16.0) < 1e-6 for b, g in vals)

    with torch.no_grad():
        out = net(P_t, fx, fy, Dx, Dy, torch.full((2,), 0.5))
    writes = blk(safe_log(P_t), fx, fy, (net._geo(Dx)[0], Dx), (net._geo(Dy)[0], Dy),
                 net.spine(torch.full((2,), 0.5)))[0]
    ok &= torch.isfinite(out).all().item() and writes.abs().max().item() > 1e-3

    learned = _build("mpnn", bp=BP_CFG).bp.sites[0]   # default path untouched
    ok &= learned.beta_head is not None and learned.g_head is not None
    bad = 0
    for cfg in ({"beta_fixed": 0.0}, {"g_fixed": 25.0}):   # g_max=20
        try:
            _build("mpnn", bp={**BP_CFG, **cfg})
        except ValueError:
            bad += 1
    print(f"[{'PASS' if ok and bad == 2 else 'FAIL'}] [bp] beta_fixed / g_fixed"
          f"{'':<28} bp_params={n_bp} beta,g={vals[0]} rejected={bad}/2")
    return ok and bad == 2


def test_bp_sweep_count_override():
    """Test-time inference effort must be settable without touching weights."""
    net = _build("mpnn", bp=BP_CFG)
    net.bp.set_n_sweeps(9)
    ok = net.bp.sites[0].n_sweeps == 9
    fx, fy, Dx, Dy, P_t, t = _make_inputs(1, 16, 8)
    with torch.no_grad():
        finite = torch.isfinite(net(P_t, fx, fy, Dx, Dy, t)).all().item()
    print(f"[{'PASS' if ok and finite else 'FAIL'}] [bp] sweep count override"
          f"{'':<28} n_sweeps=9 finite={finite}")
    return ok and finite


def test_bp_checkpoint_equivalence():
    """Gradient checkpointing must change memory, not values or gradients."""
    torch.manual_seed(8)
    outs, grads = [], []
    for ckpt in (False, True):
        torch.manual_seed(8)
        net = _build("mpnn", bp={**BP_CFG, "checkpoint": ckpt}).train()
        _perturb(net, 0.05)
        fx, fy, Dx, Dy, P_t, t = _make_inputs(2, 16, 8)
        out = net(P_t, fx, fy, Dx, Dy, t)
        out.sum().backward()
        outs.append(out.detach())
        grads.append(net.bp.sites[0].g_head[-1].weight.grad.clone())
    ok = _report("[bp] checkpoint: same forward", (outs[0] - outs[1]).abs().max().item())
    return ok and _report("[bp] checkpoint: same gradients",
                          (grads[0] - grads[1]).abs().max().item(), tol=1e-5)


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
    for cross_type, bp in (("attn", None), ("mpnn", None), ("mpnn", BP_CFG)):
        print(f"\n=== cross_type = {_tag(cross_type, bp)} ===")
        results += [t(cross_type, bp) for t in tests]

    print("\n=== belief propagation ===")
    results += [t() for t in (test_bp_disabled_by_default,
                              test_bp_gates_at_init,
                              test_bp_beta_bounds,
                              test_bp_writes_when_gated_on,
                              test_bp_runs_once_at_configured_block,
                              test_bp_multi_site,
                              test_bp_fixed_beta_g,
                              test_bp_sweep_count_override,
                              test_bp_checkpoint_equivalence)]
    print(f"\n{sum(results)}/{len(results)} property groups passed")
    raise SystemExit(0 if all(results) else 1)
