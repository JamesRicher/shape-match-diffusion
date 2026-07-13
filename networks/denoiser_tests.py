"""Checks for the matrix-denoiser primitives (blocks, conditioning, io).

Covers: identity-at-init and pair-swap symmetry of the AdaLN blocks, per-head bias
acceptance, the conditioning spine, the two bias constructors, token construction, and
the logit readout. Run: python -m networks.denoiser_tests
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sinkhorn import safe_log, sample_doubly_stochastic, log_sinkhorn
from networks.denoiser_blocks import AdaLNIntraShapeBlock, AdaLNInterShapeBlock
from networks.denoiser_conditioning import (ConditioningSpine, LogAssignmentBias,
                                            GeodesicKernelBias)
from networks.denoiser_io import TokenConstructor, LogitReadout


def _run_tests() -> None:
    torch.manual_seed(0)
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f'[{"PASS" if ok else "FAIL"}] {name:44s}' + (f'  {detail}' if detail else ""))

    B, Nx, Ny, dim, heads = 2, 12, 16, 32, 4
    x = torch.randn(B, Nx, dim)
    y = torch.randn(B, Ny, dim)
    c = torch.randn(B, dim)

    # --- identity at init (all gates zero) --------------------------------- #
    intra = AdaLNIntraShapeBlock(dim, heads).eval()
    inter = AdaLNInterShapeBlock(dim, heads).eval()
    xi = intra(x, c)
    xo, yo = inter(x, y, c)
    id_err = max((xi - x).abs().max().item(),
                 (xo - x).abs().max().item(), (yo - y).abs().max().item())
    check("identity at init", id_err < 1e-6, f"max dev={id_err:.1e}")

    # De-zero the modulation so the blocks compute a non-trivial function for the
    # remaining structural checks (mirrors network_tests randomising the bias).
    for blk in (intra, inter):
        nn.init.normal_(blk.mod.proj.weight, std=0.02)
        nn.init.normal_(blk.mod.proj.bias, std=0.02)

    # --- per-head bias accepted (shape-compatible) ------------------------- #
    shared_bias = torch.randn(B, Ny, Nx)
    perhead_bias = torch.randn(B, heads, Ny, Nx)
    ok_shapes = True
    try:
        _ = intra(x, c, bias=torch.randn(B, Nx, Nx))
        _ = intra(x, c, bias=torch.randn(B, heads, Nx, Nx))
        _ = inter(x, y, c, bias=shared_bias.transpose(-1, -2))    # X<-Y uses (Nx,Ny)
        _ = inter(x, y, c, bias=perhead_bias.transpose(-1, -2))
    except Exception as e:  # noqa: BLE001
        ok_shapes = False
        print("   ", type(e).__name__, e)
    check("per-head + shared bias accepted", ok_shapes)

    # --- inter pair-swap symmetry: f(y, x, bias^T) == swap of f(x, y, bias) - #
    # bias is the X<-Y bias of shape (Nx, Ny); its transpose is the Y<-X bias.
    bias_xy = torch.randn(B, Nx, Ny)
    ax, ay = inter(x, y, c, bias=bias_xy)
    bx, by = inter(y, x, c, bias=bias_xy.transpose(-1, -2))
    sym_err = max((ax - by).abs().max().item(), (ay - bx).abs().max().item())
    check("inter pair-swap symmetry", sym_err < 1e-5, f"residual={sym_err:.1e}")

    # same, per-head bias
    bias_xy_h = torch.randn(B, heads, Nx, Ny)
    ax, ay = inter(x, y, c, bias=bias_xy_h)
    bx, by = inter(y, x, c, bias=bias_xy_h.transpose(-1, -2))
    sym_err_h = max((ax - by).abs().max().item(), (ay - bx).abs().max().item())
    check("inter pair-swap symmetry (per-head)", sym_err_h < 1e-5, f"residual={sym_err_h:.1e}")

    # --- conditioning spine: t -> c --------------------------------------- #
    spine = ConditioningSpine(dim)

    # shape + scalar/0-dim handling
    c_batch = spine(torch.rand(B))
    c_scalar = spine(0.3)
    ok_shape = c_batch.shape == (B, dim) and c_scalar.shape == (1, dim)
    check("spine output shape (batch + scalar)", ok_shape,
          f"{tuple(c_batch.shape)}, {tuple(c_scalar.shape)}")

    # resolves t across [0, 1]: nearby t give distinct, smoothly-varying c
    ts = torch.linspace(0, 1, 21)
    cs = spine(ts)                                  # (21, dim)
    step = (cs[1:] - cs[:-1]).norm(dim=-1)          # consecutive distances
    resolves = bool((step > 1e-4).all())            # every step moves c
    d_far = (spine(torch.tensor([0.1])) - spine(torch.tensor([0.9]))).norm().item()
    d_near = (spine(torch.tensor([0.1])) - spine(torch.tensor([0.12]))).norm().item()
    check("spine resolves t (varies, far>near)", resolves and d_far > d_near,
          f"far={d_far:.2f} near={d_near:.2f}")

    # feeds the blocks: c from the spine drives a real (non-identity) modulation
    c_t = spine(torch.rand(B))
    out = intra(x, c_t)
    check("spine c drives the blocks", out.shape == x.shape and not torch.allclose(out, x))

    # gradients flow t -> c -> loss
    t_req = torch.rand(B, requires_grad=True)
    loss = spine(t_req).pow(2).sum()
    loss.backward()
    g_ok = t_req.grad is not None and torch.isfinite(t_req.grad).all()
    p_grad = all(p.grad is not None for p in spine.parameters())
    check("spine gradients flow", g_ok and p_grad)

    # --- inter bias: LogAssignmentBias ------------------------------------- #
    P_t = sample_doubly_stochastic(Ny, Nx, tau=1.0, n_iters=20, batch_shape=(B,))

    lab0 = LogAssignmentBias(heads)
    zero_ok = torch.count_nonzero(lab0(P_t)) == 0           # zero-init gamma -> zero bias
    check("inter bias zero at init", zero_ok)

    lab = LogAssignmentBias(heads)
    nn.init.normal_(lab.gamma, std=1.0)                     # de-zero
    bias = lab(P_t)                                         # (B, H, Nx, Ny)
    logP = torch.log(P_t.clamp_min(lab.eps))
    expect = lab.gamma.view(1, -1, 1, 1) * logP.transpose(-1, -2).unsqueeze(1)
    orient_ok = bias.shape == (B, heads, Nx, Ny) and torch.allclose(bias, expect)
    # spot-check the routing orientation: bias[b,h,i,j] == gamma_h * log P_t[b,j,i]
    orient_ok &= torch.allclose(bias[:, :, 3, 5], lab.gamma[None] * logP[:, 5, 3][:, None])
    check("inter bias per-head + orientation", orient_ok, f"shape={tuple(bias.shape)}")

    # feeds the inter block (X<-Y bias), stays pair-swap symmetric
    ax, ay = inter(x, y, c, bias=bias)
    bx, by = inter(y, x, c, bias=bias.transpose(-1, -2))
    fed_ok = max((ax - by).abs().max().item(), (ay - bx).abs().max().item()) < 1e-5
    check("inter bias feeds block (symmetric)", fed_ok)

    # --- intra bias: GeodesicKernelBias ------------------------------------ #
    # symmetric geodesic-like matrix with zero diagonal
    R = torch.rand(B, Nx, Nx)
    D = (R + R.transpose(-1, -2))
    D[:, torch.arange(Nx), torch.arange(Nx)] = 0.0

    gkb = GeodesicKernelBias(heads)
    gbias = gkb(D)                                          # (B, H, Nx, Nx)
    gamma = F.softplus(gkb.raw_gamma)
    g_expect = -gamma.view(1, -1, 1, 1) * (D ** 2).unsqueeze(1)
    shape_ok = gbias.shape == (B, heads, Nx, Nx) and torch.allclose(gbias, g_expect)
    nonpos = bool((gbias <= 1e-6).all())                   # -gamma D^2 <= 0
    diag0 = bool((gbias[:, :, torch.arange(Nx), torch.arange(Nx)].abs() < 1e-6).all())
    sym = torch.allclose(gbias, gbias.transpose(-1, -2), atol=1e-5)
    spread = gamma.max().item() / gamma.min().item()       # heads span a bandwidth range
    check("intra bias kernel (<=0, diag0, symmetric)", shape_ok and nonpos and diag0 and sym,
          f"bandwidth spread x{spread:.0f}")

    # feeds the intra block for a shape
    check("intra bias feeds block", intra(x, c, bias=gkb(D)).shape == x.shape)

    # --- token construction ------------------------------------------------ #
    d_f, a = 8, 6
    F_x, F_y = torch.randn(B, Nx, d_f), torch.randn(B, Ny, d_f)
    R_x, R_y = torch.rand(B, Nx, Nx), torch.rand(B, Ny, Ny)
    D_x = R_x + R_x.transpose(-1, -2); D_x[:, torch.arange(Nx), torch.arange(Nx)] = 0
    D_y = R_y + R_y.transpose(-1, -2); D_y[:, torch.arange(Ny), torch.arange(Ny)] = 0

    tok = TokenConstructor(d_f, dim, n_anchors=a)
    x0, y0 = tok(F_x, F_y, D_x, D_y, P_t)
    shape_ok = x0.shape == (B, Nx, dim) and y0.shape == (B, Ny, dim)
    check("token shapes + in_dim", shape_ok and tok.in_dim == 2 * d_f + 2 * a + 1,
          f"in_dim={tok.in_dim}")

    # pair-swap symmetry: swap X<->Y and P_t -> P_t^T  =>  x0 <-> y0
    x0s, y0s = tok(F_y, F_x, D_y, D_x, P_t.transpose(-1, -2))
    swap_err = max((x0s - y0).abs().max().item(), (y0s - x0).abs().max().item())
    check("token pair-swap symmetry", swap_err < 1e-5, f"residual={swap_err:.1e}")

    # entropy endpoints: uniform P_t -> row entropy = log(n_x); near-permutation -> ~0
    uni = torch.full((1, Ny, Nx), 1.0 / Nx)
    ent_uni = -(uni * safe_log(uni)).sum(-1)[0, 0].item()
    hard = torch.zeros(1, Nx, Nx); hard[0, torch.arange(Nx), torch.arange(Nx)] = 1.0
    ent_hard = -(hard * safe_log(hard)).sum(-1)[0, 0].item()
    check("entropy endpoints (uniform=log n, perm~0)",
          abs(ent_uni - math.log(Nx)) < 1e-4 and ent_hard < 1e-4,
          f"uni={ent_uni:.3f} log(n)={math.log(Nx):.3f} hard={ent_hard:.1e}")

    # pull-back sanity: uniform P_t -> P_t F_x = feature mean over X (uninformative)
    Pf = uni @ F_x                                          # (B, Ny, d_f)
    check("pull-back of uniform P_t = feature mean",
          torch.allclose(Pf, F_x.mean(1, keepdim=True).expand(B, Ny, d_f), atol=1e-5))

    # tokens feed the trunk
    xt = intra(x0, c, bias=gkb(D_x))
    check("tokens feed the trunk", xt.shape == (B, Nx, dim))

    # gradient flow through W_tok
    F_xg = F_x.clone().requires_grad_(True)
    tok(F_xg, F_y, D_x, D_y, P_t)[0].sum().backward()
    check("token gradients flow", F_xg.grad is not None and torch.isfinite(F_xg.grad).all())

    # --- logit readout (square: the matcher's actual n_x == n_y case) ------- #
    # Emits u0_hat (unconstrained logits); Sinkhorn projection Π_S is external.
    N = Ny
    E_x, E_y = torch.randn(B, N, dim), torch.randn(B, N, dim)
    P_sq = sample_doubly_stochastic(N, N, tau=1.0, n_iters=20, batch_shape=(B,))
    proj = lambda u, tau=1.0: log_sinkhorn(u, n_iters=20, tau=tau).exp()

    ro = LogitReadout(dim)
    u0 = ro(E_x, E_y, P_sq)                                 # (B, N, N) logits
    P0 = proj(u0)
    ds = max((P0.sum(-1) - 1).abs().max().item(), (P0.sum(-2) - 1).abs().max().item())
    check("readout emits logits; Π_S is doubly stochastic",
          u0.shape == (B, N, N) and torch.isfinite(u0).all() and ds < 1e-4, f"marg_err={ds:.1e}")

    # identity at init: W=0, alpha=1 -> u0_hat = log P_t, and Π_S(log P_t) = P_t
    id_err = (proj(ro(E_x, E_y, P_sq)) - P_sq).abs().max().item()
    check("readout identity at init (Π_S(u0_hat) = P_t)", id_err < 1e-4, f"max dev={id_err:.1e}")

    # de-zero for the structural checks
    nn.init.normal_(ro.W, std=0.1)
    ro.alpha.data.fill_(0.7)

    # pair-swap symmetry: exact on logits (symmetric W), near-exact post-projection
    L = ro(E_x, E_y, P_sq)
    L_sw = ro(E_y, E_x, P_sq.transpose(-1, -2))
    logit_err = (L - L_sw.transpose(-1, -2)).abs().max().item()
    post_err = (proj(L) - proj(L_sw).transpose(-1, -2)).abs().max().item()
    check("readout pair-swap symmetry (logits exact)", logit_err < 1e-5 and post_err < 1e-3,
          f"logits={logit_err:.1e} post-proj={post_err:.1e}")

    # external Π_S sharpens with lower tau; gradients flow to W and alpha
    sharper = proj(L, tau=0.05).max(-1).values.mean() > proj(L).max(-1).values.mean()
    ro.zero_grad()
    proj(ro(E_x, E_y, P_sq)).sum().backward()
    grads_ok = ro.W.grad is not None and ro.alpha.grad is not None
    check("readout Π_S tau-sharpen + gradients", bool(sharper) and grads_ok)

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_tests()
