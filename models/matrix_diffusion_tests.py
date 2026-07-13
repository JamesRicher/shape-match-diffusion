"""Smoke + overfit checks for MatrixDiffusionModel (no dataset needed).

Covers: training step runs and back-props, sampler emits a doubly-stochastic matrix, and
one isometric synthetic pair (shared features) can be overfit so Hungarian recovers the
identity permutation -- the Phase-3 gate in miniature.
Run: python -m models.matrix_diffusion_tests
"""
import numpy as np
import torch

from models import build_model


def _opt():
    return {
        'is_train': True, 'device': 'cpu', 'model_type': 'MatrixDiffusionModel',
        'networks': {'denoiser': {'type': 'MatrixDenoiser', 'feat_dim': 8, 'dim': 64,
                                  'heads': 4, 'depth': 3, 'n_anchors': 4}},
        'train': {'optims': {'denoiser': {'type': 'Adam', 'lr': 3e-3}}},
        'diffusion': {'eta': 0.1, 'proj_iters': 6, 'sample_steps': 20},
    }


def _pair(n=16, d=8, isometric=False):
    """Synthetic sparse pair matching the dataset schema. isometric=True makes Y an
    isometric copy of X with shared features, so identity is learnable."""
    pts_x = torch.randn(n, 3)
    D_x = torch.cdist(pts_x, pts_x); D_x = 0.5 * (D_x + D_x.T)
    F_x = torch.randn(n, d)
    if isometric:
        F_y, D_y = F_x.clone(), D_x.clone()
    else:
        pts_y = torch.randn(n, 3)
        D_y = torch.cdist(pts_y, pts_y); D_y = 0.5 * (D_y + D_y.T)
        F_y = torch.randn(n, d)
    mk = lambda F, D, P: {'sparse': {'feat': F, 'dist': D, 'verts': P, 'idx': torch.arange(n)}}
    return {'first': mk(F_x, D_x, pts_x), 'second': mk(F_y, D_y,
            pts_x if isometric else torch.randn(n, 3)), 'gt_perm': torch.eye(n)}


def _report(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name:<40}" + (f"  {detail}" if detail else ""))
    return ok


def main():
    torch.manual_seed(0)
    results = []

    model = build_model(_opt())

    # --- training step runs, loss finite, grads reach the denoiser ---------- #
    data = _pair()
    model.feed_data(data)
    loss0 = model.loss_metrics['l_ce']
    model.optimize_parameters()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all()
                  for p in model.networks['denoiser'].parameters())
    results.append(_report("training step: finite loss + grads flow",
                           torch.isfinite(loss0).item() and grad_ok, f"loss={loss0.item():.3f}"))

    # --- sampler emits a doubly-stochastic matrix --------------------------- #
    F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)
    P0 = model.sample(F_x, F_y, D_x, D_y)
    ds = max((P0.sum(-1) - 1).abs().max().item(), (P0.sum(-2) - 1).abs().max().item())
    results.append(_report("sampler output doubly stochastic", ds < 1e-3, f"marg_err={ds:.1e}"))

    # --- overfit one isometric pair -> Hungarian recovers identity ---------- #
    torch.manual_seed(1)
    model = build_model(_opt())
    pair = _pair(isometric=True)
    model.feed_data(pair); l_start = model.loss_metrics['l_ce'].item()
    for _ in range(400):
        model.feed_data(pair)
        model.optimize_parameters()
    l_end = model.loss_metrics['l_ce'].item()
    p2p = model.validate_single(pair)
    acc = (p2p == torch.arange(p2p.shape[0])).float().mean().item()
    results.append(_report("overfit: loss drops", l_end < 0.5 * l_start, f"{l_start:.3f} -> {l_end:.3f}"))
    results.append(_report("overfit: Hungarian recovers identity", acc >= 0.8, f"acc={acc:.2f}"))

    # --- diagnostics run and return valid outputs -------------------------------------- #
    # NB: this synthetic pair shares features (F_y = F_x), so the readout solves the match
    # from features alone and the loss is ~0 at every t -> slope ~= 0 here. The slope's
    # SIGN (should be > 0) is the real-data signal, tested on an actual FAUST pair, not on
    # this degenerate case; here we only assert the diagnostic runs and is finite/monotone-
    # able. Likewise divergence ~= 0 (a memorised non-symmetric map has no modes to spread).
    curve = model.loss_vs_t(pair, n_bins=6, repeats=8)
    vals = list(curve.values())
    slope = float(np.mean(vals[len(vals) // 2:]) - np.mean(vals[:len(vals) // 2]))
    slope_ok = len(curve) == 6 and np.isfinite(slope) and all(v >= 0 for v in vals)
    results.append(_report("diag: loss-vs-t runs (finite, 6 bins)", slope_ok, f"slope={slope:+.3f}"))

    div = model.trajectory_divergence(pair, n_samples=4)
    results.append(_report("diag: trajectory divergence in [0,1]", 0.0 <= div <= 1.0, f"div={div:.3f}"))

    print(f"\n{sum(results)}/{len(results)} checks passed")
    if sum(results) != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
