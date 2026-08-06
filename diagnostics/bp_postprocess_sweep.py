"""BP post-processing go/no-go sweep (notes/BP-idea.md build step 1).

Runs the standalone single-sided belief-propagation refiner
(networks/mpnn/belief_prop.bp_refine) on a TRAINED model's final assignment logits —
no training, no gradients — and sweeps the key scalars (beta, sigma, g) against sparse
geodesic error. If no (beta, sigma, g) window beats the un-refined baseline, that is a
clean negative ablation: stop before wiring BP into the denoiser. If a window helps,
those values seed the differentiable in-loop version.

Works with ANY trained matcher whose model exposes `sample()` and `_sparse_inputs()`
(the transformer MatrixDiffusionModel or the MPNN twin) — BP is a pure post-process on
the doubly-stochastic output, independent of which denoiser produced it.

Variables are placed on Y (rows of P_t, the repo convention); source metric = D_y,
labels + target metric from X. Hungarian + sparse geodesic error mirror
MatrixDiffusionModel.validate_single exactly, so the baseline column reproduces the
model's normal sparse number.

Usage:
    python -m diagnostics.bp_postprocess_sweep -c <config.yaml> [--checkpoint <pth>] \
        [--num_pairs 20] [--device cpu]
"""
import argparse
import itertools

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from datasets import build_dataset
from models import build_model
from utils.logger import get_root_logger
from utils.options import load_yaml, resolve_experiment_paths
from utils.sinkhorn import safe_log
from networks.mpnn.belief_prop import bp_refine
import os


def _load(config, checkpoint, device):
    opt = load_yaml(config)
    if device:
        opt["device"] = device
    opt["is_train"] = False
    resolve_experiment_paths(opt)
    ckpt = checkpoint or os.path.join(opt["path"]["models"], "final.pth")
    opt["path"]["resume_state"] = ckpt
    opt["path"]["resume"] = False
    model = build_model(opt)
    model.eval()
    test_set = build_dataset(opt["datasets"]["test"])
    return model, test_set, ckpt


@torch.no_grad()
def _pair_error(p2p, D_x_sparse):
    """Sparse geodesic error of a Y->X map, as in validate_single (D_x rows = true match)."""
    n = p2p.shape[0]
    rows = torch.arange(n)
    return D_x_sparse[rows, p2p].cpu().numpy()


@torch.no_grad()
def run(model, test_set, num_pairs, betas, sigmas, gs, k_cand, k_graph, sweeps):
    logger = get_root_logger()
    device = model.device
    grid = list(itertools.product(betas, sigmas, gs))
    base_errs, base_accs = [], []
    grid_errs = {key: [] for key in grid}
    grid_accs = {key: [] for key in grid}

    N = min(num_pairs, len(test_set))
    for idx in range(N):
        data = test_set[idx]
        F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)     # (1,n,d),(1,n,n)
        P0 = model.sample(F_x, F_y, D_x, D_y)                   # (1,n_y,n_x) DS
        if isinstance(P0, tuple):
            P0 = P0[0]
        logits = safe_log(P0)                                  # source=Y rows

        D_x_sparse = data["first"]["sparse"]["dist"]           # (n_x,n_x) for error lookup

        def hungarian(mat):
            ri, ci = linear_sum_assignment(-mat[0].detach().cpu().numpy())
            p = torch.empty(mat.shape[1], dtype=torch.long)
            p[torch.as_tensor(ri)] = torch.as_tensor(ci)
            return p

        p2p_base = hungarian(P0)
        base_errs.append(_pair_error(p2p_base, D_x_sparse))
        base_accs.append((p2p_base == torch.arange(p2p_base.shape[0])).float().mean().item())

        for (beta, sigma, g) in grid:
            ref = bp_refine(logits, F_y, F_x, D_y, D_x,        # variables on Y: src feats/metric = Y
                            k_logit=k_cand, k_feat=k_cand, k_graph=k_graph,
                            beta=beta, sigma=sigma, g=g, n_sweeps=sweeps)
            p2p = hungarian(ref)
            grid_errs[(beta, sigma, g)].append(_pair_error(p2p, D_x_sparse))
            grid_accs[(beta, sigma, g)].append(
                (p2p == torch.arange(p2p.shape[0])).float().mean().item())
        logger.info(f"pair {idx + 1}/{N} done")

    base_err = float(np.concatenate(base_errs).mean())
    base_acc = float(np.mean(base_accs))
    logger.info(f"\n{'beta':>6} {'sigma':>7} {'g':>6} {'err':>9} {'d_err':>9} {'acc':>7} {'d_acc':>7}")
    logger.info(f"{'BASE':>6} {'':>7} {'':>6} {base_err:>9.4f} {'':>9} {base_acc:>7.3f}")
    best = (base_err, None)
    for key in grid:
        e = float(np.concatenate(grid_errs[key]).mean())
        a = float(np.mean(grid_accs[key]))
        beta, sigma, g = key
        flag = "  <-- best" if e < best[0] else ""
        if e < best[0]:
            best = (e, key)
        logger.info(f"{beta:>6.2f} {sigma:>7.4f} {g:>6.2f} {e:>9.4f} {e - base_err:>+9.4f} "
                    f"{a:>7.3f} {a - base_acc:>+7.3f}{flag}")

    if best[1] is None:
        logger.info("\nNO-GO: no (beta, sigma, g) beat the baseline. BP does not help here.")
    else:
        logger.info(f"\nGO: best {best[1]} -> err {best[0]:.4f} "
                    f"(baseline {base_err:.4f}, improvement {base_err - best[0]:+.4f}).")
    return best


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num_pairs", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    p.add_argument("--gs", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--k_cand", type=int, default=8)
    p.add_argument("--k_graph", type=int, default=8)
    p.add_argument("--sweeps", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, test_set, ckpt = _load(args.config, args.checkpoint, args.device)
    get_root_logger().info(f"loaded {ckpt}; {len(test_set)} test pairs, sweeping "
                           f"{len(args.betas)}x{len(args.sigmas)}x{len(args.gs)} grid "
                           f"on {min(args.num_pairs, len(test_set))} pairs")
    run(model, test_set, args.num_pairs, args.betas, args.sigmas, args.gs,
        args.k_cand, args.k_graph, args.sweeps)
