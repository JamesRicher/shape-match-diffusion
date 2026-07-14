"""Measure how stochastic / varied the diffusion matcher's samples are, for any checkpoint.

For each test pair we draw K independent samples (each a full DDIM reverse from its own
randn prior -- the ONLY source of stochasticity, since the sampler is DDIM eta=0) and
report:

  Endpoint (final-output) metrics
    - endpoint divergence: mean pairwise fraction of sparse points mapped differently
      across the K final Hungarian-snapped maps. 0 = every draw identical (collapsed /
      prior ignored); ->1 = draws disagree everywhere. Same quantity as the model's
      trajectory_divergence diagnostic, but averaged over many pairs.
    - per-point match entropy: for each Y point, entropy of its empirical match
      distribution over the K draws (nats, and normalised by log K). Localises WHERE the
      uncertainty is (expect high at symmetric regions, ~0 at distinctive landmarks).
    - best-of-K vs mean sparse geodesic error: does drawing more samples ever recover a
      much better solution (genuine multi-modality) or not?

  Per-step curve
    - divergence as a function of DDIM step t, from the running predict-x0 snaps. Shows
      WHEN trajectories commit: a late split into a few stable levels = structured modes
      (healthy on symmetric inputs); flat-high = noise; ->0 early = convergence/collapse.

Interpretation caveat: low variety is only a failure for ambiguous/symmetric inputs. For
a confident asymmetric match, ~0 divergence is correct. Read divergence jointly with
error (the variety-vs-error scatter): healthy variety keeps error low (symmetry modes),
noise does not.

Outputs (to --out-dir, default experiments/<name>/results/variety/):
  summary.json                  all scalar metrics + the per-step curve
  per_step_divergence.png       divergence vs DDIM step (mean +/- std across pairs)
  variety_vs_error.png          per-pair endpoint divergence vs mean sparse geo error

Examples:
  python -m vis.sample_variety -c configs/faust_matrix_diffusion.yaml --samples 16 --num-pairs 30
  python -m vis.sample_variety -c configs/smal_matrix_diffusion.yaml \
      --checkpoint experiments/smal_matrix_diffusion/models/latest.pth --samples 8
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import build_dataset
from models import build_model
from models.base_model import to_numpy
from train import _single_collate, autofill_feat_dim, move_to_device
from utils.logger import get_root_logger
from utils.options import load_yaml, resolve_experiment_paths


# --------------------------------------------------------------------------- #
# options (mirrors evaluate.py: force inference mode, net-only checkpoint load)
# --------------------------------------------------------------------------- #
def build_opt(args):
    opt = load_yaml(args.config)
    if args.name is not None:
        opt["name"] = args.name
    if args.device is not None:
        opt["device"] = args.device
    opt["is_train"] = False
    resolve_experiment_paths(opt)
    ckpt = args.checkpoint or os.path.join(opt["path"]["models"], "final.pth")
    opt["path"]["resume_state"] = ckpt
    opt["path"]["resume"] = False   # net-only load; don't restore optimizer/epoch state
    return opt, ckpt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", required=True, help="the YAML config used for training")
    p.add_argument("-n", "--name", default=None, help="override experiment name")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint to load (default: experiments/<name>/models/final.pth)")
    p.add_argument("--device", default=None, help="'cuda'/'cpu'; auto-detected when omitted")
    p.add_argument("--samples", type=int, default=8, help="K draws per pair (default 8)")
    p.add_argument("--num-pairs", type=int, default=20,
                   help="pairs to scan, evenly spaced over the test set (0 = all)")
    p.add_argument("--steps", type=int, default=None, help="override DDIM sample_steps")
    p.add_argument("--seed", type=int, default=0, help="torch seed for the priors (reproducibility)")
    p.add_argument("--out-dir", default=None, help="output dir (default results/variety/)")
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# per-pair / aggregate metrics
# --------------------------------------------------------------------------- #
def _pairwise_disagreement(maps: np.ndarray) -> float:
    """Mean over sample pairs (i<j) of the fraction of points mapped differently.
    maps: (K, n) hard p2p per sample. Returns a scalar in [0, 1]."""
    K = maps.shape[0]
    if K < 2:
        return 0.0
    d = [np.mean(maps[i] != maps[j]) for i in range(K) for j in range(i + 1, K)]
    return float(np.mean(d))


def _per_point_entropy(maps: np.ndarray):
    """Mean per-point empirical match entropy over the K draws.
    maps: (K, n). Returns (mean nats, mean normalised-by-logK) scalars."""
    K, n = maps.shape
    ents = np.empty(n)
    for j in range(n):
        _, counts = np.unique(maps[:, j], return_counts=True)
        p = counts / K
        ents[j] = -(p * np.log(p)).sum()
    norm = np.log(K) if K > 1 else 1.0
    return float(ents.mean()), float(ents.mean() / norm)


def analyse_pair(model, data, K, steps):
    """Draw K samples for one pair; return (endpoint metrics dict, per-step divergence
    curve (steps,), t values (steps,))."""
    F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)
    n = F_x.shape[1]
    rows = np.arange(n)
    D_x0 = to_numpy(D_x[0])                                   # (n, n) sparse geodesic on X

    final_maps, step_maps, errs = [], [], []
    ts_np = None
    for _ in range(K):
        P0, traj, ts = model.sample(F_x, F_y, D_x, D_y, steps=steps, return_trajectory=True)
        # endpoint: Hungarian snap (accurate) of the final DS matrix
        row_ind, col_ind = linear_sum_assignment(-to_numpy(P0[0]))
        p = np.empty(n, dtype=int); p[row_ind] = col_ind
        final_maps.append(p)
        errs.append(float(D_x0[rows, p].mean()))             # sparse geo err (gt = identity)
        step_maps.append(to_numpy(traj[0]))                  # (steps, n) running argmax snaps
        ts_np = to_numpy(ts)

    final_maps = np.stack(final_maps)                        # (K, n)
    step_maps = np.stack(step_maps)                          # (K, steps, n)
    ent_nats, ent_norm = _per_point_entropy(final_maps)
    endpoint = {
        "endpoint_divergence": _pairwise_disagreement(final_maps),
        "per_point_entropy_nats": ent_nats,
        "per_point_entropy_norm": ent_norm,
        "mean_error": float(np.mean(errs)),
        "best_of_k_error": float(np.min(errs)),
        "worst_of_k_error": float(np.max(errs)),
    }
    step_curve = np.array([_pairwise_disagreement(step_maps[:, s, :])
                           for s in range(step_maps.shape[1])])
    return endpoint, step_curve, ts_np


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    logger = get_root_logger()
    opt, ckpt = build_opt(args)
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    test_set = build_dataset(opt["datasets"]["test"])
    autofill_feat_dim(opt, int(test_set[0]["first"]["feat"].shape[-1]))
    model = build_model(opt)
    model.eval()

    # evenly spaced, deterministic subset of pairs
    n_total = len(test_set)
    if args.num_pairs and args.num_pairs < n_total:
        stride = max(1, n_total // args.num_pairs)
        indices = list(range(0, n_total, stride))[:args.num_pairs]
    else:
        indices = list(range(n_total))

    logger.info(f'Sample-variety for "{opt["name"]}" | ckpt={ckpt} | '
                f"{len(indices)} pairs x K={args.samples} draws (device: {model.device}).")

    endpoints, curves, ts_ref = [], [], None
    for idx in tqdm(indices, desc="variety"):
        data = move_to_device(test_set[int(idx)], model.device)
        ep, curve, ts = analyse_pair(model, data, args.samples, args.steps)
        endpoints.append(ep)
        curves.append(curve)
        ts_ref = ts if ts_ref is None else ts_ref

    # aggregate endpoint scalars (mean over pairs)
    keys = endpoints[0].keys()
    agg = {k: float(np.mean([e[k] for e in endpoints])) for k in keys}
    curves = np.stack(curves)                                # (pairs, steps)
    curve_mean, curve_std = curves.mean(0), curves.std(0)

    out_dir = args.out_dir or os.path.join(opt["path"]["results"], "variety")
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "name": opt["name"], "checkpoint": ckpt,
        "num_pairs": len(indices), "samples_per_pair": args.samples,
        "steps": int(len(ts_ref)),
        **agg,
        "per_step_t": [float(t) for t in ts_ref],
        "per_step_divergence_mean": [float(v) for v in curve_mean],
        "per_step_divergence_std": [float(v) for v in curve_std],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- per-step divergence curve (x-axis = t from 1 -> 0, i.e. noise -> clean) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ts_ref, curve_mean, "-o", ms=3, label="mean over pairs")
    ax.fill_between(ts_ref, curve_mean - curve_std, curve_mean + curve_std, alpha=0.25)
    ax.set_xlabel("DDIM time t  (1 = prior, 0 = clean)")
    ax.set_ylabel("sample divergence (frac. points disagreeing)")
    ax.set_title(f'{opt["name"]}: sample divergence vs step')
    ax.invert_xaxis()                                        # reverse process runs t: 1 -> 0
    ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend()
    fig.savefig(os.path.join(out_dir, "per_step_divergence.png"), bbox_inches="tight")
    plt.close(fig)

    # --- variety vs error scatter (per pair) ---
    dv = [e["endpoint_divergence"] for e in endpoints]
    er = [e["mean_error"] for e in endpoints]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(dv, er, s=20, alpha=0.7)
    ax.set_xlabel("endpoint divergence"); ax.set_ylabel("mean sparse geo error")
    ax.set_title(f'{opt["name"]}: variety vs error (per pair)')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(out_dir, "variety_vs_error.png"), bbox_inches="tight")
    plt.close(fig)

    logger.info(f"endpoint divergence: {agg['endpoint_divergence']:.3f}  "
                f"per-point entropy: {agg['per_point_entropy_nats']:.3f} nats "
                f"({agg['per_point_entropy_norm']:.3f} of max)")
    logger.info(f"error  mean={agg['mean_error']:.4f}  best-of-K={agg['best_of_k_error']:.4f}")
    logger.info(f"per-step divergence: prior(t=1)={curve_mean[0]:.3f} -> "
                f"clean(t~0)={curve_mean[-1]:.3f}")
    logger.info(f"Wrote variety report to {out_dir}")


if __name__ == "__main__":
    main()
