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
import json

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from datasets import build_dataset
from evaluate import apply_override
from models import build_model
from utils.logger import get_root_logger
from utils.options import load_yaml, resolve_experiment_paths
from utils.sinkhorn import safe_log
from networks.mpnn.belief_prop import bp_refine
import os


def _load(config, checkpoint, device, overrides=None):
    """overrides: evaluate.py-style 'dotted.key=value' strings applied before anything is
    built, so eval-set knobs can be switched from the CLI (e.g. DT4D's intra/inter regimes,
    datasets.test.inter_class=false) without copying the YAML."""
    opt = load_yaml(config)
    for spec in (overrides or []):
        apply_override(opt, spec)
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


AXES = ("beta", "sigma", "g", "tau", "delta", "s", "sweeps", "k_cand", "k_graph")


def _cached_sample(model, data, idx, cache_dir, tag):
    """model.sample() memoised to disk as float16. Sampling is the expensive, STOCHASTIC
    part; caching it makes repeat sweeps cheap and — more importantly — scores every
    sweep against a byte-identical baseline, so runs are comparable at the 3rd decimal."""
    path = None if cache_dir is None else os.path.join(cache_dir, f"{tag}_{idx:05d}.npy")
    if path is not None and os.path.exists(path):
        P0 = torch.from_numpy(np.load(path)).float().to(model.device)
        return P0.unsqueeze(0) if P0.dim() == 2 else P0
    F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)         # (1,n,d),(1,n,n)
    P0 = model.sample(F_x, F_y, D_x, D_y)                       # (1,n_y,n_x) DS
    if isinstance(P0, tuple):
        P0 = P0[0]
    if path is not None:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path, P0[0].detach().cpu().numpy().astype(np.float16))
    return P0


@torch.no_grad()
def run(model, test_set, num_pairs, axes, pair_seed=None, cache_dir=None, tag="bp"):
    """Sweep the BP scalars over num_pairs test pairs.

    axes: dict of lists keyed by AXES; the grid is their product, so any subset can be
    swept and the rest held at a single value. pair_seed samples the pairs at random
    instead of taking 0..N-1 -- those are NOT representative, since the pair list is
    product(range(n), repeat=2) and the first n-1 entries all share one source shape."""
    logger = get_root_logger()
    grid = [dict(zip(AXES, v)) for v in itertools.product(*(axes[a] for a in AXES))]
    keys = [tuple(cfg[a] for a in AXES) for cfg in grid]
    varying = [a for a in AXES if len(axes[a]) > 1] or ["beta"]
    base_errs, base_accs, base_argmax = [], [], []
    grid_errs = {k: [] for k in keys}
    grid_accs = {k: [] for k in keys}
    grid_argmax = {k: [] for k in keys}

    N = min(num_pairs, len(test_set))
    if pair_seed is None:
        pairs = list(range(N))
    else:
        rng = np.random.default_rng(pair_seed)
        pairs = sorted(rng.choice(len(test_set), N, replace=False).tolist())
    for idx in tqdm(pairs, desc=f"BP sweep ({len(grid)} grid points)"):
        data = test_set[idx]
        P0 = _cached_sample(model, data, idx, cache_dir, tag)
        F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)
        logits = safe_log(P0)                                  # source=Y rows

        D_x_sparse = data["first"]["sparse"]["dist"]           # (n_x,n_x) for error lookup

        def hungarian(mat):
            ri, ci = linear_sum_assignment(-mat[0].detach().cpu().numpy())
            p = torch.empty(mat.shape[1], dtype=torch.long)
            p[torch.as_tensor(ri)] = torch.as_tensor(ci)
            return p

        def acc(p):
            return (p.cpu() == torch.arange(p.shape[0])).float().mean().item()

        p2p_base = hungarian(P0)
        base_errs.append(_pair_error(p2p_base, D_x_sparse))
        base_accs.append(acc(p2p_base))
        # pre-Hungarian accuracy: the assignment constraint repairs a lot of exactly the
        # local incoherence BP targets, so a flat Hungarian column with a rising argmax
        # column means "BP works but is redundant with Hungarian", NOT "BP does nothing".
        base_argmax.append(acc(P0[0].argmax(-1)))

        for cfg, key in zip(grid, keys):
            ref = bp_refine(logits, F_y, F_x, D_y, D_x,        # variables on Y: src feats/metric = Y
                            k_logit=cfg["k_cand"], k_feat=cfg["k_cand"],
                            k_graph=cfg["k_graph"], n_sweeps=cfg["sweeps"],
                            beta=cfg["beta"], sigma=cfg["sigma"], g=cfg["g"],
                            tau=cfg["tau"], delta=cfg["delta"], s=cfg["s"])
            p2p = hungarian(ref)
            grid_errs[key].append(_pair_error(p2p, D_x_sparse))
            grid_accs[key].append(acc(p2p))
            grid_argmax[key].append(acc(ref[0].argmax(-1)))

    base_per_pair = np.array([e.mean() for e in base_errs])
    base_err = float(np.concatenate(base_errs).mean())
    base_acc = float(np.mean(base_accs))
    base_am = float(np.mean(base_argmax))
    rows = {}
    # only the axes actually swept get a column, so a 6-axis sweep stays readable
    head = " ".join(f"{a:>8}" for a in varying)
    logger.info(f"\n{head} {'err':>9} {'d_err':>9} {'acc':>7} {'d_acc':>7} {'win':>6} "
                f"{'argmax':>7}")
    logger.info(f"{'BASE':>{len(head)}} {base_err:>9.4f} {'':>9} {base_acc:>7.3f} "
                f"{'':>7} {'':>6} {base_am:>7.3f}")
    best = (base_err, None)
    for cfg, key in zip(grid, keys):
        e = float(np.concatenate(grid_errs[key]).mean())
        a = float(np.mean(grid_accs[key]))
        am = float(np.mean(grid_argmax[key]))
        # paired per-pair win rate: guards against a mean driven by a few rescued disasters
        per_pair = np.array([x.mean() for x in grid_errs[key]])
        win = float((per_pair < base_per_pair).mean())
        flag = "  <-- best" if e < best[0] else ""
        if e < best[0]:
            best = (e, key)
        rows[str(key)] = {**cfg, "err": e, "d_err": e - base_err, "acc": a,
                          "d_acc": a - base_acc, "win_rate": win, "argmax_acc": am,
                          # per-pair, so CIs / plots / win rates can be recomputed without
                          # re-running the sweep
                          "per_pair_err": per_pair.tolist(),
                          "per_pair_acc": grid_accs[key]}
        vals = " ".join(f"{cfg[a]:>8g}" for a in varying)
        logger.info(f"{vals} {e:>9.4f} {e - base_err:>+9.4f} "
                    f"{a:>7.3f} {a - base_acc:>+7.3f} {win:>6.2f} {am:>7.3f}{flag}")

    if best[1] is None:
        logger.info("\nNO-GO: no (beta, sigma, g) beat the baseline. BP does not help here.")
    else:
        b = rows[str(best[1])]
        logger.info(f"\nGO: best {best[1]} -> err {best[0]:.4f} "
                    f"(baseline {base_err:.4f}, improvement {base_err - best[0]:+.4f}); "
                    f"improves {b['win_rate']:.2f} of pairs; argmax acc {base_am:.3f} -> "
                    f"{b['argmax_acc']:.3f}")
    summary = {"num_pairs": len(pairs), "pair_seed": pair_seed, "axes": axes,
               "pairs": list(pairs),
               "base": {"err": base_err, "acc": base_acc, "argmax_acc": base_am,
                        "per_pair_err": base_per_pair.tolist(),
                        "per_pair_acc": base_accs},
               "best": None if best[1] is None else str(best[1]), "grid": rows}
    return best, summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num_pairs", type=int, default=20)
    p.add_argument("--device", default=None)
    p.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    p.add_argument("--gs", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    p.add_argument("--k_cand", type=int, nargs="+", default=None,
                   help="candidates per vertex (Kc = 2x this). Default: the denoiser's "
                        "k_feat, so the prototype's candidate sets match the cross stage's")
    p.add_argument("--k_graph", type=int, nargs="+", default=None,
                   help="message-graph degree. Default: the denoiser's k_intra -- in-loop, "
                        "denoiser.py hands BP the ALREADY-BUILT intra graph, so tuning at a "
                        "different degree would not transfer (geometric evidence scales with k)")
    p.add_argument("--sweeps", type=int, nargs="+", default=[3],
                   help="BP sweeps; pass several (e.g. 1 2 4 8 16) for the test-time "
                        "inference-scaling curve, the figure that separates BP from an MPNN")
    p.add_argument("--taus", type=float, nargs="+", default=[1.0],
                   help="semiring temperature: 1 = sum-product, ->0 approaches max-product")
    p.add_argument("--deltas", type=float, nargs="+", default=[4.0],
                   help="truncation in nats; with sigma sets the capture radius sqrt(2*delta)*sigma")
    p.add_argument("--slacks", type=float, nargs="+", default=[-4.0],
                   help="slack (abstention) score, on the raw logit scale -- couples to beta")
    p.add_argument("--pair_seed", type=int, default=None,
                   help="sample --num_pairs test pairs at random with this seed; without "
                        "it the first N pairs are used, and those all share one source "
                        "shape (product(range(n), repeat=2)) -- not representative")
    p.add_argument("--json", default=None, help="write the summary dict here")
    p.add_argument("--cache_dir", default=None,
                   help="memoise model.sample() here (float16, ~0.5MB/pair); makes repeat "
                        "sweeps cheap and pins the baseline so runs are comparable")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, repeatable (e.g. --set datasets.test.inter_class=false)")
    p.add_argument("--cache_tag", default="bp",
                   help="prefix for cache files; CHANGE IT when the checkpoint or config "
                        "changes, or stale samples will be reused silently")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model, test_set, ckpt = _load(args.config, args.checkpoint, args.device, args.overrides)
    N = min(args.num_pairs, len(test_set))
    # Default the two structural degrees to the denoiser's own, so the post-process
    # prototype is tuned on the graph the in-loop version will actually be handed.
    den = (load_yaml(args.config).get("networks") or {}).get("denoiser") or {}
    k_graph = args.k_graph or [den.get("k_intra", 8)]
    k_cand = args.k_cand or [den.get("k_feat", 8)]
    axes = {"beta": args.betas, "sigma": args.sigmas, "g": args.gs, "tau": args.taus,
            "delta": args.deltas, "s": args.slacks, "sweeps": args.sweeps,
            "k_cand": k_cand, "k_graph": k_graph}
    n_grid = int(np.prod([len(v) for v in axes.values()]))
    get_root_logger().info(f"k_graph={k_graph} (denoiser k_intra={den.get('k_intra')}), "
                           f"k_cand={k_cand} (denoiser k_feat={den.get('k_feat')})")
    get_root_logger().info(f"loaded {ckpt}; {len(test_set)} test pairs, sweeping "
                           f"{n_grid} grid points on {N} pairs "
                           f"({'random, seed ' + str(args.pair_seed) if args.pair_seed is not None else 'sequential'})")
    if args.pair_seed is None and N < len(test_set):
        get_root_logger().info("WARNING: sequential pairs are one source shape vs the rest; "
                               "pass --pair_seed or run all pairs for a representative number")
    _, summary = run(model, test_set, args.num_pairs, axes, args.pair_seed,
                     args.cache_dir, args.cache_tag)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        get_root_logger().info(f"summary -> {args.json}")
