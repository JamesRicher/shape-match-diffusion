"""BP as a pure post-process on a trained checkpoint: report the dense MGE it buys.

Where `diagnostics/bp_postprocess_sweep.py` SWEEPS the BP scalars against SPARSE geodesic
error to find a working window, this script takes the derived scalars as given, applies
`belief_prop.bp_refine` once to each sampled assignment, and reports the numbers that
actually go in a table: sparse error/accuracy AND the dense whole-shape MGE + PCK AUC
after densification. No learning, no gradients — the checkpoint is untouched.

That distinction matters: the go/no-go sweep measured sparse error, and a sparse gain does
not automatically survive densification (near-misses are cheap against the quantisation
floor, cf. diagnostics/densify_floor.py). This is the check that it does.

Method. BP is injected by wrapping `model.validate_single`, the single point where every
reporting path (sparse error, densifier, PCK curves) gets its sparse map. Everything
downstream is then the model's own `validation()` verbatim, so the baseline column
reproduces `evaluate.py` exactly and the two columns differ ONLY by the refinement.

Variables are placed on Y (rows of P_t, the repo convention): source features/metric come
from Y, labels and the label metric from X — matching bp_postprocess_sweep.

Sparse-sampling regimes (--regime), the two evaluate.py reports:
  bijective   -- sparse Y point j is the GT image of sparse X point j, so sparse
                 avg_error/acc are defined. GT enters point SELECTION, so the dense MGE it
                 also yields is optimistic; a dev diagnostic, not a table number.
  independent -- each shape FPS'd on its own geometry, no GT in selection: the honest
                 regime evaluate.py reports dense MGE under. No bijective sparse target
                 exists, so only dense MGE/AUC are produced (needs a densifier).
Both columns of a comparison always share one regime, so the base->BP delta is meaningful
in either; only the absolute numbers differ in what they mean.

Usage:
    python -m diagnostics.bp_postprocess_eval -c <config.yaml> [--checkpoint <pth>]
        [--regime both] [--beta 0.5 --sigma 0.05 --g 4.0] [--num_pairs 40]
        [--pair_split report] [--set datasets.test.inter_class=false] [--device cuda]

Defaults are the derived scalars (notes/BP-loop-design.md): the sigma calibration's global
0.05 and the Stage-A sweep optimum beta 0.5 / g 4.0. Per-dataset optima differ (DT4D-inter
preferred beta 0.25, g 16) — pass them explicitly rather than trusting one global default.
"""
import argparse
import json
import os
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets import build_dataset
from evaluate import apply_override
from models import build_model
from networks.mpnn.belief_prop import bp_refine
from utils.logger import get_root_logger
from utils.options import load_yaml, resolve_experiment_paths
from utils.sinkhorn import safe_log
from diagnostics.bp_postprocess_sweep import _index_pool

# batch_size=1 collate makes the sparse operators IPC-safe; _RestoringLoader rebuilds them
# in the main process. Both are needed — the loader alone yields un-restored tuples.
from train import _RestoringLoader, _single_collate


def _load(config, checkpoint, device, overrides, dense):
    opt = load_yaml(config)
    for spec in (overrides or []):
        apply_override(opt, spec)
    if device:
        opt["device"] = device
    opt["is_train"] = False
    # MGE is the point of this script, so dense reporting is forced on regardless of the
    # config's training-time eval block (which usually has dense: false for speed).
    opt.setdefault("eval", {})
    opt["eval"]["sparse"] = True
    opt["eval"]["dense"] = bool(dense)
    if dense and not opt.get("densifier"):
        raise ValueError("dense MGE needs opt['densifier'] in the config; pass --no_dense "
                         "to report sparse error only")
    resolve_experiment_paths(opt)
    ckpt = checkpoint or os.path.join(opt["path"]["models"], "final.pth")
    opt["path"]["resume_state"] = ckpt
    opt["path"]["resume"] = False
    model = build_model(opt)
    model.eval()
    return opt, model, ckpt


def attach_bp(model, params):
    """Wrap validate_single so the sampled logits are BP-refined before Hungarian.

    Returns a restore() that puts the original method back, so baseline and BP passes can
    run against one loaded model. Refinement is a pure function of the sample — nothing in
    the model's state is mutated.
    """
    from scipy.optimize import linear_sum_assignment
    original = model.validate_single

    @torch.no_grad()
    def refined(self, data):
        F_x, F_y, D_x, D_y, _ = self._sparse_inputs(data)
        P0 = self.sample(F_x, F_y, D_x, D_y)[0]                  # (n_y, n_x)
        logits = safe_log(P0).unsqueeze(0)                       # variables on Y
        ref = bp_refine(logits, F_y, F_x, D_y, D_x,              # src feats/metric = Y
                        k_logit=params["k_cand"], k_feat=params["k_cand"],
                        k_graph=params["k_graph"], n_sweeps=params["sweeps"],
                        beta=params["beta"], sigma=params["sigma"], g=params["g"],
                        tau=params["tau"], delta=params["delta"], s=params["s"])[0]
        row, col = linear_sum_assignment(-ref.detach().cpu().numpy())
        p2p = torch.empty(ref.shape[0], dtype=torch.long)
        p2p[torch.as_tensor(row)] = torch.as_tensor(col)
        return p2p

    model.validate_single = types.MethodType(refined, model)
    return lambda: setattr(model, "validate_single", original)


def _loader(dataset, num_pairs, pair_split, pair_seed):
    pool = _index_pool(len(dataset), pair_split)
    N = min(num_pairs, len(pool)) if num_pairs else len(pool)
    if pair_seed is None:
        pairs = pool[:N]
    else:
        rng = np.random.default_rng(pair_seed)
        pairs = sorted(rng.choice(pool, N, replace=False).tolist())
    loader = DataLoader(Subset(dataset, pairs), batch_size=1, shuffle=False,
                        collate_fn=_single_collate)
    return _RestoringLoader(loader), pairs


def run_regime(model, dataset, loader, regime, out_dir, params, no_baseline, dense):
    """One baseline-vs-BP comparison under a single sparse-sampling regime (see module doc).

    Returns {'base': metrics, 'bp': metrics}; 'base' is omitted with no_baseline. The honest
    regime has no bijective sparse GT, so sparse reporting is off there and only dense MGE
    comes back."""
    logger = get_root_logger()
    dataset.independent_fps = (regime == "independent")
    model.report_sparse = (regime == "bijective")
    model.report_dense = dense
    out = {}

    if not no_baseline:
        logger.info(f"--- [{regime}] baseline (no BP) ---")
        out["base"] = model.validation(loader, out_dir=os.path.join(out_dir, regime, "base"))

    logger.info(f"--- [{regime}] BP post-process ---")
    restore = attach_bp(model, params)
    try:
        out["bp"] = model.validation(loader, out_dir=os.path.join(out_dir, regime, "bp"))
    finally:
        restore()
    return out


def report(results, regime):
    """Format one regime's base -> BP deltas; metrics the regime doesn't produce are skipped."""
    keys = [("avg_error", "sparse err", -1), ("acc", "sparse acc", +1),
            ("dense_error", "dense MGE", -1), ("auc", "dense AUC", +1)]
    res = results[regime]
    if "base" not in res:
        return f"[{regime}] BP: " + json.dumps({k: res['bp'].get(k) for k, _, _ in keys})
    rows = []
    for key, label, sign in keys:
        b, r = res["base"].get(key), res["bp"].get(key)
        if b is None or r is None:
            continue
        rel = (r - b) / b * 100 if b else float("nan")
        better = "better" if sign * (r - b) > 0 else "worse"
        rows.append(f"  {label:11s} {b:.4f} -> {r:.4f}  ({rel:+.1f}%, {better})")
    return f"[{regime}] BP post-process vs baseline:\n" + "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--checkpoint", default=None, help="defaults to <models>/final.pth")
    p.add_argument("--device", default=None)
    p.add_argument("--set", action="append", dest="overrides", metavar="KEY=VALUE",
                   help="evaluate.py-style dotted override, applied before anything is built")
    p.add_argument("--num_pairs", type=int, default=0, help="0 = the whole (split) pool")
    p.add_argument("--pair_split", default="report", choices=["report", "tune", "none"],
                   help="deterministic half of the pair list; 'report' is the half NOT "
                        "used to select the scalars")
    p.add_argument("--pair_seed", type=int, default=None)
    p.add_argument("--regime", default="bijective",
                   choices=["bijective", "independent", "both"],
                   help="sparse-sampling regime(s) to evaluate under (see module doc); "
                        "'independent' is evaluate.py's honest dense-MGE setup")
    p.add_argument("--no_dense", action="store_true", help="sparse error only (no MGE)")
    p.add_argument("--no_baseline", action="store_true", help="skip the un-refined pass")
    p.add_argument("--out", default="diagnostics/results/bp_post_eval")
    p.add_argument("--tag", default=None, help="output filename stem (defaults to the run name)")
    # the derived scalars
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--g", type=float, default=4.0)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=4.0)
    p.add_argument("--s", type=float, default=-4.0)
    p.add_argument("--sweeps", type=int, default=3)
    p.add_argument("--k_cand", type=int, default=10)
    p.add_argument("--k_graph", type=int, default=12)
    args = p.parse_args()

    params = {k: getattr(args, k) for k in
              ("beta", "sigma", "g", "tau", "delta", "s", "sweeps", "k_cand", "k_graph")}
    regimes = ["bijective", "independent"] if args.regime == "both" else [args.regime]
    if "independent" in regimes and args.no_dense:
        p.error("--regime independent has only the dense MGE to report; drop --no_dense")

    opt, model, ckpt = _load(args.config, args.checkpoint, args.device, args.overrides,
                             dense=not args.no_dense)
    logger = get_root_logger()
    logger.info(f"checkpoint: {ckpt}")
    logger.info(f"BP params: {params}")

    dataset = build_dataset(opt["datasets"]["test"])
    loader, pairs = _loader(dataset, args.num_pairs, args.pair_split, args.pair_seed)
    logger.info(f"{len(pairs)} pairs (split={args.pair_split}) of {len(dataset)}")

    out_dir = os.path.join(args.out, args.tag or opt["name"])
    os.makedirs(out_dir, exist_ok=True)
    results = {"checkpoint": ckpt, "config": args.config, "params": params,
               "pairs": pairs, "pair_split": args.pair_split,
               "overrides": args.overrides or [], "regimes": {}}

    for regime in regimes:
        results["regimes"][regime] = run_regime(
            model, dataset, loader, regime, out_dir, params,
            no_baseline=args.no_baseline, dense=not args.no_dense)

    logger.info("\n".join(report(results["regimes"], r) for r in regimes))

    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
