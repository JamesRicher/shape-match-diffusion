"""ISOLATED, DELETE-SAFE DIAGNOSTIC -- sparse-point error under INDEPENDENT FPS.

Everything this script needs lives under diagnostics/; every output it writes goes to
diagnostics/results/. It modifies no tracked file and registers nothing. To remove it and
all the data it produced, delete the diagnostics/ directory. Nothing else depends on it.

WHAT IT ANSWERS
---------------
The reported `sparse_acc` (stats.json) is measured under BIJECTIVE FPS, where sparse Y point j
is constructed to match sparse X point j (identity GT). That regime structurally CANNOT show a
global left<->right symmetry flip -- the flip maps j's neighbourhood to j's mirror, but the GT
it is scored against is still the identity, so a flipped-but-consistent map still looks perfect.
That is why every FAUST run reports ~0.96 sparse_acc while dense_error swings 0.016 -> 0.09.

This script re-scores the SAME sparse map under the honest INDEPENDENT-FPS regime (each shape
FPS'd on its own geometry, the regime dense MGE lives in), with the geodesic error evaluated AT
THE SPARSE FPS POINTS -- the direct analog of sparse_acc, but with real template GT instead of
the identity diagonal. It forks the diagnosis of where the dense flips come from:

  independent sparse error stays LOW (~ bijective)  -> the sparse map is clean under honest
      sampling; the flips are INTRODUCED BY THE DENSIFIER lift (fix lives in the densifier).
  independent sparse error is HIGH (flip_rate up)   -> the flip is ALREADY in the sparse
      matcher and bijective sparse_acc was hiding it (fix lives in the model / features / FPS).

Unlike sparse_flip_rate.py's `dense_voronoi` (which smears the sparse map over the whole
surface via nearest-anchor Voronoi and is dominated by region areas), this scores the n sparse
points directly, so it is the clean apples-to-apples counterpart of the reported sparse_acc.

GT AT A SPARSE POINT
--------------------
A sparse FPS point is generally not a template landmark, so its GT match is taken from the
NEAREST template point (geodesic, on its own shape): for Y sparse vertex iy, t* = argmin_t
d_Y(iy, corr_y[t]); the GT X vertex is corr_x[t*]. Error = d_X(pred_x_vertex, corr_x[t*]),
area-normalised (dist is already normalised, matching evaluate.py). Points exactly on a
landmark get exact GT; others get the honest nearest-landmark GT.

USAGE
-----
  python -m diagnostics.sparse_independent_error \
      -c configs/joint_gcn_diffusion/faust_matrix_diffusion_gcn_512_FMD.yaml \
         configs/joint_gcn_diffusion/faust_matrix_diffusion_gcn_512_redo.yaml

Run each config under both FPS regimes to also settle the euclidean/geodesic default drift:
  ... --fps-metric euclidean      # then again with --fps-metric geodesic

Optional: --checkpoint PATH (only with a single -c), --num-pairs N (0 = all), --flip-thresh F
(default 0.20), --acc-tol T (default 0.05), --fps-metric {config,geodesic,euclidean},
--device cuda/cpu.
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from datasets import build_dataset
from models import build_model
from models.base_model import to_numpy
from train import autofill_feat_dim
from utils.options import load_yaml, resolve_experiment_paths

_OUT_ROOT = os.path.join(os.path.dirname(__file__), 'results')


def _build(config_path, checkpoint, device, fps_metric):
    """Load a trained checkpoint + its test dataset exactly as evaluate.py does, forced into
    the honest independent-FPS regime. `fps_metric` != 'config' overrides the dataset's FPS
    metric (to compare the euclidean vs geodesic sampling regimes on one checkpoint)."""
    opt = load_yaml(config_path)
    if device is not None:
        opt['device'] = device
    opt['is_train'] = False
    resolve_experiment_paths(opt)
    ckpt = checkpoint or os.path.join(opt['path']['models'], 'final.pth')
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f'checkpoint not found: {ckpt}\nTrain first, or pass --checkpoint.')
    opt['path']['resume_state'] = ckpt
    opt['path']['resume'] = False

    dataset = build_dataset(opt['datasets']['test'])
    dataset.independent_fps = True                       # honest sampling (matches dense MGE)
    if fps_metric != 'config':
        dataset.fps_metric = fps_metric                  # override euclidean/geodesic sampling
    autofill_feat_dim(opt, int(dataset[0]['first']['feat'].shape[-1]))
    model = build_model(opt)
    model.eval()
    return model, dataset, opt, ckpt


def _sparse_independent_error(sp, idx_x, idx_y, dist_x, dist_y, corr_x, corr_y):
    """Per-sparse-point geodesic error of the model's sparse map under independent FPS.

    sp        (n,)  sparse map, Y sparse index -> X sparse index (model prediction)
    idx_x/y   (n,)  vertex indices of the FPS points on X / Y
    dist_x/y        area-normalised geodesic matrices (N_x,N_x) / (N_y,N_y)
    corr_x/y  (T,)  template point -> vertex on X / Y (GT .vts)

    Returns (n,) geodesic error, one per Y sparse point (see module docstring for the GT rule).
    """
    # GT X vertex for each Y sparse point via its nearest template landmark on Y
    d_to_landmarks = dist_y[np.ix_(idx_y, corr_y)]       # (n, T) geodesic from each Y point to landmarks
    t_star = d_to_landmarks.argmin(axis=1)               # (n,) nearest template point
    gt_x_vertex = corr_x[t_star]                         # (n,) its GT vertex on X
    pred_x_vertex = idx_x[sp]                            # (n,) where the model sent the Y point on X
    return dist_x[pred_x_vertex, gt_x_vertex]            # (n,) area-normalised geodesic error


def _summary(err, flip_thresh, acc_tol):
    """Scalar summary of a flat per-point error array (mirrors the reported sparse metrics)."""
    err = np.ravel(err)
    return {'avg_error': float(err.mean()),
            'acc_at_tol': float(np.mean(err <= acc_tol)),
            'flip_rate': float(np.mean(err > flip_thresh)),
            'gross_gt_0.1': float(np.mean(err > 0.1))}


def _bijective_reference(opt):
    """Pull the already-computed BIJECTIVE sparse stats from the run's stats.json, for the
    side-by-side reference column. Returns None if the run was never evaluated."""
    stats_path = os.path.join(opt['path']['results'], 'stats.json')
    if not os.path.isfile(stats_path):
        return None
    with open(stats_path) as f:
        s = json.load(f)
    return {k: s[k] for k in ('sparse_acc', 'sparse_avg_error') if k in s} or None


def run_one(config_path, checkpoint, device, fps_metric, num_pairs, seed, flip_thresh, acc_tol):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, fps_metric)
    name = opt['name']
    if not num_pairs:                                    # 0 -> all pairs, in order
        idxs = list(range(len(dataset)))
    else:                                                # seeded random subset (same across configs)
        n = min(num_pairs, len(dataset))
        idxs = sorted(np.random.default_rng(seed).choice(len(dataset), size=n, replace=False).tolist())

    errs = []
    regime = fps_metric if fps_metric != 'config' else getattr(dataset, 'fps_metric', 'config')
    for i in tqdm(idxs, desc=f'{name} (independent FPS / {regime})'):
        data = dataset[i]
        x, y = data['first'], data['second']
        sp = to_numpy(model.validate_single(data)).astype(np.int64)   # (n,) sparse Y->X, ONE sample

        errs.append(_sparse_independent_error(
            sp,
            to_numpy(x['sparse']['idx']).astype(np.int64),
            to_numpy(y['sparse']['idx']).astype(np.int64),
            to_numpy(x['dist']), to_numpy(y['dist']),
            to_numpy(x['corr']).astype(np.int64),
            to_numpy(y['corr']).astype(np.int64)))

    err = np.concatenate(errs)
    summary = {'name': name, 'checkpoint': ckpt, 'fps_metric': regime, 'n_pairs': len(idxs),
               'flip_thresh': flip_thresh, 'acc_tol': acc_tol,
               'independent_sparse': _summary(err, flip_thresh, acc_tol),
               'bijective_reference': _bijective_reference(opt)}

    out_dir = os.path.join(_OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    tag = '' if fps_metric == 'config' else f'_{regime}'
    np.savez(os.path.join(out_dir, f'sparse_independent{tag}.npz'), error=err)
    with open(os.path.join(out_dir, f'sparse_independent{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print_table(summaries, flip_thresh, acc_tol):
    print(f"\nbijective = reported sparse_acc / sparse_avg_error (identity GT, hides flips)")
    print(f"independent = same map re-scored at the sparse points under honest independent FPS")
    print(f"flip = fraction with geodesic error > {flip_thresh};  acc@{acc_tol} = fraction <= {acc_tol}\n")
    head = (f"{'experiment':30s}{'fps':>10s}"
            f"{'bij acc':>10s}{'bij err':>10s}"
            f"{'ind err':>10s}{'ind flip%':>11s}{'ind acc%':>10s}")
    print(head); print('-' * len(head))
    for s in summaries:
        ind = s['independent_sparse']
        ref = s['bijective_reference'] or {}
        bij_acc = f"{ref['sparse_acc']:.3f}" if 'sparse_acc' in ref else '--'
        bij_err = f"{ref['sparse_avg_error']:.4f}" if 'sparse_avg_error' in ref else '--'
        print(f"{s['name']:30s}{s['fps_metric']:>10s}"
              f"{bij_acc:>10s}{bij_err:>10s}"
              f"{ind['avg_error']:>10.4f}{ind['flip_rate']*100:>10.1f}%{ind['acc_at_tol']*100:>9.1f}%")
    print("\n(independent err ~ bijective err => sparse map is clean; flips enter at the densifier.")
    print(" independent err/flip% high while bijective acc stays ~0.96 => the flip is in the")
    print(" sparse matcher and bijective sparse_acc was masking it.)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', nargs='+', required=True, help='one or more training configs')
    p.add_argument('--checkpoint', default=None, help='checkpoint override (only with a single -c)')
    p.add_argument('--num-pairs', type=int, default=0, help='cap pairs for a quick look, seeded random subset (0 = all)')
    p.add_argument('--seed', type=int, default=0,
                   help='seed for the random pair subset; SAME across all -c configs so they stay comparable')
    p.add_argument('--fps-metric', choices=('config', 'geodesic', 'euclidean'), default='config',
                   help="override the dataset FPS metric (default: whatever the config says); "
                        "run both to settle the euclidean/geodesic default drift")
    p.add_argument('--flip-thresh', type=float, default=0.20, help='geodesic-error flip threshold')
    p.add_argument('--acc-tol', type=float, default=0.05, help='geodesic tolerance counted as a hit')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    args = p.parse_args()
    if args.checkpoint and len(args.config) > 1:
        p.error('--checkpoint can only be used with a single -c')

    summaries = []
    for cfg in args.config:
        summaries.append(run_one(cfg, args.checkpoint, args.device, args.fps_metric,
                                 args.num_pairs, args.seed, args.flip_thresh, args.acc_tol))
    _print_table(summaries, args.flip_thresh, args.acc_tol)
    print(f"\nper-experiment JSON + error arrays under: {_OUT_ROOT}/<name>/")


if __name__ == '__main__':
    main()
