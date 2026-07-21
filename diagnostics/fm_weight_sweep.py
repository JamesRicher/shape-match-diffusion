"""ISOLATED, DELETE-SAFE DIAGNOSTIC -- WKS-vs-landmark weight sweep in the FM densifier.

Everything this script needs lives under diagnostics/; every output it writes goes to
diagnostics/results/. It modifies no tracked file and registers nothing. To remove it and
all the data it produced, delete the diagnostics/ directory. Nothing else depends on it.

WHAT IT ANSWERS
---------------
The FunctionalMapDensifier's ONLY symmetry-breaking signal is the landmarks; the global WKS
block is intrinsically symmetric and cannot tell left from right. So the WKS<->landmark balance
(`lm_weight`) directly governs whether the dense map flips. This sweeps `lm_weight` over a
CACHED sparse map (sample once, re-run only the linear FM solve per weight) and reports the
dense flip rate / MGE at each weight -- for the model's landmarks and, as a ceiling, for exact
GT landmarks.

INTERPRETATION
  flip_rate has a clear minimum at some lm_weight  -> dense flips are a WEIGHTING problem;
      retune lm_weight, no retrain needed (cheap fix).
  flip_rate high at EVERY weight (model landmarks)  -> the landmarks themselves are flipped;
      the fix is upstream (features / diffusion / FPS), not the densifier.
  GT-landmark flip_rate ~0 across weights but model curve stays high -> densifier is capable;
      the model's landmarks are the problem (pairs with sparse_independent_error.py).
  GT-landmark flip_rate also high -> densifier / WKS-symmetry issue independent of landmarks.

Read alongside sparse_independent_error.py (is the sparse map clean?) and sparse_flip_rate.py
(where do the flips enter?). This one asks: given the sparse map, can the densifier weighting
recover a flip-free dense map?

USAGE
-----
  python -m diagnostics.fm_weight_sweep \
      -c configs/joint_gcn_diffusion/faust_matrix_diffusion_gcn_512_FMD.yaml \
         configs/joint_gcn_diffusion/faust_matrix_diffusion_gcn_512_redo.yaml

Optional: --checkpoint PATH (single -c only), --weights "0 0.5 1 2 5 10 20 50",
--num-pairs N (0 = all), --seed S, --flip-thresh F (default 0.20),
--fps-metric {config,geodesic,euclidean}, --no-gt (skip the GT-landmark ceiling), --device.
"""
import argparse
import dataclasses
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from datasets import build_dataset
from metrics.geo_metric import calculate_geodesic_error
from models import build_model
from models.base_model import to_numpy
from train import autofill_feat_dim
from utils.options import load_yaml, resolve_experiment_paths

_OUT_ROOT = os.path.join(os.path.dirname(__file__), 'results')
_DEFAULT_WEIGHTS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def _build(config_path, checkpoint, device, fps_metric):
    """Load a trained checkpoint + its test dataset as evaluate.py does, forced into the honest
    independent-FPS regime; `fps_metric` != 'config' overrides the dataset FPS metric."""
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
    dataset.independent_fps = True
    if fps_metric != 'config':
        dataset.fps_metric = fps_metric
    autofill_feat_dim(opt, int(dataset[0]['first']['feat'].shape[-1]))
    model = build_model(opt)
    model.eval()
    if getattr(model, 'densifier', None) is None or \
            type(model.densifier).__name__ != 'FunctionalMapDensifier':
        raise ValueError(f"{opt['name']}: this sweep requires a FunctionalMapDensifier "
                         f"(got {type(getattr(model, 'densifier', None)).__name__}).")
    return model, dataset, opt, ckpt


def _gt_ctx_identity(model, ctx, corr_x, corr_y, n):
    """Densifier inputs for EXACT GT landmarks: n evenly spaced template points give matched
    (X vertex, Y vertex) pairs, with an identity sparse map. Returns (identity, gt_ctx)."""
    T = corr_x.shape[0]
    t_sub = np.unique(np.linspace(0, T - 1, n).round().astype(np.int64))
    gt_idx_x = torch.as_tensor(corr_x[t_sub], dtype=torch.long, device=ctx.idx_x.device)
    gt_idx_y = torch.as_tensor(corr_y[t_sub], dtype=torch.long, device=ctx.idx_y.device)
    gt_ctx = dataclasses.replace(ctx, idx_x=gt_idx_x, idx_y=gt_idx_y)
    identity = torch.arange(len(t_sub), dtype=torch.long, device=ctx.idx_x.device)
    return identity, gt_ctx


def _flip_stats(err, thresh):
    err = np.ravel(err)
    return {'mean': float(err.mean()), 'flip_rate': float(np.mean(err > thresh))}


def run_one(config_path, checkpoint, device, fps_metric, weights, num_pairs, seed,
            flip_thresh, do_gt):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, fps_metric)
    name = opt['name']
    if not num_pairs:
        idxs = list(range(len(dataset)))
    else:
        n = min(num_pairs, len(dataset))
        idxs = sorted(np.random.default_rng(seed).choice(len(dataset), size=n, replace=False).tolist())

    # per weight: lists of per-correspondence error arrays (one per pair), for model & GT landmarks
    e_model = {w: [] for w in weights}
    e_gt = {w: [] for w in weights}
    orig_weight = model.densifier.lm_weight

    regime = fps_metric if fps_metric != 'config' else getattr(dataset, 'fps_metric', 'config')
    for i in tqdm(idxs, desc=f'{name} (independent FPS / {regime})'):
        data = dataset[i]
        x, y = data['first'], data['second']
        sp_t = model.validate_single(data)                       # (n,) sparse Y->X, ONE sample (cached)
        ctx = model._densify_context(data)
        dist_x = to_numpy(x['dist'])
        corr_x = to_numpy(x['corr']).astype(np.int64)
        corr_y = to_numpy(y['corr']).astype(np.int64)
        gt_inputs = _gt_ctx_identity(model, ctx, corr_x, corr_y, sp_t.shape[0]) if do_gt else None

        for w in weights:
            model.densifier.lm_weight = w                        # only rescales the landmark block
            fm = to_numpy(model.densifier.densify(sp_t, ctx))    # re-solve FM (cheap), same cached sparse map
            e_model[w].append(calculate_geodesic_error(dist_x, corr_x, corr_y, fm, return_mean=False))
            if do_gt:
                identity, gt_ctx = gt_inputs
                gt = to_numpy(model.densifier.densify(identity, gt_ctx))
                e_gt[w].append(calculate_geodesic_error(dist_x, corr_x, corr_y, gt, return_mean=False))
    model.densifier.lm_weight = orig_weight                      # restore (paranoia; instance is discarded anyway)

    summary = {'name': name, 'checkpoint': ckpt, 'fps_metric': regime, 'n_pairs': len(idxs),
               'flip_thresh': flip_thresh, 'default_lm_weight': orig_weight, 'weights': weights,
               'model_landmarks': {}, 'gt_landmarks': {} if do_gt else None}
    for w in weights:
        summary['model_landmarks'][str(w)] = _flip_stats(np.concatenate(e_model[w]), flip_thresh)
        if do_gt:
            summary['gt_landmarks'][str(w)] = _flip_stats(np.concatenate(e_gt[w]), flip_thresh)

    out_dir = os.path.join(_OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'fm_weight_sweep.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print_table(summary, thresh):
    name, weights = summary['name'], summary['weights']
    print(f"\n=== {name}  (fps={summary['fps_metric']}, n_pairs={summary['n_pairs']}, "
          f"default lm_weight={summary['default_lm_weight']}) ===")
    print(f"cells = flip% / MGE   (flip = geodesic error > {thresh})\n")
    head = f"{'landmarks':16s}" + "".join(f"{('w=' + str(w)):>15s}" for w in weights)
    print(head); print('-' * len(head))
    for label, key in (('model', 'model_landmarks'), ('GT (ceiling)', 'gt_landmarks')):
        block = summary[key]
        if block is None:
            continue
        row = f"{label:16s}"
        for w in weights:
            s = block[str(w)]
            row += f"{s['flip_rate'] * 100:6.1f}% /{s['mean']:5.3f}"
        print(row)
    best = min(weights, key=lambda w: summary['model_landmarks'][str(w)]['flip_rate'])
    print(f"\nmin model flip% at lm_weight={best} "
          f"({summary['model_landmarks'][str(best)]['flip_rate'] * 100:.1f}%)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', nargs='+', required=True, help='one or more training configs')
    p.add_argument('--checkpoint', default=None, help='checkpoint override (only with a single -c)')
    p.add_argument('--weights', default=None,
                   help='space-separated lm_weight values (default: "0 0.5 1 2 5 10 20 50")')
    p.add_argument('--num-pairs', type=int, default=0, help='cap pairs for a quick look, seeded subset (0 = all)')
    p.add_argument('--seed', type=int, default=0, help='seed for the pair subset; SAME across all -c configs')
    p.add_argument('--fps-metric', choices=('config', 'geodesic', 'euclidean'), default='config',
                   help='override the dataset FPS metric (default: config)')
    p.add_argument('--flip-thresh', type=float, default=0.20, help='geodesic-error flip threshold')
    p.add_argument('--no-gt', action='store_true', help='skip the exact-GT-landmark ceiling curve')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    args = p.parse_args()
    if args.checkpoint and len(args.config) > 1:
        p.error('--checkpoint can only be used with a single -c')
    weights = [float(w) for w in args.weights.split()] if args.weights else list(_DEFAULT_WEIGHTS)

    for cfg in args.config:
        summary = run_one(cfg, args.checkpoint, args.device, args.fps_metric, weights,
                          args.num_pairs, args.seed, args.flip_thresh, not args.no_gt)
        _print_table(summary, args.flip_thresh)
    print(f"\nper-experiment JSON under: {_OUT_ROOT}/<name>/")


if __name__ == '__main__':
    main()
