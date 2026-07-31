"""DIAGNOSTIC -- how much of the dense MGE is the 512-anchor QUANTIZATION FLOOR vs
ANCHOR ERRORS (symmetry flips)?

Sizes the prize before designing a densification-stage refinement. The dense error splits
into two pots that want opposite treatments:

  (a) quantization / sub-patch  -- with the RIGHT anchors, a dense vertex still only lands
      near its target (patch granularity + densifier smoothing). A within-patch refiner
      (coarse-to-fine super-resolution) attacks this.
  (b) anchor errors / flips     -- an anchor maps to the wrong limb, dragging its whole
      patch with it. Only a stage ALLOWED to disagree with the sparse map fixes this
      (soft flip-repair). Prior diagnostics say this dominates the FAUST MGE tail.

To separate them we build, on the SAME honest independent-FPS anchor sets the dense MGE
lives on, the ORACLE sparse map: for each Y anchor, its true X-vertex (via the template)
snapped to the nearest X anchor -- the best assignment those two anchor sets can express.
Pushed through the real densifier this is the FLOOR: the error that survives PERFECT anchors.

  * oracle_fm      -- oracle sparse map -> real FM densifier -> dense MGE  == floor (a)
  * diffusion      -- the model's real map -> same densifier              == actual
  * nn (optional)  -- feature-NN map -> same densifier                    == reference
  * voronoi_floor  -- model-free: oracle map + pure nearest-anchor lift; the granularity
                      ceiling ANY nearest-anchor densifier is bounded by (no FM smoothing).

Read-out:  actual - floor  is the pot a flip-repair stage could recover;  floor  is the
pot a within-patch refiner is bounded by.  If floor << actual, build the soft flip-repair
design with a DELIBERATELY WEAK coarse prior; if floor is large, a tighter within-patch
candidate restriction is worth it.

  python -m diagnostics.densify_floor -c configs/joint_diffusionnet/faust_diffusionnet_512_FMD.yaml
  python -m diagnostics.densify_floor -c configs/joint_diffusionnet/scape_diffusionnet_512_FMD.yaml \
      --num-pairs 40 --no-nn
"""
import argparse
import json
import os

import numpy as np
import torch

from models.base_model import to_numpy
from diagnostics.nn_baseline_dense import (
    _build, _benchmark_dir, _dense_mge, _feature_similarity, _nn_sparse_map,
    _summary, calculate_geodesic_error,
)


def _oracle_sparse_map(data):
    """Best sparse Y->X assignment expressible on the two independent anchor sets.

    For each Y anchor (a real Y vertex) we find its true X-vertex via the shared template --
    the template point whose Y-vertex is geodesically nearest the anchor, mapped through to
    X -- then snap that to the nearest X anchor. Returns (n_y,) sparse X indices; collisions
    (several Y anchors -> one X anchor) are allowed, exactly as a real sparse map permits."""
    x, y = data['first'], data['second']
    dist_x = to_numpy(x['dist']); dist_y = to_numpy(y['dist'])
    corr_x = to_numpy(x['corr']).astype(np.int64)
    corr_y = to_numpy(y['corr']).astype(np.int64)
    idx_x = to_numpy(x['sparse']['idx']).astype(np.int64)
    idx_y = to_numpy(y['sparse']['idx']).astype(np.int64)

    t_star = dist_y[np.ix_(idx_y, corr_y)].argmin(axis=1)   # (n_y,) template pt nearest each Y anchor
    x_star = corr_x[t_star]                                  # (n_y,) true X-vertex for each Y anchor
    oracle = dist_x[np.ix_(x_star, idx_x)].argmin(axis=1)   # (n_y,) nearest X anchor to the true X-vertex
    return torch.from_numpy(oracle).long()


def _voronoi_floor_error(data, oracle):
    """Model-free pure-quantization ceiling: every Y vertex -> nearest Y anchor -> that
    anchor's oracle X-vertex. No FM smoothing, so this is the hard granularity floor any
    nearest-anchor densifier is bounded by. Returns per-template-point geodesic errors."""
    x, y = data['first'], data['second']
    dist_x = to_numpy(x['dist']); dist_y = to_numpy(y['dist'])
    corr_x = to_numpy(x['corr']).astype(np.int64)
    corr_y = to_numpy(y['corr']).astype(np.int64)
    idx_x = to_numpy(x['sparse']['idx']).astype(np.int64)
    idx_y = to_numpy(y['sparse']['idx']).astype(np.int64)

    anchor_of = dist_y[:, idx_y].argmin(axis=1)             # (Vy,) nearest Y anchor per Y vertex
    dense_p2p = idx_x[to_numpy(oracle).astype(np.int64)[anchor_of]]   # (Vy,) predicted X vertex
    return calculate_geodesic_error(dist_x, corr_x, corr_y, dense_p2p, return_mean=False)


@torch.no_grad()
def run(config_path, checkpoint, device, fps_metric, num_pairs, seed, with_nn):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, fps_metric)
    name = opt['name']
    if not num_pairs:
        idxs = list(range(len(dataset)))
    else:
        n = min(num_pairs, len(dataset))
        idxs = sorted(np.random.default_rng(seed).choice(len(dataset), size=n, replace=False).tolist())

    from tqdm import tqdm
    errs = {'oracle_fm': [], 'voronoi_floor': [], 'diffusion': [], 'nn': []}
    for i in tqdm(idxs, desc=f'{name} (densify floor)'):
        data = dataset[i]
        oracle = _oracle_sparse_map(data)
        errs['oracle_fm'].append(_dense_mge(model, data, oracle))
        errs['voronoi_floor'].append(_voronoi_floor_error(data, oracle))
        errs['diffusion'].append(_dense_mge(model, data, model.validate_single(data)))
        if with_nn:
            errs['nn'].append(_dense_mge(model, data, _nn_sparse_map(_feature_similarity(model, data))))

    err = {k: np.concatenate(v) for k, v in errs.items() if v}
    summary = {'name': name, 'checkpoint': ckpt, 'n_pairs': len(idxs),
               'fps_metric': getattr(dataset, 'fps_metric', 'config'),
               'feat_source': getattr(model.densifier, 'feat_source', None)}
    for k in ('oracle_fm', 'voronoi_floor', 'diffusion', 'nn'):
        if k in err:
            summary[k] = _summary(err[k])

    floor = summary['oracle_fm']['dense_MGE']
    actual = summary['diffusion']['dense_MGE']
    summary['floor_MGE'] = floor
    summary['actual_MGE'] = actual
    summary['anchor_error_MGE'] = actual - floor                     # pot a flip-repair stage can recover
    summary['frac_from_anchors'] = (actual - floor) / actual if actual > 0 else 0.0

    out_dir = _benchmark_dir(ckpt, name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, 'densify_floor.npz'),
             **{f'{k}_error': v for k, v in err.items()})
    summary['out_dir'] = out_dir
    with open(os.path.join(out_dir, 'densify_floor.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print(s):
    print(f"\nexperiment      : {s['name']}")
    print(f"checkpoint      : {s['checkpoint']}")
    print(f"pairs / fps     : {s['n_pairs']} / {s['fps_metric']}   densifier feat_source: {s['feat_source']}")
    print(f"\n{'arm':>17} {'dense MGE':>11} {'median':>9} {'p90':>9} {'gross>0.1':>11}")
    print('-' * 60)
    for key, label in (('voronoi_floor', 'voronoi floor'), ('oracle_fm', 'oracle+FM (floor)'),
                       ('nn', 'feature-NN'), ('diffusion', 'diffusion (actual)')):
        if key in s:
            a = s[key]
            print(f"{label:>17} {a['dense_MGE']:>11.4f} {a['median']:>9.4f} {a['p90']:>9.4f} "
                  f"{a['gross_gt_0.1']*100:>10.1f}%")
    print('-' * 60)
    print(f"floor (perfect anchors, real densifier) = {s['floor_MGE']:.4f}")
    print(f"actual (model)                          = {s['actual_MGE']:.4f}")
    print(f"recoverable by better anchors (flips)   = {s['anchor_error_MGE']:+.4f}"
          f"  ({s['frac_from_anchors']*100:.1f}% of actual)")
    if s['frac_from_anchors'] > 0.6:
        print("-> ANCHOR errors dominate: build the soft flip-repair design (weak coarse prior).")
    elif s['frac_from_anchors'] < 0.3:
        print("-> QUANTIZATION dominates: a tighter within-patch candidate restriction pays off.")
    else:
        print("-> mixed: both a within-patch refiner and flip-repair contribute.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True, help='training config (arch + test set)')
    p.add_argument('--checkpoint', default=None, help='checkpoint override (e.g. cross-dataset)')
    p.add_argument('--num-pairs', type=int, default=0, help='cap pairs (seeded subset); 0 = all')
    p.add_argument('--seed', type=int, default=0, help='seed for the --num-pairs subset')
    p.add_argument('--no-nn', action='store_true', help='skip the feature-NN reference arm')
    p.add_argument('--fps-metric', choices=('config', 'geodesic', 'euclidean'), default='config',
                   help='override the dataset FPS metric (default: whatever the config says)')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    args = p.parse_args()

    s = run(args.config, args.checkpoint, args.device, args.fps_metric,
            args.num_pairs, args.seed, not args.no_nn)
    _print(s)
    print(f"\nper-pair errors + JSON under: {s['out_dir']}/")


if __name__ == '__main__':
    main()
