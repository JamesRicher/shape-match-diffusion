"""DIAGNOSTIC -- how accurate is the DENSIFIER ALONE, given a perfect sparse map?

Pushes the GROUND-TRUTH 512-point correspondence through the real densifier and scores the
resulting whole-shape map. Nothing is sampled and no feature matching happens on the scored
arm: the sparse stage is replaced by the exact answer, so every unit of error that comes out
is manufactured by the densification stage itself. That number is the hard floor of the whole
sparse-then-densify architecture -- no sparse matcher, however good, can beat it.

This is the BIJECTIVE-anchor counterpart of diagnostics/densify_floor.py, and the two answer
different questions. densify_floor works on independent (honest) FPS anchor sets, where the two
sets do not correspond, so its "oracle" map must SNAP each Y anchor's true X-vertex to the
nearest X anchor -- its floor therefore bundles densifier error together with that snapping
error. Here the dataset is left in its bijective regime, where Y anchor j is by construction the
exact GT image of X anchor j, so the sparse map is the identity and there is nothing to snap.

  * gt_fm       -- identity sparse map -> the real densifier -> dense MGE.  THE MEASUREMENT.
  * gt_voronoi  -- same perfect map, but lifted by pure nearest-anchor assignment instead of
                   the densifier. The model-free granularity bound: what 512 anchors buy you
                   with no smoothing at all, so gt_fm - gt_voronoi prices the densifier's
                   smoothing against naive quantization.
  * diffusion   -- the model's own sparse map on the SAME anchors (optional). The gap to gt_fm
                   is the pot owned by the sparse stage in this regime.

Also reported for the gt_fm arm:
  * anchor_drift -- geodesic error at the ANCHOR vertices themselves. The densifier was handed
                    their correct images; anything above ~0 is the FM smoothing walking away
                    from correspondences it was given, i.e. pure densifier infidelity.
  * patch_radius -- distance from each Y vertex to its nearest Y anchor. The geometric budget
                    the quantization error is drawn from; makes the floor interpretable.

Read-out: if gt_fm is a large fraction of the reported dense MGE, sparse-stage work (better
denoiser, best-of-K, flip repair) has little headroom left and the densification stage is the
bottleneck. If gt_fm is small, the sparse map owns the error.

  python -m diagnostics.densifier_gt_transport -c configs/joint_diffusionnet/faust_diffusionnet_512_FMD_snrfd_tw_gt.yaml
  python -m diagnostics.densifier_gt_transport -c configs/joint_diffusionnet/scape_diffusionnet_512_FMD_snrfd_tw_gt.yaml \
      --num-pairs 40 --no-diffusion
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from models.base_model import to_numpy
from diagnostics.nn_baseline_dense import (
    _build, _benchmark_dir, _dense_mge, _summary, calculate_geodesic_error,
)


def _gt_sparse_map(data):
    """The GT sparse map on bijective anchors: the identity.

    In the bijective regime the dataset builds Y anchor j as the GT image of X anchor j
    (gt_perm = I), so 'Y sparse point j matches X sparse point j' IS the ground truth and no
    snapping or template lookup is needed. Returns (n_y,) sparse X indices."""
    n = data['second']['sparse']['idx'].shape[0]
    return torch.arange(n, dtype=torch.long)


def _voronoi_error(data):
    """Pure-quantization arm: every Y vertex takes its nearest Y anchor's GT X-vertex, with no
    densifier smoothing. The granularity bound any nearest-anchor lift is subject to at this
    anchor count. Returns per-template-point geodesic errors."""
    x, y = data['first'], data['second']
    dist_x = to_numpy(x['dist'])
    dist_y = to_numpy(y['dist'])
    idx_x = to_numpy(x['sparse']['idx']).astype(np.int64)
    idx_y = to_numpy(y['sparse']['idx']).astype(np.int64)

    anchor_of = dist_y[:, idx_y].argmin(axis=1)          # (Vy,) nearest Y anchor per Y vertex
    dense_p2p = idx_x[anchor_of]                          # identity sparse map -> its X anchor
    return calculate_geodesic_error(dist_x, to_numpy(x['corr']), to_numpy(y['corr']),
                                    dense_p2p, return_mean=False)


def _gt_densify(model, data, sparse_p2p):
    """Densify the GT sparse map ONCE and read both numbers off the same dense map: the
    whole-shape geodesic error, and the error at the anchor vertices themselves. Densification is
    the expensive step here, so the two must not each trigger their own call.

    The anchor error is the fidelity check: the densifier was given these anchors' correct images,
    so anything above ~0 is its smoothing walking away from correspondences it was handed --
    distinct from the sub-patch quantization the other vertices suffer.
    Returns (per-template-point errors, per-anchor errors)."""
    x, y = data['first'], data['second']
    dense_p2p = to_numpy(model.densifier.densify(sparse_p2p, model._densify_context(data)))
    dist_x = to_numpy(x['dist'])
    idx_x = to_numpy(x['sparse']['idx']).astype(np.int64)
    idx_y = to_numpy(y['sparse']['idx']).astype(np.int64)
    mge = calculate_geodesic_error(dist_x, to_numpy(x['corr']), to_numpy(y['corr']),
                                   dense_p2p, return_mean=False)
    drift = dist_x[idx_x, dense_p2p[idx_y]]               # (n,) true X anchor vs predicted X vertex
    return mge, drift


def _patch_radius(data):
    """Distance from each Y vertex to its nearest Y anchor (the patch a dense vertex is
    quantized into). Returns the per-vertex array, area-normalised like the MGE."""
    idx_y = to_numpy(data['second']['sparse']['idx']).astype(np.int64)
    return to_numpy(data['second']['dist'])[:, idx_y].min(axis=1)


def _override_feat_source(model, feat_source):
    """Swap the densifier's dense data-term feature source after construction, mirroring what
    BaseDensifier.__init__ sets. 'wks' turns off wants_model_feats so the model stops densely
    running its extractor and the densifier falls back to its own network-free signature --
    the arm that asks whether the TRAINED features, rather than the FM machinery, are what
    breaks densification. No-op when 'config'."""
    if feat_source == 'config':
        return
    d = model.densifier
    d.feat_source = feat_source
    d.gcn_feats = feat_source == 'gcn'
    d.wants_model_feats = feat_source in ('gcn', 'diffnet')


@torch.no_grad()
def run(config_path, checkpoint, device, fps_metric, num_pairs, seed, with_diffusion,
        feat_source='config', overrides=None, tag=None):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, fps_metric,
                                       overrides=overrides)
    _override_feat_source(model, feat_source)
    # _build forces the honest independent-FPS regime (the dense-MGE reporting setup). This
    # diagnostic needs the opposite: bijective anchors, where Y anchor j IS the GT image of X
    # anchor j, so the identity is the exact sparse GT. Flipped back deliberately -- the numbers
    # here are a densifier-stage floor, not an honest end-to-end score.
    dataset.independent_fps = False
    name = opt['name']
    if not num_pairs:
        idxs = list(range(len(dataset)))
    else:
        n = min(num_pairs, len(dataset))
        idxs = sorted(np.random.default_rng(seed).choice(len(dataset), size=n, replace=False).tolist())

    errs = {'gt_fm': [], 'gt_voronoi': [], 'diffusion': [], 'anchor_drift': [], 'patch_radius': []}
    accs = []
    for i in tqdm(idxs, desc=f'{name} (GT transport through densifier)'):
        data = dataset[i]
        gt = _gt_sparse_map(data)
        mge, drift = _gt_densify(model, data, gt)          # one densify, both readouts
        errs['gt_fm'].append(mge)
        errs['anchor_drift'].append(drift)
        errs['gt_voronoi'].append(_voronoi_error(data))    # numpy only, no densify
        errs['patch_radius'].append(_patch_radius(data))
        if with_diffusion:
            p2p = model.validate_single(data)
            errs['diffusion'].append(_dense_mge(model, data, p2p))
            accs.append(float((p2p.cpu() == gt).float().mean()))       # sparse acc on these anchors

    err = {k: np.concatenate(v) for k, v in errs.items() if v}
    summary = {'name': name, 'checkpoint': ckpt, 'n_pairs': len(idxs),
               'n_sparse': int(getattr(dataset, 'n_sparse', 0)),
               'fps_metric': getattr(dataset, 'fps_metric', 'config'),
               'feat_source': getattr(model.densifier, 'feat_source', None),
               # the densifier knobs a sweep moves, so each result file is self-describing
               'densifier': {k: getattr(model.densifier, k, None)
                             for k in ('k_fm', 'n_e', 'lm_bands', 'lm_weight', 'variance', 'mu')}}
    for k in ('gt_fm', 'gt_voronoi', 'diffusion', 'anchor_drift', 'patch_radius'):
        if k in err:
            summary[k] = _summary(err[k])

    summary['densifier_floor_MGE'] = summary['gt_fm']['dense_MGE']
    # negative => the densifier's smoothing beats naive nearest-anchor quantization
    summary['fm_minus_voronoi'] = summary['gt_fm']['dense_MGE'] - summary['gt_voronoi']['dense_MGE']
    if 'diffusion' in summary:
        actual = summary['diffusion']['dense_MGE']
        summary['actual_MGE'] = actual
        summary['sparse_stage_MGE'] = actual - summary['densifier_floor_MGE']
        summary['frac_from_densifier'] = summary['densifier_floor_MGE'] / actual if actual > 0 else 0.0
        summary['diffusion_sparse_acc'] = float(np.mean(accs))

    out_dir = _benchmark_dir(ckpt, name)
    os.makedirs(out_dir, exist_ok=True)
    # An --fps-metric override is a different sampling regime, not a rerun, so it gets its own
    # file stem -- the config-default run keeps the plain name and is never clobbered by a sweep.
    stem = ('densifier_gt_transport'
            + (f'__fps_{fps_metric}' if fps_metric != 'config' else '')
            + (f'__feat_{feat_source}' if feat_source != 'config' else '')
            + (f'__{tag}' if tag else ''))
    np.savez(os.path.join(out_dir, f'{stem}.npz'),
             **{f'{k}_error': v for k, v in err.items()})
    summary['out_dir'] = out_dir
    with open(os.path.join(out_dir, f'{stem}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print(s):
    print(f"\nexperiment      : {s['name']}")
    print(f"checkpoint      : {s['checkpoint']}")
    print(f"pairs / anchors : {s['n_pairs']} / {s['n_sparse'] or '?'}   fps: {s['fps_metric']}   "
          f"densifier feat_source: {s['feat_source']}")
    d = s.get('densifier') or {}
    print(f"densifier knobs : " + '  '.join(f'{k}={v}' for k, v in d.items() if v is not None))
    print('  (BIJECTIVE anchors: the sparse GT is the identity -- a densifier-stage floor, '
          'not an honest end-to-end score)')
    print(f"\n{'arm':>20} {'dense MGE':>11} {'median':>9} {'p90':>9} {'gross>0.1':>11}")
    print('-' * 63)
    for key, label in (('gt_voronoi', 'GT + voronoi lift'), ('gt_fm', 'GT + densifier'),
                       ('diffusion', 'diffusion (actual)')):
        if key in s:
            a = s[key]
            print(f"{label:>20} {a['dense_MGE']:>11.4f} {a['median']:>9.4f} {a['p90']:>9.4f} "
                  f"{a['gross_gt_0.1']*100:>10.1f}%")
    print('-' * 63)
    a, p = s['anchor_drift'], s['patch_radius']
    print(f"anchor drift (err at the anchors the densifier was given)"
          f" = {a['dense_MGE']:.4f}  (p90 {a['p90']:.4f})")
    print(f"patch radius (Y vertex -> nearest anchor)"
          f"                = {p['dense_MGE']:.4f}  (p90 {p['p90']:.4f})")
    print(f"densifier floor (perfect sparse map)"
          f"                     = {s['densifier_floor_MGE']:.4f}")
    print(f"densifier vs naive voronoi"
          f"                              = {s['fm_minus_voronoi']:+.4f}"
          f"  ({'densifier helps' if s['fm_minus_voronoi'] < 0 else 'voronoi is no worse'})")
    if 'actual_MGE' in s:
        print(f"actual (model's own sparse map, same anchors)            = {s['actual_MGE']:.4f}"
              f"   [sparse acc {s['diffusion_sparse_acc']*100:.1f}%]")
        print(f"owned by the sparse stage"
              f"                               = {s['sparse_stage_MGE']:+.4f}"
              f"  ({(1 - s['frac_from_densifier'])*100:.1f}% of actual)")
        if s['frac_from_densifier'] > 0.6:
            print('-> the DENSIFIER is the bottleneck: a perfect sparse map barely moves the MGE.')
        elif s['frac_from_densifier'] < 0.3:
            print('-> the SPARSE STAGE owns the error: densification is nearly free of blame.')
        else:
            print('-> mixed: both stages contribute materially.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True, help='training config (arch + test set)')
    p.add_argument('--checkpoint', default=None, help='checkpoint override (e.g. cross-dataset)')
    p.add_argument('--num-pairs', type=int, default=0, help='cap pairs (seeded subset); 0 = all')
    p.add_argument('--seed', type=int, default=0, help='seed for the --num-pairs subset')
    p.add_argument('--no-diffusion', action='store_true',
                   help="skip the model's own sparse map arm (no sampling; much faster)")
    p.add_argument('--fps-metric', choices=('config', 'geodesic', 'euclidean'), default='config',
                   help='override the dataset FPS metric (default: whatever the config says)')
    p.add_argument('--feat-source', choices=('config', 'frozen', 'gcn', 'wks', 'diffnet'),
                   default='config',
                   help="override the densifier's dense data-term features (default: the config's). "
                        "'wks' is the network-free spectral signature -- use it to ask whether the "
                        'trained features or the FM machinery is responsible for the floor')
    p.add_argument('--set', action='append', metavar='KEY=VALUE', default=None, dest='overrides',
                   help='override a config value by dotted key, repeatable (e.g. densifier.k_fm=200); '
                        'pair with --tag so swept runs get their own output files')
    p.add_argument('--tag', default=None,
                   help='suffix for this run\'s output files, so a sweep does not clobber the '
                        'config-default run (e.g. --tag k200)')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    args = p.parse_args()

    s = run(args.config, args.checkpoint, args.device, args.fps_metric,
            args.num_pairs, args.seed, not args.no_diffusion, args.feat_source,
            args.overrides, args.tag)
    _print(s)
    print(f"\nper-pair errors + JSON under: {s['out_dir']}/")


if __name__ == '__main__':
    main()
