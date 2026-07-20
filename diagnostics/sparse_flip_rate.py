"""ISOLATED, DELETE-SAFE DIAGNOSTIC -- sparse-landmark symmetry-flip decomposition.

Everything this script needs lives under diagnostics/; every output it writes goes to
diagnostics/results/. It modifies no tracked file and registers nothing. To remove it and
all the data it produced, delete the diagnostics/ directory. Nothing else depends on it.

WHAT IT ANSWERS
---------------
The dense-MGE regressions are left/right symmetry flips. We know the FunctionalMapDensifier's
only symmetry-breaking signal is the sparse landmarks (the global WKS term is symmetric), so
the open question is WHERE the flips enter -- the sparse map (features/diffusion) or the
densifier. This forks the fix: upstream flips need a retrain; a densifier that amplifies (or
mishandles) landmarks can be fixed cheaply in functional_map.py.

For each experiment it runs the trained checkpoint under the honest INDEPENDENT-FPS regime
(the only regime dense MGE is scored in) and, from a SINGLE sample per pair, computes three
per-correspondence geodesic-error arrays with the exact same metric and template GT that
evaluate.py uses (calculate_geodesic_error over all template points):

  dense_fm      -- the shipped densifier's dense map. Reproduces stats.json dense_error
                   (sanity check).
  dense_voronoi -- the SPARSE map propagated to the surface by nearest Y anchor (Voronoi),
                   scored with exact template GT. This is the sparse map's own flip rate
                   projected onto the surface -- needs no approximate per-anchor GT.
  dense_gtlm    -- the densifier fed EXACT ground-truth landmarks (template-point pairs).
                   Densifier best case: isolates densifier capability from sparse-map errors.

INTERPRETATION  (a correspondence is "flipped" if its error > --flip-thresh, default 0.20,
the valley between the correct and far-side modes in the error histogram)

  voronoi_flip ~= fm_flip        -> densifier faithfully propagates the sparse map;
                                    flips originate upstream (features/diffusion) => RETRAIN.
  fm_flip >> voronoi_flip        -> densifier AMPLIFIES a few bad landmarks into regions
                                    => cheap densifier-side fix (landmark-consistency filter).
  fm_flip << voronoi_flip        -> densifier actively repairs sparse-map flips (good to know).
  gtlm_flip ~= 0                 -> densifier is capable given clean landmarks; the model's
                                    landmarks are the problem.
  gtlm_flip high                 -> densifier flips even with perfect landmarks => densifier bug.

USAGE
-----
  python -m diagnostics.sparse_flip_rate \
      -c configs/joint_gcn_diffusion/faust_matrix_diffusion_gcn_512_FMD.yaml \
         configs/joint_gcn_diffusion/faust_matrix_diffusion_gcn_512_redo.yaml \
         configs/joint_gcn_diffusion/runA_patch64_geo.yaml \
         configs/joint_gcn_diffusion/runB_patch48_euc.yaml

Optional: --checkpoint PATH (only with a single -c), --num-pairs N (cap for a quick look,
0 = all), --flip-thresh F, --device cuda/cpu.
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


def _build(config_path, checkpoint, device):
    """Load a trained checkpoint + its test dataset exactly as evaluate.py does, but force the
    honest independent-FPS sampling regime the dense metric lives in."""
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
    autofill_feat_dim(opt, int(dataset[0]['first']['feat'].shape[-1]))
    model = build_model(opt)
    model.eval()
    return model, dataset, opt, ckpt


def _voronoi_p2p(sparse_p2p, idx_x, idx_y, dist_y):
    """Propagate the sparse map to every Y vertex via its nearest Y anchor (geodesic Voronoi).
    Returns a dense (N_y,) Y-vertex -> X-vertex map built from the sparse map alone."""
    nearest_anchor = dist_y[:, idx_y].argmin(axis=1)     # (N_y,) -> sparse anchor index
    return idx_x[sparse_p2p[nearest_anchor]]             # (N_y,) -> X vertex


def _gt_landmark_p2p(model, ctx, corr_x, corr_y, n):
    """Run the configured densifier on EXACT ground-truth landmarks: n evenly spaced template
    points give matched (X vertex, Y vertex) pairs. Returns the dense (N_y,) map."""
    T = corr_x.shape[0]
    t_sub = np.unique(np.linspace(0, T - 1, n).round().astype(np.int64))
    gt_idx_x = torch.as_tensor(corr_x[t_sub], dtype=torch.long)
    gt_idx_y = torch.as_tensor(corr_y[t_sub], dtype=torch.long)
    gt_ctx = dataclasses.replace(ctx, idx_x=gt_idx_x.to(ctx.idx_x.device),
                                 idx_y=gt_idx_y.to(ctx.idx_y.device))
    identity = torch.arange(len(t_sub), dtype=torch.long, device=ctx.idx_x.device)
    return to_numpy(model.densifier.densify(identity, gt_ctx))


def _flip_stats(err, thresh):
    """Scalar summary of a flat per-correspondence error array."""
    err = np.ravel(err)
    return {'mean': float(err.mean()),
            'flip_rate': float(np.mean(err > thresh)),
            'gross_gt_0.1': float(np.mean(err > 0.1))}


def run_one(config_path, checkpoint, device, num_pairs, flip_thresh):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device)
    name = opt['name']
    has_dens = getattr(model, 'densifier', None) is not None
    n_pairs = len(dataset) if not num_pairs else min(num_pairs, len(dataset))

    e_fm, e_vor, e_gt = [], [], []
    for i in tqdm(range(n_pairs), desc=f'{name} (independent FPS)'):
        data = dataset[i]
        x, y = data['first'], data['second']
        sp_t = model.validate_single(data)                       # (n,) sparse Y->X, ONE sample
        sp = to_numpy(sp_t).astype(np.int64)

        idx_x = to_numpy(x['sparse']['idx']).astype(np.int64)
        idx_y = to_numpy(y['sparse']['idx']).astype(np.int64)
        dist_x = to_numpy(x['dist'])
        dist_y = to_numpy(y['dist'])
        corr_x = to_numpy(x['corr']).astype(np.int64)
        corr_y = to_numpy(y['corr']).astype(np.int64)

        # (b) sparse map, Voronoi-propagated -- exact template GT, no per-anchor GT needed
        vor = _voronoi_p2p(sp, idx_x, idx_y, dist_y)
        e_vor.append(calculate_geodesic_error(dist_x, corr_x, corr_y, vor, return_mean=False))

        if has_dens:
            ctx = model._densify_context(data)
            # (a) shipped densifier on the model's landmarks -- reproduces stats dense_error
            fm = to_numpy(model.densifier.densify(sp_t, ctx))
            e_fm.append(calculate_geodesic_error(dist_x, corr_x, corr_y, fm, return_mean=False))
            # (c) densifier on exact GT landmarks -- best case
            gt = _gt_landmark_p2p(model, ctx, corr_x, corr_y, len(idx_x))
            e_gt.append(calculate_geodesic_error(dist_x, corr_x, corr_y, gt, return_mean=False))

    out_dir = os.path.join(_OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    arrays, summary = {}, {'name': name, 'checkpoint': ckpt, 'n_pairs': n_pairs,
                           'flip_thresh': flip_thresh}
    arrays['voronoi'] = np.concatenate(e_vor)
    summary['dense_voronoi'] = _flip_stats(arrays['voronoi'], flip_thresh)
    if has_dens:
        arrays['fm'] = np.concatenate(e_fm)
        arrays['gtlm'] = np.concatenate(e_gt)
        summary['dense_fm'] = _flip_stats(arrays['fm'], flip_thresh)
        summary['dense_gtlm'] = _flip_stats(arrays['gtlm'], flip_thresh)

    np.savez(os.path.join(out_dir, 'errors.npz'), **arrays)
    with open(os.path.join(out_dir, 'flip_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print_table(summaries, thresh):
    cols = [('dense_fm', 'FM'), ('dense_voronoi', 'Voronoi(sparse)'), ('dense_gtlm', 'GT-landmarks')]
    print(f"\nflip rate = fraction with geodesic error > {thresh}   (MGE = mean error)\n")
    head = f"{'experiment':28s}" + "".join(f"{lbl:>18s}" for _, lbl in cols)
    print(head); print('-' * len(head))
    for s in summaries:
        row = f"{s['name']:28s}"
        for key, _ in cols:
            if key in s:
                row += f"{s[key]['flip_rate']*100:7.1f}% /{s[key]['mean']:6.3f}"
            else:
                row += f"{'--':>18s}"
        print(row)
    print("\n(cells show  flip% / MGE.  FM vs Voronoi: ~equal => flips are upstream (retrain);")
    print(" FM >> Voronoi => densifier amplifies (cheap fix).  GT-landmarks ~0 => densifier capable.)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', nargs='+', required=True, help='one or more training configs')
    p.add_argument('--checkpoint', default=None, help='checkpoint override (only with a single -c)')
    p.add_argument('--num-pairs', type=int, default=0, help='cap pairs for a quick look (0 = all)')
    p.add_argument('--flip-thresh', type=float, default=0.20, help='geodesic-error flip threshold')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    args = p.parse_args()
    if args.checkpoint and len(args.config) > 1:
        p.error('--checkpoint can only be used with a single -c')

    summaries = []
    for cfg in args.config:
        summaries.append(run_one(cfg, args.checkpoint, args.device, args.num_pairs, args.flip_thresh))
    _print_table(summaries, args.flip_thresh)
    print(f"\nper-experiment JSON + error arrays under: {_OUT_ROOT}/<name>/")


if __name__ == '__main__':
    main()
