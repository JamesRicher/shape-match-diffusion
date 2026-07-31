"""ISOLATED, DELETE-SAFE DIAGNOSTIC -- best-of-K with the FROZEN selector + multimodality readout.

One self-contained pipeline that, for each test pair, draws K diffusion samples, densifies each,
and reports three things:

  1. IMPROVED BEST-OF-K.  The deployable number: dense MGE of the frozen unsupervised selector
     (combined = w_iso*z(isometry) + (1-w_iso)*z(coverage), cycle dropped), reported next to
     feature-NN, the single sample (K=1), and the GT oracle, with oracle-recovery %. w_iso defaults
     to 0.700 -- the weight already chosen on the FAUST->SCAPE train split (analyze_selector.py,
     ~78% recovery). No ground truth is used to pick; MGE/oracle are for scoring only.

  2. MULTIMODALITY.  Per pair, the K densified maps are clustered by pairwise map disagreement
     (area-weighted mean geodesic gap on X, normalised by X diameter; complete-linkage cut at
     --mode-thresh). Reports n_modes and dominant-mode occupancy per pair, and pooled histograms of
     n_modes, occupancy, per-vertex dispersion, and the single->oracle MGE spread. This is the
     honest test of "is the correct answer actually a distinct mode among the K?" -- if pairs are
     unimodal (n_modes==1) the selector has nothing to select and the ceiling is low.

  3. PER-VERTEX ENTROPY / DISPERSION for polyscope.  For the most-multimodal pairs it exports the
     source (Y) mesh with two scalar fields over its vertices:
       * dispersion -- mean pairwise geodesic distance on X between the K targets of that vertex,
         normalised by X diameter (geometry-aware uncertainty; where do the samples disagree?).
       * entropy    -- Shannon entropy of the K discrete targets, normalised by log K.
     plus per-vertex geodesic error of the single / selector / oracle maps, so you can SEE where
     selection rescues the shape. View with --view <pair_file.npz> (needs polyscope).

Run in the `shapematch` env.

USAGE (the decisive cross-dataset run, FAUST model on SCAPE data):
  python -m diagnostics.best_of_k_analysis \
      -c configs/joint_diffusionnet/scape_diffusionnet_512_FMD.yaml \
      --checkpoint experiments/diffusionnet/faust_diffusionnet_512_FMD/models/final.pth \
      -K 32 --num-pairs 60 --eta 0.5 --split test --exclude-self --export-pairs 6

Then inspect a multimodal pair in polyscope:
  python -m diagnostics.best_of_k_analysis --view diagnostics/results/new/<name>/<tag>/polyscope/pair_XX.npz

Optional: --w-iso W (frozen selector weight), --mode-thresh F (cluster cut, frac of X diameter),
--n-iso-pairs, --seed, --device.
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from tqdm import tqdm

from metrics import build_metric
from models.base_model import to_numpy
from diagnostics.nn_baseline_dense import _build, _dense_mge, _feature_nn_sparse_map
from diagnostics.best_of_k_oracle import _hungarian
from diagnostics.selector import (
    _coverage_defect, _isometry_defect, _long_range_pairs, _sample_P0, _zscore,
)

_OUT_ROOT = os.path.join(os.path.dirname(__file__), 'results', 'new')
calculate_geodesic_error = build_metric({"type": "calculate_geodesic_error"})


# --------------------------------------------------------------------------- #
# per-pair sampling + densification
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _dense_maps(model, data, K, sample_eta):
    """K densified Y->X vertex maps (K, n_y) from K diffusion samples, plus the soft P0 stack."""
    P0 = _sample_P0(model, data, K, sample_eta)                       # (K, n_y_sparse, n_x_sparse)
    ctx = model._densify_context(data)
    maps = np.stack([to_numpy(model.densifier.densify(_hungarian(P0[k]), ctx)) for k in range(K)])
    return P0, maps


def _per_vertex_err(dist_x, corr_x, corr_y, p2p):
    """Per-point geodesic error of a dense map (same metric as validation, no mean). Length is the
    number of GT correspondence points (corr_y), not the Y vertex count -- use .mean() for MGE."""
    return calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=False)


def _err_field(dist_x, corr_x, corr_y, p2p, n_y):
    """Same error, but SCATTERED onto the full (n_y,) Y-vertex array via corr_y so it can be shown
    as a polyscope scalar quantity. Unevaluated vertices (not GT points) are left 0."""
    err = _per_vertex_err(dist_x, corr_x, corr_y, p2p)
    field = np.zeros(n_y)
    field[corr_y] = err
    return field


# --------------------------------------------------------------------------- #
# multimodality: cluster the K maps, per-vertex geometric dispersion
# --------------------------------------------------------------------------- #
def _pairwise_and_dispersion(maps, dist_x, mass_y):
    """From K dense maps (K, n_y) build:
      * D    (K, K)  -- area-weighted mean geodesic map disagreement on X, normalised by X diameter.
      * disp (n_y,)  -- per-vertex mean pairwise geodesic spread of the K targets, same normaliser.
    Both share the single O(K^2) loop over sample pairs (each inner op is vectorised over vertices).
    """
    K = maps.shape[0]
    diam = float(dist_x.max()) or 1.0
    wsum = float(mass_y.sum())
    D = np.zeros((K, K))
    disp = np.zeros(maps.shape[1])
    n_pairs = 0
    for a in range(K):
        ta = maps[a]
        for b in range(a + 1, K):
            d = dist_x[ta, maps[b]]                                   # (n_y,) geodesic gap on X
            D[a, b] = D[b, a] = float((mass_y * d).sum() / wsum / diam)
            disp += d
            n_pairs += 1
    if n_pairs:
        disp /= (n_pairs * diam)
    return D, disp


def _cluster_modes(D, thresh):
    """Complete-linkage cluster of the K maps by map-disagreement D, cut at `thresh`.
    Returns (n_modes, dominant_occupancy_frac, labels)."""
    K = D.shape[0]
    if K < 2:
        return 1, 1.0, np.ones(K, dtype=int)
    labels = fcluster(linkage(squareform(D, checks=False), method='complete'),
                      t=thresh, criterion='distance')
    counts = np.bincount(labels)[1:]
    return int(labels.max()), float(counts.max() / K), labels


def _vertex_entropy(maps):
    """Normalised Shannon entropy in [0,1] of the K discrete targets, per Y vertex (0 = all agree)."""
    K, n_y = maps.shape
    if K < 2:
        return np.zeros(n_y)
    ent = np.empty(n_y)
    logK = np.log(K)
    for i in range(n_y):
        _, c = np.unique(maps[:, i], return_counts=True)
        p = c / K
        ent[i] = float(-(p * np.log(p)).sum() / logK)
    return ent


# --------------------------------------------------------------------------- #
# main run
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run(cfg, checkpoint, device, K, num_pairs, seed, eta, w_iso, n_iso_pairs,
        mode_thresh, split, exclude_self, export_pairs):
    model, dataset, opt, ckpt = _build(cfg, checkpoint, device, 'config',
                                       split=split, exclude_self=exclude_self)
    name = opt['name']
    idxs = (list(range(len(dataset))) if not num_pairs else
            sorted(np.random.default_rng(seed).choice(
                len(dataset), size=min(num_pairs, len(dataset)), replace=False).tolist()))

    ckpt_stem = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
    eta_tag = '' if eta == 0.0 else f'_eta{eta:g}'
    tag = f'analysis_{ckpt_stem}_{split}_K{K}{eta_tag}_w{w_iso:g}'
    out_dir = os.path.join(_OUT_ROOT, name, tag)
    ps_dir = os.path.join(out_dir, 'polyscope')
    os.makedirs(ps_dir, exist_ok=True)

    # per-pair accumulators
    rows = {k: [] for k in ['feature_nn', 'single', 'selector', 'oracle', 'worst']}
    n_modes, occ, mean_disp = [], [], []
    export_cache = []                                                 # per-pair heavy payload, pruned later

    for pi in tqdm(idxs, desc=f'{name} best-of-{K} analysis eta={eta}'):
        data = dataset[pi]
        x, y = data['first'], data['second']
        dist_x, dist_y = to_numpy(x['dist']), to_numpy(y['dist'])
        mass_x, mass_y = to_numpy(x['mass']), to_numpy(y['mass'])
        corr_x, corr_y = to_numpy(x['corr']), to_numpy(y['corr'])
        ii, jj = _long_range_pairs(dist_y, n_iso_pairs, seed + pi)

        _, maps = _dense_maps(model, data, K, eta)                   # (K, n_y)
        mge = np.empty(K); iso = np.empty(K); cov = np.empty(K)
        for k in range(K):
            mge[k] = _per_vertex_err(dist_x, corr_x, corr_y, maps[k]).mean()
            iso[k] = _isometry_defect(maps[k], dist_x, dist_y, mass_y, ii, jj)
            cov[k] = _coverage_defect(maps[k], mass_x, mass_y)

        combined = w_iso * _zscore(iso) + (1.0 - w_iso) * _zscore(cov)   # FROZEN selector, no GT
        pick = int(combined.argmin())
        best = int(mge.argmin())

        rows['feature_nn'].append(_dense_mge(model, data, _feature_nn_sparse_map(model, data)).mean())
        rows['single'].append(float(mge[0]))
        rows['selector'].append(float(mge[pick]))
        rows['oracle'].append(float(mge[best]))
        rows['worst'].append(float(mge.max()))

        D, disp = _pairwise_and_dispersion(maps, dist_x, mass_y)
        nm, oc, _ = _cluster_modes(D, mode_thresh)
        n_modes.append(nm); occ.append(oc); mean_disp.append(float(disp.mean()))

        # keep the heaviest per-vertex payload only for the pairs we will export
        export_cache.append((pi, nm, float(disp.mean()), dict(
            y_verts=to_numpy(y['verts']), y_faces=to_numpy(y['faces']),
            x_verts=to_numpy(x['verts']), x_faces=to_numpy(x['faces']),
            dispersion=disp, maps=maps, mge=mge, pick=pick, best=best,
            single_map=maps[0], selector_map=maps[pick], oracle_map=maps[best],
            err_single=_err_field(dist_x, corr_x, corr_y, maps[0], maps.shape[1]),
            err_selector=_err_field(dist_x, corr_x, corr_y, maps[pick], maps.shape[1]),
            err_oracle=_err_field(dist_x, corr_x, corr_y, maps[best], maps.shape[1]))))

    rows = {k: np.asarray(v) for k, v in rows.items()}
    n_modes = np.asarray(n_modes); occ = np.asarray(occ); mean_disp = np.asarray(mean_disp)
    single_m, oracle_m = rows['single'].mean(), rows['oracle'].mean()

    def block(v):
        return {'dense_MGE': float(v.mean()), 'median': float(np.median(v)),
                'gross_gt_0.1': float(np.mean(v > 0.1)),
                'oracle_recovery': (float((single_m - v.mean()) / (single_m - oracle_m))
                                    if single_m > oracle_m else None)}

    summary = {
        'name': name, 'checkpoint': ckpt, 'K': K, 'eta': eta, 'w_iso': w_iso,
        'split': split, 'n_pairs': len(idxs), 'mode_thresh': mode_thresh,
        'feature_nn': block(rows['feature_nn']), 'single': block(rows['single']),
        'selector': block(rows['selector']), 'oracle': block(rows['oracle']),
        'worst': block(rows['worst']),
        'multimodality': {
            'mean_n_modes': float(n_modes.mean()), 'frac_multimodal': float(np.mean(n_modes > 1)),
            'median_dominant_occupancy': float(np.median(occ)),
            'mean_vertex_dispersion': float(mean_disp.mean())},
    }

    # ---- export the most-multimodal pairs for polyscope --------------------- #
    exported = []
    if export_pairs:
        order = sorted(export_cache, key=lambda t: (t[1], t[2]), reverse=True)[:export_pairs]
        for pi, nm, md, payload in order:
            f = os.path.join(ps_dir, f'pair_{pi:03d}.npz')
            np.savez(f, pair_idx=pi, n_modes=nm,
                     entropy=_vertex_entropy(payload['maps']), **payload)
            exported.append({'pair_idx': pi, 'n_modes': int(nm), 'file': f})
    summary['exported_pairs'] = exported

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, 'per_pair.npz'), idxs=idxs, n_modes=n_modes, occupancy=occ,
             mean_dispersion=mean_disp, **rows)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    _histograms(os.path.join(out_dir, 'histograms.png'), rows, n_modes, occ, mean_disp, name, K)
    summary['out_dir'] = out_dir
    return summary


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _histograms(path, rows, n_modes, occ, mean_disp, name, K):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    spread = rows['worst'] - rows['oracle']
    sel_gap = rows['selector'] - rows['oracle']
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f'{name}   best-of-{K} multimodality', fontsize=13)

    ax[0, 0].hist(n_modes, bins=np.arange(0.5, max(int(n_modes.max()), 2) + 1.5), color='#4C72B0',
                  edgecolor='white')
    ax[0, 0].set(title='modes per pair (map clusters)', xlabel='n_modes', ylabel='pairs')
    ax[0, 0].axvline(1.5, color='k', ls=':', lw=1)

    ax[0, 1].hist(occ, bins=20, range=(0, 1), color='#55A868', edgecolor='white')
    ax[0, 1].set(title='dominant-mode occupancy', xlabel='largest cluster / K', ylabel='pairs')

    ax[0, 2].hist(mean_disp, bins=25, color='#C44E52', edgecolor='white')
    ax[0, 2].set(title='per-pair mean vertex dispersion', xlabel='geodesic spread / diam', ylabel='pairs')

    ax[1, 0].hist(spread, bins=25, color='#8172B3', edgecolor='white')
    ax[1, 0].set(title='MGE spread within pair (worst - oracle)', xlabel='dense MGE', ylabel='pairs')

    ax[1, 1].hist(sel_gap, bins=25, color='#CCB974', edgecolor='white')
    ax[1, 1].set(title='selector - oracle gap (0 = perfect selection)', xlabel='dense MGE', ylabel='pairs')

    ax[1, 2].scatter(rows['oracle'], rows['selector'], s=14, alpha=0.6, color='#4C72B0')
    lim = [0, max(rows['selector'].max(), rows['oracle'].max()) * 1.05]
    ax[1, 2].plot(lim, lim, 'k:', lw=1)
    ax[1, 2].set(title='per-pair: selector vs oracle', xlabel='oracle MGE', ylabel='selector MGE',
                 xlim=lim, ylim=lim)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _print(s):
    print(f"\n{s['name']}  split={s['split']}  K={s['K']}  eta={s['eta']}  w_iso={s['w_iso']}  "
          f"pairs={s['n_pairs']}\nckpt={s['checkpoint']}")
    print(f"\n{'method':>13} {'dense MGE':>11} {'median':>9} {'gross>0.1':>11} {'oracle-recov':>13}")
    print('-' * 60)
    for lab, key in [('feature-NN', 'feature_nn'), ('single K=1', 'single'),
                     ('SELECTOR', 'selector'), ('worst-of-K', 'worst'), ('ORACLE', 'oracle')]:
        b = s[key]
        rec = '' if b['oracle_recovery'] is None else f"{b['oracle_recovery']*100:.0f}%"
        print(f"{lab:>13} {b['dense_MGE']:>11.4f} {b['median']:>9.4f} "
              f"{b['gross_gt_0.1']*100:>10.1f}% {rec:>13}")
    print('-' * 60)
    m = s['multimodality']
    print(f"multimodality: mean n_modes={m['mean_n_modes']:.2f}  "
          f"frac multimodal(>1)={m['frac_multimodal']*100:.0f}%  "
          f"median dominant occ={m['median_dominant_occupancy']*100:.0f}%  "
          f"mean vertex dispersion={m['mean_vertex_dispersion']:.3f}")
    if s['exported_pairs']:
        print("\nexported for polyscope (most multimodal):")
        for e in s['exported_pairs']:
            print(f"  pair {e['pair_idx']:>3}  n_modes={e['n_modes']}  -> {e['file']}")
    print(f"\nhistograms: {os.path.join(s['out_dir'], 'histograms.png')}")
    print(f"summary:    {os.path.join(s['out_dir'], 'summary.json')}")


# --------------------------------------------------------------------------- #
# polyscope viewer
# --------------------------------------------------------------------------- #
def view(path):
    """Open one exported pair in polyscope: Y colored by entropy/dispersion + per-map error,
    X drawn beside it. Needs polyscope (pip install polyscope) and a display."""
    import polyscope as ps
    d = np.load(path)
    yv, yf, xv, xf = d['y_verts'], d['y_faces'], d['x_verts'], d['x_faces']
    offset = np.array([1.3 * (xv[:, 0].max() - xv[:, 0].min()) + 0.1, 0, 0])

    ps.init()
    ps.set_ground_plane_mode('none')
    Y = ps.register_surface_mesh('Y (source)', yv, yf, smooth_shade=True)
    Y.add_scalar_quantity('target dispersion (geo/diam)', d['dispersion'], enabled=True, cmap='viridis')
    Y.add_scalar_quantity('target entropy (norm)', d['entropy'], cmap='viridis')
    Y.add_scalar_quantity('err single', d['err_single'], cmap='reds')
    Y.add_scalar_quantity('err selector', d['err_selector'], cmap='reds')
    Y.add_scalar_quantity('err oracle', d['err_oracle'], cmap='reds')
    # transfer X.x-coordinate through the selector map so you can read the correspondence visually
    Y.add_scalar_quantity('selector map (X.x transferred)', xv[d['selector_map'], 0], cmap='coolwarm')

    X = ps.register_surface_mesh('X (target)', xv + offset, xf, smooth_shade=True)
    X.add_scalar_quantity('X.x', xv[:, 0], cmap='coolwarm')
    print(f"pair {int(d['pair_idx'])}: n_modes={int(d['n_modes'])}  "
          f"single MGE={float(d['mge'][0]):.4f}  selector MGE={float(d['mge'][int(d['pick'])]):.4f}  "
          f"oracle MGE={float(d['mge'][int(d['best'])]):.4f}")
    ps.show()


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--view', default=None, help='open an exported polyscope/pair_XX.npz and exit')
    p.add_argument('-c', '--config', help='training config (arch + dataset)')
    p.add_argument('--checkpoint', default=None, help='checkpoint override (e.g. FAUST model on SCAPE config)')
    p.add_argument('-K', type=int, default=32, help='samples per pair')
    p.add_argument('--num-pairs', type=int, default=60, help='cap pairs (densify is Kx); 0 = all')
    p.add_argument('--eta', type=float, default=0.25, help='DDIM->DDPM sampler stochasticity (diversity source)')
    p.add_argument('--w-iso', type=float, default=0.700, help='FROZEN selector weight: w*z(iso)+(1-w)*z(cov)')
    p.add_argument('--n-iso-pairs', type=int, default=2000, help='long-range Y pairs for the isometry score')
    p.add_argument('--mode-thresh', type=float, default=0.05,
                   help='cluster cut for mode counting (frac of X diameter map-disagreement)')
    p.add_argument('--export-pairs', type=int, default=6, help='most-multimodal pairs to dump for polyscope')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--exclude-self', action='store_true', help='drop identity self-pairs')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None)
    args = p.parse_args()

    if args.view:
        view(args.view)
        return
    if not args.config:
        p.error('-c/--config is required unless --view is used')
    s = run(args.config, args.checkpoint, args.device, args.K, args.num_pairs, args.seed,
            args.eta, args.w_iso, args.n_iso_pairs, args.mode_thresh, args.split,
            args.exclude_self, args.export_pairs)
    _print(s)


if __name__ == '__main__':
    main()
