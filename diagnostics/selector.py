"""ISOLATED, DELETE-SAFE DIAGNOSTIC -- unsupervised best-of-K selector, validated vs the oracle.

Draws K diffusion samples/pair, densifies each, and picks one WITHOUT ground truth using three
intrinsic map-quality scores (lower = better), each at its natural level:

  * isometry -- dense Gromov-Wasserstein distortion over long-range Y pairs:
        mean_w |D_X(pi(i),pi(j)) - D_Y(i,j)|,  area-weighted. General quality + partial-flip signal.
  * coverage -- dense pushforward TV: 0.5*sum_k |mu(k)/sum mu - m_X(k)/sum m_X|, where
        mu(k)=sum_{i:pi(i)=k} m_Y(i). Symmetry-robust flip/collapse detector.
  * cycle    -- sparse round-trip from P0: fwd=row-argmax, rev=col-argmax, geodesic Y-return
        error D_Y(i, rev[fwd[i]]). Cheap; partly redundant with coverage.

The selector picks argmin of each score alone and of the z-scored sum, then reports the resulting
dense MGE against the GT oracle (best-of-K under true MGE) and the single sample. Also reports
each score's Spearman correlation with true per-sample MGE and its oracle-recovery %, so the
validation -- not a guess -- says which score(s) to keep.

USAGE (the decisive cross-dataset run, FAUST model on SCAPE data):
  python -m diagnostics.selector \
      -c configs/joint_diffusionnet/scape_diffusionnet_512_FMD.yaml \
      --checkpoint experiments/faust_diffusionnet_512_FMD/models/final.pth \
      -K 16 --num-pairs 100 --eta 0.5

Optional: --num-pairs N (0=all), --eta (DDIM->DDPM stochasticity), --n-iso-pairs, --seed, --device.
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import spearmanr

from diagnostics.nn_baseline_dense import _build, _dense_mge, _feature_nn_sparse_map
from diagnostics.best_of_k_oracle import _hungarian
from metrics import build_metric
from models.base_model import to_numpy

_OUT_ROOT = os.path.join(os.path.dirname(__file__), 'results')
calculate_geodesic_error = build_metric({"type": "calculate_geodesic_error"})

SELECTORS = ['isometry', 'coverage', 'cycle', 'iso+cov', 'combined']


# --------------------------------------------------------------------------- #
# unsupervised per-sample scores (lower = better)
# --------------------------------------------------------------------------- #
def _isometry_defect(p2p, dist_x, dist_y, mass_y, i, j):
    """Area-weighted mean |D_X(pi(i),pi(j)) - D_Y(i,j)| over a fixed long-range pair set."""
    dY = dist_y[i, j]
    dX = dist_x[p2p[i], p2p[j]]
    w = mass_y[i] * mass_y[j]
    return float((w * np.abs(dX - dY)).sum() / w.sum())


def _coverage_defect(p2p, mass_x, mass_y):
    """TV between Y's pushforward area measure on X and X's own area measure."""
    mu = np.zeros_like(mass_x)
    np.add.at(mu, p2p, mass_y)
    mu = mu / mu.sum()
    mx = mass_x / mass_x.sum()
    return float(0.5 * np.abs(mu - mx).sum())


def _cycle_defect(P0, dist_y_sparse):
    """Sparse round-trip Y->X->Y geodesic return error from the soft assignment P0 (n_y, n_x)."""
    fwd = to_numpy(P0.argmax(1))                     # (n_y,) Y_sparse -> X_sparse
    rev = to_numpy(P0.argmax(0))                     # (n_x,) X_sparse -> Y_sparse
    back = rev[fwd]                                   # (n_y,) Y_sparse -> Y_sparse
    return float(dist_y_sparse[np.arange(len(fwd)), back].mean())


def _long_range_pairs(dist_y, n_pairs, seed):
    """Pick the n_pairs farthest-apart Y-vertex pairs among a random pool (long-range emphasis)."""
    rng = np.random.default_rng(seed)
    Ny = dist_y.shape[0]
    ii = rng.integers(0, Ny, size=4 * n_pairs)
    jj = rng.integers(0, Ny, size=4 * n_pairs)
    keep = np.argsort(dist_y[ii, jj])[-n_pairs:]
    return ii[keep], jj[keep]


def _zscore(a):
    s = a.std()
    return (a - a.mean()) / s if s > 1e-12 else np.zeros_like(a)


# --------------------------------------------------------------------------- #
# main sweep
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _sample_P0(model, data, K, sample_eta):
    F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)
    rep = lambda z: z.repeat(K, *([1] * (z.dim() - 1)))
    return model.sample(rep(F_x), rep(F_y), rep(D_x), rep(D_y), sample_eta=sample_eta)  # (K,ny,nx)


@torch.no_grad()
def run(config_path, checkpoint, device, K, num_pairs, seed, sample_eta, n_iso_pairs, split, exclude_self):
    from tqdm import tqdm
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, 'config',
                                       split=split, exclude_self=exclude_self)
    name = opt['name']
    idxs = (list(range(len(dataset))) if not num_pairs else
            sorted(np.random.default_rng(seed).choice(len(dataset),
                   size=min(num_pairs, len(dataset)), replace=False).tolist()))

    # per-pair: true MGE of single/oracle/each selector; pooled (score, mge) for correlation;
    # and WITHIN-pair rank correlation (the number that actually predicts selection quality --
    # pooled correlation is inflated by between-pair difficulty, which selection cannot exploit).
    rows = {k: [] for k in ['feature_nn', 'single', 'oracle'] + SELECTORS}
    pooled = {m: [] for m in ['isometry', 'coverage', 'cycle']}
    pooled_mge = []
    within = {m: [] for m in ['isometry', 'coverage', 'cycle']}
    raw = {m: [] for m in ['isometry', 'coverage', 'cycle']}   # per-pair (K,) score arrays
    raw_mge = []                                                # per-pair (K,) true MGE -> offline tuning

    for pi in tqdm(idxs, desc=f'{name} selector K={K} eta={sample_eta}'):
        data = dataset[pi]
        x, y = data['first'], data['second']
        ctx = model._densify_context(data)
        dist_x, dist_y = to_numpy(x['dist']), to_numpy(y['dist'])
        mass_x, mass_y = to_numpy(x['mass']), to_numpy(y['mass'])
        corr_x, corr_y = to_numpy(x['corr']), to_numpy(y['corr'])
        dy_sparse = to_numpy(y['sparse']['dist'])
        ii, jj = _long_range_pairs(dist_y, n_iso_pairs, seed + pi)

        P0 = _sample_P0(model, data, K, sample_eta)
        mge = np.empty(K)
        sc = {m: np.empty(K) for m in ['isometry', 'coverage', 'cycle']}
        for k in range(K):
            sp = _hungarian(P0[k])
            dense = to_numpy(model.densifier.densify(sp, ctx))
            mge[k] = calculate_geodesic_error(dist_x, corr_x, corr_y, dense, return_mean=False).mean()
            sc['isometry'][k] = _isometry_defect(dense, dist_x, dist_y, mass_y, ii, jj)
            sc['coverage'][k] = _coverage_defect(dense, mass_x, mass_y)
            sc['cycle'][k] = _cycle_defect(P0[k], dy_sparse)

        rows['feature_nn'].append(_dense_mge(model, data, _feature_nn_sparse_map(model, data)).mean())
        rows['single'].append(mge[0])
        rows['oracle'].append(mge.min())
        rows['isometry'].append(mge[sc['isometry'].argmin()])
        rows['coverage'].append(mge[sc['coverage'].argmin()])
        rows['cycle'].append(mge[sc['cycle'].argmin()])
        iso_cov = _zscore(sc['isometry']) + _zscore(sc['coverage'])           # cycle dropped
        rows['iso+cov'].append(mge[iso_cov.argmin()])
        combined = iso_cov + _zscore(sc['cycle'])
        rows['combined'].append(mge[combined.argmin()])
        for m in pooled:
            pooled[m].extend(sc[m].tolist())
            raw[m].append(sc[m])
            if mge.std() > 1e-9 and sc[m].std() > 1e-9:          # within-pair rank corr
                within[m].append(spearmanr(sc[m], mge).correlation)
        pooled_mge.extend(mge.tolist())
        raw_mge.append(mge)

    rows = {k: np.asarray(v) for k, v in rows.items()}
    single_m, oracle_m = rows['single'].mean(), rows['oracle'].mean()

    def block(v):
        return {'dense_MGE': float(v.mean()), 'gross_gt_0.1': float(np.mean(v > 0.1)),
                'oracle_recovery': (float((single_m - v.mean()) / (single_m - oracle_m))
                                    if single_m > oracle_m else None)}

    corr_pooled = {m: float(spearmanr(pooled[m], pooled_mge).correlation) for m in pooled}
    corr_within = {m: (float(np.mean(within[m])) if within[m] else None) for m in pooled}
    summary = {'name': name, 'checkpoint': ckpt, 'K': K, 'sample_eta': sample_eta, 'split': split,
               'n_pairs': len(idxs), 'n_iso_pairs': n_iso_pairs,
               'feature_nn': block(rows['feature_nn']),
               'single': block(rows['single']), 'oracle': block(rows['oracle']),
               'selectors': {k: block(rows[k]) for k in SELECTORS},
               'score_spearman_within_pair': corr_within,
               'score_spearman_pooled': corr_pooled}

    out_dir = os.path.join(_OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    # tag with the source model + K + eta so runs never clobber (dir is keyed by the -c config,
    # so a FAUST checkpoint on a SCAPE config lands beside a native SCAPE run otherwise).
    ckpt_stem = os.path.basename(os.path.dirname(os.path.dirname(ckpt)))
    eta_tag = '' if sample_eta == 0.0 else f'_eta{sample_eta:g}'
    tag = f'selector_{ckpt_stem}_{split}_K{K}{eta_tag}'
    np.savez(os.path.join(out_dir, f'{tag}.npz'), **rows,
             score_mge=np.stack(raw_mge),                      # (n_pairs, K) true MGE per sample
             **{f'score_{m}': np.stack(raw[m]) for m in raw})  # (n_pairs, K) each score -> offline tuning
    with open(os.path.join(out_dir, f'{tag}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    summary['out_file'] = os.path.join(out_dir, f'{tag}.json')
    return summary


def _print(s):
    print(f"\n{s['name']}  split={s['split']}  K={s['K']}  eta={s['sample_eta']}  pairs={s['n_pairs']}  ckpt={s['checkpoint']}")
    print(f"\nscore->MGE Spearman  (within-pair = what selection can exploit; pooled inflated by difficulty):")
    print(f"    {'score':>9} {'within':>8} {'pooled':>8}")
    for m in s['score_spearman_within_pair']:
        w = s['score_spearman_within_pair'][m]
        ws = '  n/a ' if w is None else f"{w:+.3f}"
        print(f"    {m:>9} {ws:>8} {s['score_spearman_pooled'][m]:>+8.3f}")
    print(f"\n{'method':>12} {'dense MGE':>11} {'gross>0.1':>11} {'oracle-recovery':>16}")
    print('-' * 52)
    for lab, key in [('feature-NN', ('feature_nn',)), ('single K=1', ('single',)),
                     ('sel:isometry', ('selectors', 'isometry')),
                     ('sel:coverage', ('selectors', 'coverage')), ('sel:cycle', ('selectors', 'cycle')),
                     ('sel:iso+cov', ('selectors', 'iso+cov')),
                     ('sel:COMBINED', ('selectors', 'combined')), ('ORACLE', ('oracle',))]:
        b = s
        for kk in key:
            b = b[kk]
        rec = '' if b['oracle_recovery'] is None else f"{b['oracle_recovery']*100:.0f}%"
        print(f"{lab:>12} {b['dense_MGE']:>11.4f} {b['gross_gt_0.1']*100:>10.1f}% {rec:>16}")
    print('-' * 52)
    print("oracle-recovery = fraction of the single->oracle gap the selector captures (100% = oracle).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('-K', type=int, default=16, help='samples per pair')
    p.add_argument('--num-pairs', type=int, default=100, help='cap pairs (densify is Kx); 0 = all')
    p.add_argument('--eta', type=float, default=0.0, help='DDIM->DDPM sampler stochasticity')
    p.add_argument('--n-iso-pairs', type=int, default=2000, help='long-range Y pairs for isometry defect')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                   help="dataset split: tune hyperparameters on 'train' (held-out), report on 'test'")
    p.add_argument('--exclude-self', action='store_true', help='drop identity self-pairs (i==j)')
    p.add_argument('--seed', type=int, default=0, help='seed for the random pair subset (unbiased)')
    p.add_argument('--device', default=None)
    args = p.parse_args()
    s = run(args.config, args.checkpoint, args.device, args.K, args.num_pairs, args.seed,
            args.eta, args.n_iso_pairs, args.split, args.exclude_self)
    _print(s)
    print(f"\nwrote: {s['out_file']}")


if __name__ == '__main__':
    main()
