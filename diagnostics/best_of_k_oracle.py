"""ISOLATED, DELETE-SAFE DIAGNOSTIC -- oracle best-of-K: is sample-and-select viable?

Draws K diffusion samples per pair, densifies each, and reports dense MGE for:
  * feature-NN   -- the greedy baseline (diagnostics/nn_baseline_dense.py matcher).
  * single       -- one diffusion sample (the current model, = K=1).
  * oracle-K     -- per pair, the BEST of the K samples under the TRUE GT metric.
  * worst-K      -- per pair, the worst of the K (shows the spread the selector must beat).

oracle-K is the upper bound on ANY selection rule, so it is a clean go/no-go for the whole
sample-and-select idea (approach #1):

  oracle-K << single  -> the correct mode IS in the K samples; multimodality survives; build
      the unsupervised selector next.
  oracle-K ~= single  -> the K samples collapse to the same (often flipped) answer; diversity
      is on the wrong points; select cannot help without changing features/training/sampler.

The sampler is DDIM (deterministic given the init), so the only diversity source is the init
noise; --eta is a placeholder for a future stochastic sampler and is ignored here.

USAGE (the decisive cross-dataset run, FAUST model on SCAPE data):
  python -m diagnostics.best_of_k_oracle \
      -c configs/joint_diffusionnet/scape_diffusionnet_512_FMD.yaml \
      --checkpoint experiments/faust_diffusionnet_512_FMD/models/final.pth \
      -K 8 --num-pairs 60

Optional: --num-pairs N (0 = all; cap it, densify is K x slower), --seed, --device.
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from diagnostics.nn_baseline_dense import _build, _dense_mge, _feature_nn_sparse_map
from models.base_model import to_numpy

_OUT_ROOT = os.path.join(os.path.dirname(__file__), 'results')


def _hungarian(P0):
    """(n_y, n_x) DS -> (n_y,) sparse Y->X via Hungarian, as validate_single does."""
    r, c = linear_sum_assignment(-to_numpy(P0))
    p2p = np.empty(P0.shape[0], dtype=np.int64)
    p2p[r] = c
    return torch.from_numpy(p2p)


@torch.no_grad()
def _k_sample_maps(model, data, K):
    """K sparse maps from K diffusion samples (K different noise inits, batched)."""
    F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)
    rep = lambda z: z.repeat(K, *([1] * (z.dim() - 1)))
    P0 = model.sample(rep(F_x), rep(F_y), rep(D_x), rep(D_y))     # (K, n_y, n_x)
    return [_hungarian(P0[k]) for k in range(K)]


@torch.no_grad()
def run(config_path, checkpoint, device, K, num_pairs, seed):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, 'config')
    name = opt['name']
    idxs = (list(range(len(dataset))) if not num_pairs else
            sorted(np.random.default_rng(seed).choice(len(dataset),
                   size=min(num_pairs, len(dataset)), replace=False).tolist()))

    nn_mge, single_mge, oracle_mge, worst_mge = [], [], [], []
    for i in tqdm(idxs, desc=f'{name} best-of-{K} (dense MGE)'):
        data = dataset[i]
        nn_mge.append(_dense_mge(model, data, _feature_nn_sparse_map(model, data)).mean())
        per_sample = [ _dense_mge(model, data, m).mean() for m in _k_sample_maps(model, data, K) ]
        per_sample = np.array(per_sample)
        single_mge.append(per_sample[0])          # K=1: one sample, the current model
        oracle_mge.append(per_sample.min())       # best of K under true GT (upper bound)
        worst_mge.append(per_sample.max())

    def agg(v):
        v = np.asarray(v)
        return {'dense_MGE': float(v.mean()), 'median': float(np.median(v)),
                'gross_gt_0.1': float(np.mean(v > 0.1))}

    # the decisive subset: pairs the single sample gets GROSS (>0.1). Does oracle-K rescue them?
    single_arr, oracle_arr = np.asarray(single_mge), np.asarray(oracle_mge)
    flipped = single_arr > 0.1
    n_flip = int(flipped.sum())
    flip_block = None
    if n_flip:
        flip_block = {'n_flipped_pairs': n_flip,
                      'single_mean_on_flipped': float(single_arr[flipped].mean()),
                      'oracle_mean_on_flipped': float(oracle_arr[flipped].mean()),
                      'rescued_frac': float(np.mean(oracle_arr[flipped] <= 0.1))}  # oracle brings <0.1

    summary = {'name': name, 'checkpoint': ckpt, 'K': K, 'n_pairs': len(idxs),
               'feature_nn': agg(nn_mge), 'single_sample': agg(single_mge),
               'oracle_K': agg(oracle_mge), 'worst_K': agg(worst_mge),
               'oracle_gain_vs_single': float(single_arr.mean() - oracle_arr.mean()),
               'oracle_gain_vs_nn': float(np.mean(nn_mge) - oracle_arr.mean()),
               'on_flipped_pairs': flip_block}
    out_dir = os.path.join(_OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, 'best_of_k.npz'),
             nn=nn_mge, single=single_mge, oracle=oracle_mge, worst=worst_mge)
    with open(os.path.join(out_dir, 'best_of_k.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print(s):
    print(f"\n{s['name']}   K={s['K']}   pairs={s['n_pairs']}   ckpt={s['checkpoint']}")
    print(f"\n{'matcher':>14} {'dense MGE':>11} {'median':>9} {'gross>0.1':>11}")
    print('-' * 48)
    for key, lab in [('feature_nn', 'feature-NN'), ('single_sample', 'single (K=1)'),
                     ('worst_K', 'worst-of-K'), ('oracle_K', 'ORACLE-K')]:
        a = s[key]
        print(f"{lab:>14} {a['dense_MGE']:>11.4f} {a['median']:>9.4f} {a['gross_gt_0.1']*100:>10.1f}%")
    print('-' * 48)
    print(f"oracle-K gain vs single = {s['oracle_gain_vs_single']:+.4f}   "
          f"vs feature-NN = {s['oracle_gain_vs_nn']:+.4f}")
    fb = s.get('on_flipped_pairs')
    if fb:
        print(f"\nDECISIVE -- on the {fb['n_flipped_pairs']} pairs the single sample gets gross (>0.1):")
        print(f"  single mean {fb['single_mean_on_flipped']:.4f} -> oracle-K mean {fb['oracle_mean_on_flipped']:.4f}"
              f"   rescued (<0.1) = {fb['rescued_frac']*100:.0f}%")
        print("  >> high rescued% => the correct mode is among the K; build the selector.")
        print("  >> ~0 rescued%   => all K flip together; select can't help.")
    else:
        print("\n(no single-sample gross pairs in this subset -- raise --num-pairs to hit the flip tail)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('-K', type=int, default=8, help='samples per pair')
    p.add_argument('--num-pairs', type=int, default=60, help='cap pairs (densify is Kx slower); 0 = all')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eta', type=float, default=0.0, help='(placeholder; DDIM sampler is deterministic)')
    p.add_argument('--device', default=None)
    args = p.parse_args()
    _print(run(args.config, args.checkpoint, args.device, args.K, args.num_pairs, args.seed))
    print(f"\nper-pair arrays + JSON under: {os.path.join(_OUT_ROOT, '<name>')}/")


if __name__ == '__main__':
    main()
