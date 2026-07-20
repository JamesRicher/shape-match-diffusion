"""ISOLATED, DELETE-SAFE DIAGNOSTIC -- is the sampler multimodal, and does it use P_t?

Lives entirely under diagnostics/; all output goes to diagnostics/results/. Modifies no
tracked file. Delete the diagnostics/ folder to remove it and its data.

WHY
---
The sparse map behaves "almost deterministically". That is NOT the same as "the denoiser
ignores P_t" (the P_t pathway weights can be significant and the output still be near-
deterministic if the reverse dynamics are contractive to a feature-conditioned attractor,
and/or the single-GT row-CE loss has trained the map distribution down to one mode). The
weight check can't see this; the OUTPUT-level quantities can. This script reports both, per
pair, from the model's own diagnostics (no new model code):

  trajectory_divergence(n_samples)  -- mean pairwise disagreement (fraction of sparse points
      mapped differently) across independent prior draws. ~0 => every noise draw collapses to
      the SAME map (no diversity to exploit for a 'sample + select' symmetry fix); >0 => the
      sampler spreads over modes (on a symmetric pose that should include the flipped mode).

  loss_vs_t slope                   -- Row-CE as a function of diffusion time t. slope =
      mean(high-t loss) - mean(low-t loss). Positive => loss falls toward clean/small t, i.e.
      the denoiser DOES condition on P_t. ~0 (flat) => it is not using P_t. (This is the
      architectural check; you verified the weights survive -- this confirms it end-to-end.)

READING THE RESULT
------------------
  divergence ~ 0  &  slope > 0   -> P_t is used, but the sampler is collapsed to one mode
                                    (contractive dynamics / unimodal loss). Turning up noise
                                    won't help; the fix is a symmetry-aware (multi-hypothesis)
                                    loss so the model is ALLOWED to represent both maps.
  divergence > 0                 -> the sampler already spreads over modes; 'sample coherent
                                    maps + select with a weak signal' is worth prototyping now.
  slope ~ 0                      -> contradicts the weight check; P_t is not affecting the
                                    output -> look at conditioning wiring first.

These run in the default (bijective) sampling regime, which is what trajectory_divergence was
designed for (it needs gt_perm for loss_vs_t; the symmetric partner is usually present among
the FPS anchors, so mode-spreading is still detectable).

USAGE
-----
  python -m diagnostics.symmetry_stochasticity \
      -c configs/joint_gcn_diffusion/faust_matrix_diffusion_gcn_512_FMD.yaml \
         configs/joint_gcn_diffusion/faust_matrix_diffusion_gcn_512_redo.yaml \
         configs/joint_gcn_diffusion/runA_patch64_geo.yaml \
         configs/joint_gcn_diffusion/runB_patch48_euc.yaml \
      --num-pairs 8

COMPUTE: each pair runs n_samples full DDIM samples + t_bins*t_repeats single forwards. Start
with a small --num-pairs / --n-samples on the box before a big sweep.
"""
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from datasets import build_dataset
from models import build_model
from train import autofill_feat_dim
from utils.options import load_yaml, resolve_experiment_paths

_OUT_ROOT = os.path.join(os.path.dirname(__file__), 'results')


def _build(config_path, checkpoint, device, split):
    """Load a trained checkpoint + its dataset as evaluate.py does. Default (bijective)
    sampling regime -- gt_perm is present for loss_vs_t; independent_fps left False."""
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

    dataset = build_dataset(opt['datasets'][split])
    autofill_feat_dim(opt, int(dataset[0]['first']['feat'].shape[-1]))
    model = build_model(opt)
    model.eval()
    return model, dataset, opt, ckpt


def _pair_indices(dataset, num_pairs, explicit):
    if explicit:
        return [i for i in explicit if 0 <= i < len(dataset)]
    n = min(num_pairs, len(dataset))
    return list(np.linspace(0, len(dataset) - 1, n).round().astype(int))


def run_one(config_path, checkpoint, device, split, indices_arg, num_pairs,
            n_samples, t_bins, t_repeats):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, split)
    name = opt['name']
    idxs = _pair_indices(dataset, num_pairs, indices_arg)

    per_pair = []
    for i in tqdm(idxs, desc=f'{name} stochasticity'):
        data = dataset[int(i)]
        div = float(model.trajectory_divergence(data, n_samples=n_samples))
        curve = model.loss_vs_t(data, n_bins=t_bins, repeats=t_repeats)
        vals = list(curve.values())
        half = len(vals) // 2
        slope = float(np.mean(vals[half:]) - np.mean(vals[:half]))
        per_pair.append({'pair': int(i), 'trajectory_divergence': div,
                         'loss_t_slope': slope, 'loss_vs_t': curve})

    divs = np.array([p['trajectory_divergence'] for p in per_pair])
    slopes = np.array([p['loss_t_slope'] for p in per_pair])
    summary = {
        'name': name, 'checkpoint': ckpt, 'n_pairs': len(idxs),
        'n_samples': n_samples, 't_bins': t_bins, 't_repeats': t_repeats,
        'divergence_mean': float(divs.mean()), 'divergence_median': float(np.median(divs)),
        'divergence_max': float(divs.max()), 'slope_mean': float(slopes.mean()),
        'per_pair': per_pair,
    }
    out_dir = os.path.join(_OUT_ROOT, f'{name}_stochasticity')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'stochasticity.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print_table(summaries):
    print("\ntrajectory_divergence: 0 = every prior draw collapses to the SAME map; "
          ">0 = spreads over modes")
    print("loss_t_slope: >0 = denoiser conditions on P_t; ~0 = it does not\n")
    head = f"{'experiment':30s}{'div mean':>10s}{'div med':>10s}{'div max':>10s}{'slope mean':>12s}"
    print(head); print('-' * len(head))
    for s in summaries:
        print(f"{s['name']:30s}{s['divergence_mean']:10.3f}{s['divergence_median']:10.3f}"
              f"{s['divergence_max']:10.3f}{s['slope_mean']:12.3f}")
    print("\ndiv~0 & slope>0 => P_t used but sampler collapsed (fix = symmetry-aware loss, not more noise).")
    print("div>0           => already multimodal => 'sample coherent maps + select' is viable now.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', nargs='+', required=True, help='one or more training configs')
    p.add_argument('--checkpoint', default=None, help='checkpoint override (only with a single -c)')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--num-pairs', type=int, default=8, help='pairs to probe (evenly spaced)')
    p.add_argument('--pair-indices', type=int, nargs='+', default=None,
                   help='explicit dataset pair indices (overrides --num-pairs)')
    p.add_argument('--n-samples', type=int, default=8, help='prior draws for trajectory divergence')
    p.add_argument('--t-bins', type=int, default=10, help='diffusion-time bins for loss-vs-t')
    p.add_argument('--t-repeats', type=int, default=16, help='noise draws per t bin')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    args = p.parse_args()
    if args.checkpoint and len(args.config) > 1:
        p.error('--checkpoint can only be used with a single -c')

    summaries = []
    for cfg in args.config:
        summaries.append(run_one(cfg, args.checkpoint, args.device, args.split,
                                 args.pair_indices, args.num_pairs,
                                 args.n_samples, args.t_bins, args.t_repeats))
    _print_table(summaries)
    print(f"\nper-experiment JSON (incl. per-pair loss-vs-t curves) under: {_OUT_ROOT}/<name>_stochasticity/")


if __name__ == '__main__':
    main()
