"""DIAGNOSTIC -- how much of the map is built by the denoiser's INNER block iterations?

The reverse (DDIM) trajectory is known to be inert: the match is feature-decided in the first
step or two and frozen thereafter (see the reverse-trajectory-inert memory). This asks the
complementary question with the outer loop collapsed to a SINGLE step -- inside one denoiser
call, does the assignment actually improve from block to block?

Each of the denoiser's blocks ends with `u = self._regauge(u)`, so every per-block state is a
gauged assignment logit in the same space as the returned u0_hat. Each one is therefore decoded
by the SAME Hungarian and lifted by the SAME densifier as the real pipeline, and scored with the
same geodesic metric -- so the per-block curve is directly comparable to the reported dense MGE.

States are captured with forward hooks on the block writes, NOT by editing the network: the hook
sees `write(hx, hy, u, c)` and this module applies `net._regauge` to it, exactly as the block loop
does. The reconstruction is checked every pair -- the last captured state must equal what forward
returned -- so a silent divergence from the real loop cannot go unnoticed.

Stages on the x-axis:
  input     -- the read-in P_t the denoiser conditions on. At one step that is the pure-noise
               prior, so this is the "no denoiser at all" endpoint.
  block_i   -- the state after block i, decoded and densified like any other.
  pipeline  -- the real single-step result (_reproject + _final_snap + Hungarian). Differs from
               the last block only by the sampler's final projection; it is the right endpoint
               and a cross-check that the whole chain is wired to the deployed path.

Runs with BP off by construction (the post-process is never attached, and an in-network BP stage
would sit between the write and the re-gauge, so it is refused rather than silently mis-captured).

Example:
    python -m diagnostics.block_trajectory -c configs/final/dt4d_mpnn_512_final_cold_co.yaml \\
        --set datasets.test.inter_class=true --tag dt4d_inter --num-pairs 100
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from diagnostics.nn_baseline_dense import (_benchmark_dir, _build, calculate_geodesic_error)
from models.base_model import to_numpy
from train import seed_everything
from utils.sinkhorn import safe_log


class _BlockTap:
    """Capture each block's written state via forward hooks on the denoiser's write modules.

    The block loop is `u = write(...)` then `u = self._regauge(u)`, so the hook output is the
    pre-gauge half and _regauge (applied here) completes it. Nothing in the network changes.
    """
    def __init__(self, net):
        if getattr(net, 'bp', None) is not None:
            raise ValueError(
                'this denoiser has an in-network BP stage, which writes to u between the block '
                'write and the re-gauge; the hooked states would not be the loop states. Run a '
                'config whose denoiser has no bp: block.')
        self.net = net
        self.raw = []
        self._handles = [w.register_forward_hook(self._hook) for w in net.write]

    def _hook(self, module, inputs, output):
        self.raw.append(output)

    def states(self):
        """Per-block states in the network's own gauge, [after block 0, ..., after block L-1]."""
        return [self.net._regauge(u) for u in self.raw]

    def reset(self):
        self.raw = []

    def close(self):
        for h in self._handles:
            h.remove()


@torch.no_grad()
def _single_step(model, tap, F_x, F_y, D_x, D_y):
    """One DDIM step, mirroring MPNNDiffusionModel.sample(steps=1), with the block states kept.

    At steps=1 the schedule is t: 1 -> 0, where alpha_bar(0) = 1 kills the noise term, so the
    iterate after the update is just the (reprojected) u0_hat. Returns (states, u_final).
    """
    net = model.networks['denoiser']
    n = F_x.shape[1]
    u = torch.randn(1, n, n, device=model.device)          # alpha_bar(1) = 0 prior
    P_t = model._read_in(u)

    tap.reset()
    t = torch.ones(1, device=model.device)
    u0_hat = net(P_t, F_x, F_y, D_x, D_y, t)
    states = tap.states()
    assert torch.allclose(states[-1], u0_hat, atol=1e-5), \
        'hook reconstruction diverged from the network output -- the block loop has changed'

    u_final = model._reproject(u0_hat) if model.reproject else u0_hat
    return [safe_log(P_t)] + states, u_final


@torch.no_grad()
def run(config_path, checkpoint, device, num_pairs, overrides, tag, seed):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, 'config',
                                       overrides=overrides)
    seed_everything(seed)
    net = model.networks['denoiser']
    tap = _BlockTap(net)

    n = len(dataset) if not num_pairs else min(num_pairs, len(dataset))
    idxs = list(range(n))
    rows = []
    try:
        for i in tqdm(idxs, desc=f'block trajectory ({opt["name"]}, 1 DDIM step, no bp)'):
            data = dataset[i]
            F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)
            states, u_final = _single_step(model, tap, F_x, F_y, D_x, D_y)

            # one context per pair, reused by every stage: densify() only reads it, and building
            # it runs the extractor densely over both meshes -- by far the costliest step here
            ctx = model._densify_context(data)
            dist_x = to_numpy(data['first']['dist'])
            corr_x, corr_y = to_numpy(data['first']['corr']), to_numpy(data['second']['corr'])

            mges = []
            for u in states + [u_final]:                   # input, blocks..., pipeline
                p2p = model._decode(model._final_snap(u)[0])
                dense_p2p = model.densifier.densify(p2p, ctx)
                err = calculate_geodesic_error(dist_x, corr_x, corr_y, to_numpy(dense_p2p),
                                               return_mean=False)
                mges.append(float(err.mean()))
            rows.append(mges)
    finally:
        tap.close()                                        # never leave hooks on the model

    mge = np.asarray(rows)                                 # (pairs, stages)
    depth = mge.shape[1] - 2
    stages = ['input'] + [f'block_{i}' for i in range(depth)] + ['pipeline']
    summary = {'name': opt['name'], 'checkpoint': ckpt, 'n_pairs': int(mge.shape[0]),
               'depth': depth, 'stages': stages, 'sample_steps': 1, 'bp': None,
               'dataset': dict(opt['datasets']['test']), 'overrides': list(overrides or []),
               'mean_MGE': {s: float(mge[:, k].mean()) for k, s in enumerate(stages)},
               'median_MGE': {s: float(np.median(mge[:, k])) for k, s in enumerate(stages)}}

    out_dir = _benchmark_dir(ckpt, opt['name'])
    os.makedirs(out_dir, exist_ok=True)
    stem = 'block_trajectory' + (f'__{tag}' if tag else '')
    np.savez(os.path.join(out_dir, f'{stem}.npz'), mge=mge, stages=np.array(stages))
    with open(os.path.join(out_dir, f'{stem}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    _plot(mge, stages, opt['name'], os.path.join(out_dir, f'{stem}.png'))
    summary['out_dir'] = out_dir
    return summary


def _plot(mge, stages, name, path):
    """Mean dense MGE per stage with an IQR band -- per-pair cost is skewed, so show spread."""
    x = np.arange(len(stages))
    q1, q3 = np.percentile(mge, [25, 75], axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(x, q1, q3, alpha=0.2, label='IQR over pairs')
    ax.plot(x, np.median(mge, axis=0), marker='o', label='median')
    ax.plot(x, mge.mean(axis=0), marker='s', linestyle='--', label='mean')
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=45, ha='right')
    ax.set_ylabel('dense MGE (densified, whole shape)')
    ax.set_title(f'{name} -- per-block assignment quality, 1 DDIM step')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _print(s):
    print(f"\nexperiment : {s['name']}")
    print(f"pairs      : {s['n_pairs']}  |  depth: {s['depth']}  |  1 DDIM step, BP off")
    print(f"{'stage':<12}{'mean MGE':>12}{'median MGE':>14}")
    for st in s['stages']:
        print(f"{st:<12}{s['mean_MGE'][st]:>12.4f}{s['median_MGE'][st]:>14.4f}")
    print(f"\ncurve + per-pair MGE under: {s['out_dir']}/")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True, help='the trained model\'s config')
    p.add_argument('--checkpoint', default=None, help='checkpoint override')
    p.add_argument('--num-pairs', type=int, default=100,
                   help='pairs to trace, taken in order; 0 = the whole test set')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    p.add_argument('--seed', type=int, default=0, help='global seed, as in evaluate.py')
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                   help='evaluate.py-style dotted-key override, repeatable')
    p.add_argument('--tag', default=None, help='suffix for the output files, so regimes never clobber')
    args = p.parse_args()

    torch.set_float32_matmul_precision('high')             # evaluate.py's numerics
    _print(run(args.config, args.checkpoint, args.device, args.num_pairs,
               args.overrides, args.tag, args.seed))


if __name__ == '__main__':
    main()
