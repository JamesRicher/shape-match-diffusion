"""Per-pair inference runtime of the full pipeline, for the runtime table.

Times METHOD work only -- from in-memory tensors to the final dense p2p -- so the number is
comparable to ULRSSM's, whose timer also starts after to_device and stops before the metric.
Dataset loading (the dist matrices dominate wall clock), the geodesic metric and map saving are
all outside the timed region, which is why the tqdm s/it of a normal eval is NOT this number.

Every phase boundary is CUDA-synchronised. Without that, async kernels get charged to whichever
later op happens to synchronise, which silently moves time between phases (and, in a timer that
never syncs at all, out of the measured region entirely).

Phases, when BP is off:
  feat     -- _sparse_inputs: the extractor on both shapes + sparse token assembly
  sample   -- the DDIM reverse process (scales with diffusion.sample_steps)
  decode   -- Hungarian on the final assignment
  densify  -- _densify_context (dense extractor pass) + the functional-map lift
With --bp the sparse map comes from validate_single (BP is attached to it wholesale, as in
evaluate.py), so feat/sample/decode collapse into one 'match' phase.

Reported per phase and for the total: mean, sd, median and IQR over pairs. Per-pair times are
kept in the npz -- the distribution is skewed by mesh size, so the sd alone understates spread.

Example:
    python -m diagnostics.runtime_bench -c configs/final/faust_mpnn_512_final_cold_co.yaml \\
        --set datasets.test.exclude_self=false --tag faust_FINAL --num-pairs 50 --bp
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from diagnostics.nn_baseline_dense import _benchmark_dir, _build
from evaluate import attach_bp_postprocess
from train import seed_everything


def _stats(times):
    """mean/sd/median/IQR of one phase's per-pair times, in milliseconds."""
    t = np.asarray(times, dtype=float) * 1e3
    q1, q3 = np.percentile(t, [25, 75])
    return {'mean_ms': float(t.mean()), 'sd_ms': float(t.std(ddof=1)) if t.size > 1 else 0.0,
            'median_ms': float(np.median(t)), 'iqr_ms': float(q3 - q1),
            'min_ms': float(t.min()), 'max_ms': float(t.max()), 'n': int(t.size)}


class _Phases:
    """Accumulate CUDA-synchronised phase timings, one row per pair."""
    def __init__(self, device):
        self.cuda = torch.device(device).type == 'cuda'
        self.rows = {}

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def start(self):
        self.sync()
        self._t = time.perf_counter()

    def stop(self, name):
        """Close the current phase and immediately open the next one."""
        self.sync()
        now = time.perf_counter()
        self.rows.setdefault(name, []).append(now - self._t)
        self._t = now


@torch.no_grad()
def run(config_path, checkpoint, device, num_pairs, warmup, with_bp, overrides, tag, seed):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, 'config',
                                       overrides=overrides)
    seed_everything(seed)
    bp_params = attach_bp_postprocess(model, opt) if with_bp else None

    n = len(dataset) if not num_pairs else min(num_pairs, len(dataset))
    idxs = list(range(min(n + warmup, len(dataset))))
    ph = _Phases(model.device)
    n_verts = []

    for pos, i in enumerate(tqdm(idxs, desc=f'runtime ({opt["name"]}, '
                                            f'{"bp" if with_bp else "no bp"})')):
        data = dataset[i]                                   # NOT timed: disk + host->device
        measure = pos >= warmup                             # first pairs prime CUDA/cudnn
        ph.start()
        if with_bp:
            p2p = model.validate_single(data)               # extractor + sample + BP + Hungarian
            ph.stop('match')
        else:
            F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)
            ph.stop('feat')
            P0 = model.sample(F_x, F_y, D_x, D_y)[0]
            ph.stop('sample')
            p2p = model._decode(P0)
            ph.stop('decode')
        model.densifier.densify(p2p, model._densify_context(data))
        ph.stop('densify')
        if not measure:                                     # discard the warm-up rows
            for v in ph.rows.values():
                v.pop()
        else:
            n_verts.append(int(data['second']['verts'].shape[0]))

    phases = {k: np.asarray(v) for k, v in ph.rows.items()}
    total = np.sum(np.stack(list(phases.values())), axis=0)
    summary = {'name': opt['name'], 'checkpoint': ckpt, 'device': str(model.device),
               'n_pairs': int(total.size), 'warmup': warmup,
               'dataset': dict(opt['datasets']['test']),
               'sample_steps': opt.get('diffusion', {}).get('sample_steps'),
               'n_sparse': opt['datasets']['test'].get('n_sparse'),
               'bp': bp_params, 'overrides': list(overrides or []),
               'mean_target_verts': float(np.mean(n_verts)),
               'total': _stats(total),
               'phases': {k: _stats(v) for k, v in phases.items()}}

    out_dir = _benchmark_dir(ckpt, opt['name'])
    os.makedirs(out_dir, exist_ok=True)
    stem = 'runtime' + (f'__{tag}' if tag else '') + ('' if with_bp else '__nobp')
    np.savez(os.path.join(out_dir, f'{stem}.npz'), total=total, n_verts=np.asarray(n_verts),
             **phases)
    summary['out_dir'] = out_dir
    with open(os.path.join(out_dir, f'{stem}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print(s):
    print(f"\nexperiment : {s['name']}")
    print(f"device     : {s['device']}  |  pairs: {s['n_pairs']} (warm-up {s['warmup']} dropped)")
    print(f"settings   : n_sparse={s['n_sparse']} sample_steps={s['sample_steps']} "
          f"bp={'on' if s['bp'] else 'off'}")
    print(f"{'phase':<10}{'mean ms':>12}{'sd ms':>10}{'median ms':>12}{'IQR ms':>10}")
    for k, v in list(s['phases'].items()) + [('TOTAL', s['total'])]:
        print(f"{k:<10}{v['mean_ms']:>12.1f}{v['sd_ms']:>10.1f}"
              f"{v['median_ms']:>12.1f}{v['iqr_ms']:>10.1f}")
    print(f"\nper-pair times + JSON under: {s['out_dir']}/")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True, help='the trained model\'s config')
    p.add_argument('--checkpoint', default=None, help='checkpoint override')
    p.add_argument('--num-pairs', type=int, default=50,
                   help='pairs to time, taken in order; 0 = the whole test set')
    p.add_argument('--warmup', type=int, default=5,
                   help='leading pairs to run but discard (CUDA context, cudnn autotune)')
    p.add_argument('--bp', action='store_true',
                   help="attach the config's BP post-process, as evaluate.py does")
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    p.add_argument('--seed', type=int, default=0, help='global seed, as in evaluate.py')
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                   help='evaluate.py-style dotted-key override, repeatable')
    p.add_argument('--tag', default=None,
                   help='suffix for the output files, so regimes never clobber each other')
    args = p.parse_args()

    # match evaluate.py's numerics: TF32 changes how long the matmuls take, not just their values
    torch.set_float32_matmul_precision('high')
    _print(run(args.config, args.checkpoint, args.device, args.num_pairs, args.warmup,
               args.bp, args.overrides, args.tag, args.seed))


if __name__ == '__main__':
    main()
