"""DIAGNOSTIC -- does the diffusion do anything beyond feature-NN?

Results land in the benchmarked model's OWN experiment dir, under
results/benchmarks/<eval_config_name>/ (see _benchmark_dir): a model is identified by its
checkpoint, and the leaf is keyed by the eval config so in-distribution and cross-dataset runs
sit side by side without clobbering.

WHAT IT ANSWERS
---------------
The worry: "all the information is already in the features; the diffusion denoiser adds
nothing." This runs the SAME forward pipeline but BYPASSES the diffusion model. For each test
pair it builds the sparse map two ways and pushes BOTH through the identical densifier + dense
geodesic metric, so the only thing that differs is the sparse matcher:

  * nn         -- honest-FPS sparse points matched purely by feature cosine-similarity
                  (argmax), on the exact trained features the denoiser consumes (_sparse_inputs).
  * hungarian  -- the SAME cosine-similarity matrix, but solved as a linear assignment instead
                  of a row argmax. Non-learned, and carries the bijectivity prior.
  * diffusion  -- the model's validate_single (sample + Hungarian), i.e. the real pipeline.

The hungarian arm is the control that separates two things nn-vs-diffusion confounds: the
diffusion arm gets BOTH a learned denoiser and a Hungarian snap, while nn gets neither, so a
diffusion win over nn alone cannot tell them apart. Unconstrained argmax fails loudly when
several sparse points collapse onto one target (fine median, huge p90), which linear assignment
repairs for free -- no shape knowledge required.

All three use independent (honest) FPS -- the regime the reported dense MGE lives in -- the same
densifier (feat_source and all), and metrics.calculate_geodesic_error exactly as evaluate.py's
validation() does. So:

  nn ~ hungarian ~ diffusion  -> the denoiser is dead weight for the final map.
  nn >> hungarian ~ diffusion -> the diffusion's apparent win is just bijectivity; a linear
      assignment on the raw features buys the same thing without the model.
  hungarian >> diffusion      -> the denoiser is doing real relational work beyond assignment.

USAGE
-----
  python -m diagnostics.nn_baseline_dense -c configs/joint_diffusionnet/faust_diffusionnet_512_FMD.yaml
  # cross-dataset (FAUST model on SCAPE data), like evaluate.py -c <scape cfg> -n <faust name>:
  python -m diagnostics.nn_baseline_dense -c configs/joint_diffusionnet/scape_diffusionnet_512_FMD.yaml \
      --checkpoint experiments/diffusionnet/faust_diffusionnet_512_FMD/models/final.pth

Optional: --num-pairs N (0 = all), --no-diffusion (feature arms only, faster), --no-hungarian,
--fps-metric
{config,geodesic,euclidean}, --device cuda/cpu, --seed (for the --num-pairs subset).
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from datasets import build_dataset
from metrics import build_metric
from models import build_model
from models.base_model import to_numpy
from train import autofill_feat_dim
from utils.options import load_yaml, resolve_experiment_paths

calculate_geodesic_error = build_metric({"type": "calculate_geodesic_error"})


def _benchmark_dir(ckpt, eval_name):
    """Where this benchmark's artifacts land: inside the BENCHMARKED MODEL's own experiment
    dir, under results/benchmarks/<eval_name>/. The model is identified by its checkpoint
    (``<experiment_root>/models/final.pth``), so experiment_root is two levels up -- this stays
    correct for cross-dataset runs, where the model (checkpoint) and the eval set (config)
    differ. Keying the leaf by the eval config keeps in-distribution and cross results side by
    side without clobbering (e.g. a FAUST model gets .../benchmarks/faust... and
    .../benchmarks/scape... for its cross run)."""
    experiment_root = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    return os.path.join(experiment_root, 'results', 'benchmarks', eval_name)


def _build(config_path, checkpoint, device, fps_metric, split='test', exclude_self=False):
    """Load a trained checkpoint + a dataset split like evaluate.py, forced into the honest
    independent-FPS regime with dense reporting on (the densifier is required). `split` selects
    datasets.<split> -- use 'train' as a held-out set for tuning selector hyperparameters, so the
    reported 'test' numbers stay uncontaminated (ret_evecs is forced on so the densifier works)."""
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
    opt.setdefault('eval', {})['dense'] = True                       # densifier path must be built

    ds_opt = dict(opt['datasets'][split])
    ds_opt['ret_evecs'] = True                                       # densifier needs eigenbases
    ds_opt.setdefault('num_evecs', opt['datasets'].get('test', {}).get('num_evecs', 128))
    if exclude_self:
        ds_opt['exclude_self'] = True                               # drop identity self-pairs
    dataset = build_dataset(ds_opt)
    dataset.independent_fps = True                                   # honest sampling (dense MGE regime)
    if fps_metric != 'config':
        dataset.fps_metric = fps_metric
    autofill_feat_dim(opt, int(dataset[0]['first']['feat'].shape[-1]))
    model = build_model(opt)
    model.eval()
    if getattr(model, 'densifier', None) is None:
        raise ValueError("this diagnostic needs a densifier (eval.dense pipeline); none configured")
    return model, dataset, opt, ckpt


def _feature_similarity(model, data):
    """(n_y, n_x) cosine similarity over the honest-FPS sparse points, using the exact per-point
    features the denoiser consumes. Shared by both feature arms so they differ only in solver."""
    F_x, F_y, _, _, _ = model._sparse_inputs(data)                  # (1, n, d) each, trained feats
    fx = F.normalize(F_x.squeeze(0), dim=-1)
    fy = F.normalize(F_y.squeeze(0), dim=-1)
    return fy @ fx.T


def _nn_sparse_map(sim):
    """Row argmax: each Y point takes its nearest X, with no bijectivity constraint (several Y
    points may collapse onto one X). Returns (n_y,) LongTensor of X sparse indices."""
    return sim.argmax(dim=1)


def _feature_nn_sparse_map(model, data):
    """Sparse Y->X map from pure feature cosine-NN. Kept as the one-call form the other
    diagnostics (selector.py, best_of_k_oracle.py) import."""
    return _nn_sparse_map(_feature_similarity(model, data))


def _hungarian_sparse_map(sim):
    """Optimal linear assignment on the same similarity, i.e. the NN arm plus bijectivity and
    nothing else. Mirrors validate_single's snap so the only difference from the diffusion arm is
    where the score matrix came from. Rows left unassigned when n_y > n_x fall back to argmax."""
    p2p = _nn_sparse_map(sim).cpu()
    row_ind, col_ind = linear_sum_assignment(-sim.detach().cpu().numpy())
    p2p[torch.as_tensor(row_ind)] = torch.as_tensor(col_ind)
    return p2p


def _dense_mge(model, data, sparse_p2p):
    """Densify a sparse map and score whole-shape geodesic error, identical to validation()."""
    dense_p2p = model.densifier.densify(sparse_p2p, model._densify_context(data))
    return calculate_geodesic_error(
        to_numpy(data['first']['dist']), to_numpy(data['first']['corr']),
        to_numpy(data['second']['corr']), to_numpy(dense_p2p), return_mean=False)


def _summary(err):
    err = np.ravel(err)
    return {'dense_MGE': float(err.mean()),
            'median': float(np.median(err)),
            'p90': float(np.percentile(err, 90)),
            'gross_gt_0.1': float(np.mean(err > 0.1))}


@torch.no_grad()
def run(config_path, checkpoint, device, fps_metric, num_pairs, seed, with_diffusion, with_hungarian):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, fps_metric)
    name = opt['name']
    if not num_pairs:
        idxs = list(range(len(dataset)))
    else:
        n = min(num_pairs, len(dataset))
        idxs = sorted(np.random.default_rng(seed).choice(len(dataset), size=n, replace=False).tolist())

    errs = {'nn': [], 'hungarian': [], 'diffusion': []}
    arms = ' vs '.join(['nn'] + ['hungarian'] * with_hungarian + ['diffusion'] * with_diffusion)
    for i in tqdm(idxs, desc=f'{name} ({arms}, dense MGE)'):
        data = dataset[i]
        sim = _feature_similarity(model, data)                      # one forward, both feature arms
        errs['nn'].append(_dense_mge(model, data, _nn_sparse_map(sim)))
        if with_hungarian:
            errs['hungarian'].append(_dense_mge(model, data, _hungarian_sparse_map(sim)))
        if with_diffusion:
            errs['diffusion'].append(_dense_mge(model, data, model.validate_single(data)))

    err = {k: np.concatenate(v) for k, v in errs.items() if v}
    summary = {'name': name, 'checkpoint': ckpt, 'n_pairs': len(idxs),
               'fps_metric': getattr(dataset, 'fps_metric', 'config'),
               'feat_source': getattr(model.densifier, 'feat_source', None),
               'nn_baseline': _summary(err['nn'])}                  # key kept for older result files
    for k in ('hungarian', 'diffusion'):
        if k in err:
            summary[k] = _summary(err[k])
    if 'diffusion' in err:
        summary['delta_MGE_nn_minus_diffusion'] = summary['nn_baseline']['dense_MGE'] - summary['diffusion']['dense_MGE']
        if 'hungarian' in err:                                      # the control: is the win just bijectivity?
            summary['delta_MGE_hungarian_minus_diffusion'] = summary['hungarian']['dense_MGE'] - summary['diffusion']['dense_MGE']

    out_dir = _benchmark_dir(ckpt, name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, 'nn_baseline_dense.npz'),
             nn_error=err['nn'],
             hungarian_error=err.get('hungarian', np.array([])),
             diffusion_error=err.get('diffusion', np.array([])))
    summary['out_dir'] = out_dir
    with open(os.path.join(out_dir, 'nn_baseline_dense.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print(s):
    print(f"\nexperiment      : {s['name']}")
    print(f"checkpoint      : {s['checkpoint']}")
    print(f"pairs / fps     : {s['n_pairs']} / {s['fps_metric']}   densifier feat_source: {s['feat_source']}")
    print(f"\n{'matcher':>17} {'dense MGE':>11} {'median':>9} {'p90':>9} {'gross>0.1':>11}")
    print('-' * 60)
    for key, label in (('nn_baseline', 'feature-NN'), ('hungarian', 'feature+Hungarian'),
                       ('diffusion', 'diffusion')):
        if key in s:
            a = s[key]
            print(f"{label:>17} {a['dense_MGE']:>11.4f} {a['median']:>9.4f} {a['p90']:>9.4f} "
                  f"{a['gross_gt_0.1']*100:>10.1f}%")
    if 'diffusion' in s:
        print('-' * 60)
        print(f"delta (NN - diffusion)        = {s['delta_MGE_nn_minus_diffusion']:+.4f}")
        if 'delta_MGE_hungarian_minus_diffusion' in s:
            # the honest number: NN lacks bijectivity, so only this delta isolates the denoiser
            d = s['delta_MGE_hungarian_minus_diffusion']
            verdict = ("denoiser beats plain assignment" if d > 0 else
                       "denoiser adds nothing over plain assignment")
            print(f"delta (Hungarian - diffusion) = {d:+.4f}  -> {verdict}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True, help='training config to load arch + test set from')
    p.add_argument('--checkpoint', default=None, help='checkpoint override (e.g. a FAUST model on a SCAPE config)')
    p.add_argument('--num-pairs', type=int, default=0, help='cap pairs (seeded subset); 0 = all')
    p.add_argument('--seed', type=int, default=0, help='seed for the --num-pairs subset')
    p.add_argument('--no-diffusion', action='store_true', help='feature arms only (skip the diffusion pass)')
    p.add_argument('--no-hungarian', action='store_true', help='skip the linear-assignment control arm')
    p.add_argument('--fps-metric', choices=('config', 'geodesic', 'euclidean'), default='config',
                   help='override the dataset FPS metric (default: whatever the config says)')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    args = p.parse_args()

    s = run(args.config, args.checkpoint, args.device, args.fps_metric,
            args.num_pairs, args.seed, not args.no_diffusion, not args.no_hungarian)
    _print(s)
    print(f"\nper-pair errors + JSON under: {s['out_dir']}/")


if __name__ == '__main__':
    main()
