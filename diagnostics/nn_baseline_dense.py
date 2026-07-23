"""ISOLATED, DELETE-SAFE DIAGNOSTIC -- does the diffusion do anything beyond feature-NN?

Everything this needs lives under diagnostics/; every output goes to diagnostics/results/.
It modifies no tracked file and registers nothing. Delete diagnostics/ to remove it entirely.

WHAT IT ANSWERS
---------------
The worry: "all the information is already in the features; the diffusion denoiser adds
nothing." This runs the SAME forward pipeline but BYPASSES the diffusion model. For each test
pair it builds the sparse map two ways and pushes BOTH through the identical densifier + dense
geodesic metric, so the only thing that differs is the sparse matcher:

  * nn        -- honest-FPS sparse points matched purely by feature cosine-similarity
                 (argmax), on the exact trained features the denoiser consumes (_sparse_inputs).
  * diffusion -- the model's validate_single (sample + Hungarian), i.e. the real pipeline.

Both use independent (honest) FPS -- the regime the reported dense MGE lives in -- the same
densifier (feat_source and all), and metrics.calculate_geodesic_error exactly as evaluate.py's
validation() does. So:

  nn dense MGE ~ diffusion dense MGE  -> the diffusion adds nothing the features + densifier
      don't already give; the denoiser is dead weight for the final map.
  nn dense MGE >> diffusion dense MGE -> the diffusion is doing real relational work NN can't.

USAGE
-----
  python -m diagnostics.nn_baseline_dense -c configs/joint_diffusionnet/faust_diffusionnet_512_FMD.yaml
  # cross-dataset (FAUST model on SCAPE data), like evaluate.py -c <scape cfg> -n <faust name>:
  python -m diagnostics.nn_baseline_dense -c configs/joint_diffusionnet/scape_diffusionnet_512_FMD.yaml \
      --checkpoint experiments/faust_diffusionnet_512_FMD/models/final.pth

Optional: --num-pairs N (0 = all), --no-diffusion (NN only, faster), --fps-metric
{config,geodesic,euclidean}, --device cuda/cpu, --seed (for the --num-pairs subset).
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import build_dataset
from metrics import build_metric
from models import build_model
from models.base_model import to_numpy
from train import autofill_feat_dim
from utils.options import load_yaml, resolve_experiment_paths

_OUT_ROOT = os.path.join(os.path.dirname(__file__), 'results')
calculate_geodesic_error = build_metric({"type": "calculate_geodesic_error"})


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


def _feature_nn_sparse_map(model, data):
    """Sparse Y->X map from pure feature cosine-NN over the honest-FPS sparse points, using the
    exact per-point features the denoiser consumes. Returns (n_y,) LongTensor of X sparse indices."""
    F_x, F_y, _, _, _ = model._sparse_inputs(data)                  # (1, n, d) each, trained feats
    fx = F.normalize(F_x.squeeze(0), dim=-1)
    fy = F.normalize(F_y.squeeze(0), dim=-1)
    return (fy @ fx.T).argmax(dim=1)                                # (n_y,) each Y point's nearest X


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
def run(config_path, checkpoint, device, fps_metric, num_pairs, seed, with_diffusion):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, fps_metric)
    name = opt['name']
    if not num_pairs:
        idxs = list(range(len(dataset)))
    else:
        n = min(num_pairs, len(dataset))
        idxs = sorted(np.random.default_rng(seed).choice(len(dataset), size=n, replace=False).tolist())

    nn_errs, diff_errs = [], []
    for i in tqdm(idxs, desc=f'{name} (NN vs diffusion, dense MGE)'):
        data = dataset[i]
        nn_errs.append(_dense_mge(model, data, _feature_nn_sparse_map(model, data)))
        if with_diffusion:
            diff_errs.append(_dense_mge(model, data, model.validate_single(data)))

    nn_err = np.concatenate(nn_errs)
    summary = {'name': name, 'checkpoint': ckpt, 'n_pairs': len(idxs),
               'fps_metric': getattr(dataset, 'fps_metric', 'config'),
               'feat_source': getattr(model.densifier, 'feat_source', None),
               'nn_baseline': _summary(nn_err)}
    if with_diffusion:
        diff_err = np.concatenate(diff_errs)
        summary['diffusion'] = _summary(diff_err)
        summary['delta_MGE_nn_minus_diffusion'] = summary['nn_baseline']['dense_MGE'] - summary['diffusion']['dense_MGE']

    out_dir = os.path.join(_OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, 'nn_baseline_dense.npz'),
             nn_error=nn_err, diffusion_error=(np.concatenate(diff_errs) if with_diffusion else np.array([])))
    with open(os.path.join(out_dir, 'nn_baseline_dense.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print(s):
    print(f"\nexperiment      : {s['name']}")
    print(f"checkpoint      : {s['checkpoint']}")
    print(f"pairs / fps     : {s['n_pairs']} / {s['fps_metric']}   densifier feat_source: {s['feat_source']}")
    print(f"\n{'matcher':>12} {'dense MGE':>11} {'median':>9} {'p90':>9} {'gross>0.1':>11}")
    print('-' * 56)
    nn = s['nn_baseline']
    print(f"{'feature-NN':>12} {nn['dense_MGE']:>11.4f} {nn['median']:>9.4f} {nn['p90']:>9.4f} {nn['gross_gt_0.1']*100:>10.1f}%")
    if 'diffusion' in s:
        d = s['diffusion']
        print(f"{'diffusion':>12} {d['dense_MGE']:>11.4f} {d['median']:>9.4f} {d['p90']:>9.4f} {d['gross_gt_0.1']*100:>10.1f}%")
        print('-' * 56)
        delta = s['delta_MGE_nn_minus_diffusion']
        verdict = ("diffusion helps" if delta > 0 else "diffusion no better / worse")
        print(f"delta (NN - diffusion) = {delta:+.4f}  -> {verdict}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True, help='training config to load arch + test set from')
    p.add_argument('--checkpoint', default=None, help='checkpoint override (e.g. a FAUST model on a SCAPE config)')
    p.add_argument('--num-pairs', type=int, default=0, help='cap pairs (seeded subset); 0 = all')
    p.add_argument('--seed', type=int, default=0, help='seed for the --num-pairs subset')
    p.add_argument('--no-diffusion', action='store_true', help='NN baseline only (skip the diffusion pass)')
    p.add_argument('--fps-metric', choices=('config', 'geodesic', 'euclidean'), default='config',
                   help='override the dataset FPS metric (default: whatever the config says)')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    args = p.parse_args()

    s = run(args.config, args.checkpoint, args.device, args.fps_metric,
            args.num_pairs, args.seed, not args.no_diffusion)
    _print(s)
    print(f"\nper-pair errors + JSON under: {os.path.join(_OUT_ROOT, s['name'])}/")


if __name__ == '__main__':
    main()
