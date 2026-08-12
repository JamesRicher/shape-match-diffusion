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

All three are scored on the SAME three numbers evaluate.py reports for the full pipeline:

  * dense MGE            -- honest independent FPS, densified, metrics.calculate_geodesic_error.
  * sparse independent   -- the same independent-FPS sparse map re-scored AT the sparse points
                            against real template GT (diagnostics.sparse_independent_error's
                            nearest-landmark rule). The regime dense MGE lives in, minus the
                            densifier, so it separates matcher error from lift error.
  * sparse bijective     -- a SECOND pass with dataset.independent_fps = False, where sparse Y
                            point j is built to match sparse X point j, giving the identity-GT
                            acc + avg_error that stats.json calls sparse_acc / sparse_avg_error.
                            Uses GT in point selection, so it is a dev diagnostic, not an honest
                            number (it structurally cannot show a symmetry flip). --no-bijective
                            skips this pass; it costs one extra forward per pair.

The dense arms use the same densifier (feat_source and all) and metrics.calculate_geodesic_error
exactly as evaluate.py's validation() does. So:

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
--no-bijective (skip the bijective-FPS sparse pass), --fps-metric
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
from evaluate import apply_override
from train import autofill_feat_dim
from utils.options import load_yaml, resolve_experiment_paths
from diagnostics.sparse_independent_error import _sparse_independent_error

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


# --------------------------------------------------------------------------- #
# Eval-time feature degradation
# --------------------------------------------------------------------------- #
# The worry is "the features do all the work". These knobs corrupt the extractor's per-vertex
# field at a SINGLE point -- DiffusionNetExtractor._per_vertex -- through which BOTH the denoiser's
# sparse conditioning (_sparse_inputs -> extract) and the densifier's global data term
# (_densify_context -> extract_dense) flow. So every arm (nn, hungarian, diffusion) sees the exact
# same corrupted features, which is the fairness constraint: the only variable is the matcher, and
# we watch how each arm's dense MGE degrades as the feature signal is dialed down. This is the
# "graceful degradation" regime -- it simulates the cross-dataset unreliability on the
# in-distribution set with a continuous knob, and the prize is the crossover strength (if any) where
# the diffusion arm starts beating plain Hungarian.

def _gaussian_noise(feats, sigma):
    """Additive isotropic Gaussian noise scaled to the feature RMS, so `sigma` is an interpretable
    inverse-SNR knob (sigma=1 => noise std == signal std) that is comparable across datasets and
    feature scales. Global (not per-channel) scale keeps relative channel magnitudes intact."""
    scale = feats.detach().std()
    return feats + sigma * scale * torch.randn_like(feats)


def _geodesic_blur(feats, shape, t):
    """Heat-kernel low-pass of the feature field over the manifold -- geodesically-nearby vertices
    get averaged together, creating the LOCAL ambiguity that per-point matching cannot resolve but a
    relational prior might (the actual cross-dataset failure mode, cf. limb garbling). Implemented
    spectrally: project onto the LBO eigenbasis (mass-weighted), damp mode k by exp(-t * lambda_k),
    reconstruct. Eigenvalues are normalised by their mean so the diffusion time `t` is O(1) and
    comparable across meshes of different scale."""
    dev = feats.device
    evecs = shape['evecs'].to(dev).float()                          # (V, K)
    evals = shape['evals'].to(dev).float()                          # (K,)
    mass = shape['mass'].to(dev).float()                            # (V,)
    k = min(evecs.shape[1], evals.shape[0])
    evecs, evals = evecs[:, :k], evals[:k]
    scale = evals[evals > 0].mean().clamp_min(1e-8)                 # scale-invariant diffusion time
    decay = torch.exp(-t * evals / scale)                          # (K,)
    coeff = evecs.t() @ (mass[:, None] * feats)                    # (K, d) mass-weighted projection
    return evecs @ (decay[:, None] * coeff)                        # (V, d) smoothed field


def _install_feature_degradation(model, mode, strength):
    """Monkeypatch the extractor's _per_vertex so every downstream consumer sees the corrupted
    field. Returns a short tag for output filenames. No-op (returns '') when mode is None."""
    if mode is None:
        return ''
    ext = model.networks.get('extractor')
    if ext is None or not getattr(ext, 'needs_operators', False):
        raise ValueError("feature degradation needs a DiffusionNet extractor (needs_operators); "
                         "this config exposes no per-vertex feature field to corrupt")
    orig = ext._per_vertex
    if mode == 'noise':
        def wrapped(shape, _orig=orig): return _gaussian_noise(_orig(shape), strength)
    elif mode == 'blur':
        def wrapped(shape, _orig=orig): return _geodesic_blur(_orig(shape), shape, strength)
    else:
        raise ValueError(f"unknown degradation mode {mode!r}")
    ext._per_vertex = wrapped
    return f'{mode}{strength:g}'


def _build(config_path, checkpoint, device, fps_metric, split='test', exclude_self=False,
           overrides=None):
    """Load a trained checkpoint + a dataset split like evaluate.py, forced into the honest
    independent-FPS regime with dense reporting on (the densifier is required). `split` selects
    datasets.<split> -- use 'train' as a held-out set for tuning selector hyperparameters, so the
    reported 'test' numbers stay uncontaminated (ret_evecs is forced on so the densifier works).

    overrides: optional list of evaluate.py-style 'dotted.key=value' strings, applied to the
    config before anything is built, so eval-time knobs (densifier.k_fm, ...) can be swept from a
    caller's CLI without copying YAML. Defaulted, so existing callers are unaffected."""
    opt = load_yaml(config_path)
    for spec in (overrides or []):
        apply_override(opt, spec)
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
    probe = dataset[0]['first']  # extractor configs ship no frozen 'feat'; autofill is then a no-op
    autofill_feat_dim(opt, int(probe['feat'].shape[-1]) if 'feat' in probe else 0)
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


def _sparse_gt(data):
    """Per-pair arguments of the independent-FPS sparse GT rule, hoisted out of the arm loop
    (they depend only on the pair, not on the matcher). See sparse_independent_error."""
    x, y = data['first'], data['second']
    return (to_numpy(x['sparse']['idx']).astype(np.int64),
            to_numpy(y['sparse']['idx']).astype(np.int64),
            to_numpy(x['dist']), to_numpy(y['dist']),
            to_numpy(x['corr']).astype(np.int64),
            to_numpy(y['corr']).astype(np.int64))


def _bijective_sparse(data, sparse_p2p):
    """(acc, per-point error) under bijective FPS, exactly as the model's validation() computes
    the reported sparse_acc / sparse_avg_error: sparse Y point j matches sparse X point j, so
    the GT is the identity diagonal and the error is read off the sparse geodesic submatrix."""
    D_x = to_numpy(data['first']['sparse']['dist'])                 # (n, n) geodesic on X
    p2p = to_numpy(sparse_p2p).astype(np.int64)
    rows = np.arange(len(p2p))
    return float((p2p == rows).mean()), D_x[rows, p2p]


def _summary(err):
    err = np.ravel(err)
    return {'dense_MGE': float(err.mean()),
            'median': float(np.median(err)),
            'p90': float(np.percentile(err, 90)),
            'gross_gt_0.1': float(np.mean(err > 0.1))}


def _sparse_summary(err):
    """Independent-FPS sparse error, at the sparse points (the honest sparse number)."""
    err = np.ravel(err)
    return {'sparse_MGE': float(err.mean()),
            'median': float(np.median(err)),
            'p90': float(np.percentile(err, 90)),
            'gross_gt_0.1': float(np.mean(err > 0.1))}


def _arm_maps(model, data, with_hungarian, with_diffusion):
    """The sparse Y->X map of every enabled arm for one pair, from a single feature forward."""
    sim = _feature_similarity(model, data)                          # one forward, both feature arms
    maps = {'nn': _nn_sparse_map(sim)}
    if with_hungarian:
        maps['hungarian'] = _hungarian_sparse_map(sim)
    if with_diffusion:
        maps['diffusion'] = model.validate_single(data)
    return maps


@torch.no_grad()
def run(config_path, checkpoint, device, fps_metric, num_pairs, seed, with_diffusion, with_hungarian,
        degrade_mode=None, degrade_strength=0.0, overrides=None, tag=None, with_bijective=True):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, fps_metric, overrides=overrides)
    name = opt['name']
    torch.manual_seed(seed)                                          # reproducible noise draws
    deg_tag = _install_feature_degradation(model, degrade_mode, degrade_strength)
    if not num_pairs:
        idxs = list(range(len(dataset)))
    else:
        n = min(num_pairs, len(dataset))
        idxs = sorted(np.random.default_rng(seed).choice(len(dataset), size=n, replace=False).tolist())

    errs = {'nn': [], 'hungarian': [], 'diffusion': []}             # dense, per vertex
    sp_errs = {'nn': [], 'hungarian': [], 'diffusion': []}          # sparse, independent FPS
    arms = ' vs '.join(['nn'] + ['hungarian'] * with_hungarian + ['diffusion'] * with_diffusion)
    for i in tqdm(idxs, desc=f'{name} ({arms}, independent FPS)'):
        data = dataset[i]
        gt = _sparse_gt(data)                                       # matcher-independent, hoisted
        for k, p2p in _arm_maps(model, data, with_hungarian, with_diffusion).items():
            errs[k].append(_dense_mge(model, data, p2p))
            sp_errs[k].append(_sparse_independent_error(to_numpy(p2p).astype(np.int64), *gt))

    # second pass: the bijective-FPS regime, where the identity GT makes acc defined. Same model
    # (degradation still installed), same pairs -- only the sparse sampling changes.
    bij_acc = {k: [] for k in errs}
    bij_err = {k: [] for k in errs}
    if with_bijective:
        dataset.independent_fps = False
        try:
            for i in tqdm(idxs, desc=f'{name} ({arms}, bijective FPS sparse)'):
                data = dataset[i]
                for k, p2p in _arm_maps(model, data, with_hungarian, with_diffusion).items():
                    acc, e = _bijective_sparse(data, p2p)
                    bij_acc[k].append(acc)
                    bij_err[k].append(e)
        finally:
            dataset.independent_fps = True                          # leave the honest regime set

    err = {k: np.concatenate(v) for k, v in errs.items() if v}
    sp_err = {k: np.concatenate(v) for k, v in sp_errs.items() if v}
    summary = {'name': name, 'checkpoint': ckpt, 'n_pairs': len(idxs),
               'fps_metric': getattr(dataset, 'fps_metric', 'config'),
               'feat_source': getattr(model.densifier, 'feat_source', None),
               'overrides': list(overrides or []),
               'degradation': (None if degrade_mode is None
                               else {'mode': degrade_mode, 'strength': degrade_strength}),
               'nn_baseline': _summary(err['nn'])}                  # key kept for older result files
    for k in ('hungarian', 'diffusion'):
        if k in err:
            summary[k] = _summary(err[k])
    # sparse stats nest inside each arm, so the pre-existing dense keys stay where they were
    for k, key in (('nn', 'nn_baseline'), ('hungarian', 'hungarian'), ('diffusion', 'diffusion')):
        if key not in summary:
            continue
        summary[key]['sparse_independent'] = _sparse_summary(sp_err[k])
        if bij_err[k]:
            summary[key]['sparse_bijective'] = {
                'acc': float(np.mean(bij_acc[k])),
                'avg_error': float(np.concatenate(bij_err[k]).mean())}
    if 'diffusion' in err:
        summary['delta_MGE_nn_minus_diffusion'] = summary['nn_baseline']['dense_MGE'] - summary['diffusion']['dense_MGE']
        if 'hungarian' in err:                                      # the control: is the win just bijectivity?
            summary['delta_MGE_hungarian_minus_diffusion'] = summary['hungarian']['dense_MGE'] - summary['diffusion']['dense_MGE']

    out_dir = _benchmark_dir(ckpt, name)
    os.makedirs(out_dir, exist_ok=True)
    # keep clean/degraded/overridden runs un-clobbered: the out dir is keyed by the eval config
    # name alone, so two runs that differ only by --set (e.g. DT4D intra vs inter) need --tag.
    stem = ('nn_baseline_dense' + (f'__{tag}' if tag else '')
            + (f'__{deg_tag}' if deg_tag else ''))
    np.savez(os.path.join(out_dir, f'{stem}.npz'),
             nn_error=err['nn'],
             hungarian_error=err.get('hungarian', np.array([])),
             diffusion_error=err.get('diffusion', np.array([])),
             nn_sparse_error=sp_err['nn'],
             hungarian_sparse_error=sp_err.get('hungarian', np.array([])),
             diffusion_sparse_error=sp_err.get('diffusion', np.array([])))
    summary['out_dir'] = out_dir
    with open(os.path.join(out_dir, f'{stem}.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print(s):
    print(f"\nexperiment      : {s['name']}")
    print(f"checkpoint      : {s['checkpoint']}")
    print(f"pairs / fps     : {s['n_pairs']} / {s['fps_metric']}   densifier feat_source: {s['feat_source']}")
    if s.get('degradation'):
        d = s['degradation']
        print(f"feature degrade : {d['mode']} @ strength {d['strength']:g}  (all arms, extractor output)")
    arms = [(k, l) for k, l in (('nn_baseline', 'feature-NN'), ('hungarian', 'feature+Hungarian'),
                                ('diffusion', 'diffusion')) if k in s]
    print(f"\n{'matcher':>17} {'dense MGE':>11} {'median':>9} {'p90':>9} {'gross>0.1':>11}")
    print('-' * 60)
    for key, label in arms:
        a = s[key]
        print(f"{label:>17} {a['dense_MGE']:>11.4f} {a['median']:>9.4f} {a['p90']:>9.4f} "
              f"{a['gross_gt_0.1']*100:>10.1f}%")

    # sparse stats: the same maps scored before the densifier lift
    print(f"\n{'matcher':>17} {'sparse MGE':>11} {'median':>9} {'p90':>9} {'gross>0.1':>11}"
          f" | {'bij acc':>8} {'bij err':>9}")
    print('-' * 82)
    for key, label in arms:
        a = s[key].get('sparse_independent')
        if a is None:
            continue
        b = s[key].get('sparse_bijective')
        bij = (f" | {b['acc']:>8.3f} {b['avg_error']:>9.4f}" if b else f" | {'--':>8} {'--':>9}")
        print(f"{label:>17} {a['sparse_MGE']:>11.4f} {a['median']:>9.4f} {a['p90']:>9.4f} "
              f"{a['gross_gt_0.1']*100:>10.1f}%{bij}")
    print("(sparse MGE = independent FPS, real GT, at the sparse points -- honest. bij acc/err = "
          "bijective\n FPS, identity GT -- dev diagnostic; it cannot show a symmetry flip.)")
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
    p.add_argument('--no-bijective', action='store_true',
                   help='skip the bijective-FPS sparse pass (sparse acc); saves one forward per pair')
    p.add_argument('--fps-metric', choices=('config', 'geodesic', 'euclidean'), default='config',
                   help='override the dataset FPS metric (default: whatever the config says)')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    p.add_argument('--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
                   help="evaluate.py-style config override, repeatable (e.g. "
                        "--set datasets.test.inter_class=false)")
    p.add_argument('--tag', default=None,
                   help='suffix for the output filenames; use it whenever --set changes the eval '
                        'set, since the output dir is keyed by config name alone')
    deg = p.add_mutually_exclusive_group()
    deg.add_argument('--feature-noise', type=float, default=None, metavar='SIGMA',
                     help='corrupt extractor features (all arms) with additive Gaussian noise; '
                          'SIGMA is noise std / feature std (inverse SNR). Sweep e.g. 0.25 0.5 1 2')
    deg.add_argument('--feature-blur', type=float, default=None, metavar='T',
                     help='corrupt extractor features (all arms) with heat-kernel geodesic blur; '
                          'T is scale-normalised diffusion time (larger = more local smoothing). '
                          'Sweep e.g. 0.05 0.1 0.2 0.5')
    args = p.parse_args()

    if args.feature_noise is not None:
        degrade_mode, degrade_strength = 'noise', args.feature_noise
    elif args.feature_blur is not None:
        degrade_mode, degrade_strength = 'blur', args.feature_blur
    else:
        degrade_mode, degrade_strength = None, 0.0

    s = run(args.config, args.checkpoint, args.device, args.fps_metric,
            args.num_pairs, args.seed, not args.no_diffusion, not args.no_hungarian,
            degrade_mode, degrade_strength, args.overrides, args.tag,
            with_bijective=not args.no_bijective)
    _print(s)
    print(f"\nper-pair errors + JSON under: {s['out_dir']}/")


if __name__ == '__main__':
    main()
