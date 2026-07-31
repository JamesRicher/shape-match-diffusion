import argparse
import json
import os
import re

import numpy as np
import yaml
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from models.base_model import to_numpy
from train import _single_collate, _RestoringLoader, autofill_feat_dim
from utils.data_utils import sqrt_surface_area
from utils.logger import get_root_logger
from utils.options import _YamlLoader, load_yaml, resolve_experiment_paths
from utils.texture_util import (render_color_transfer_figure,
                                render_texture_transfer_figure)


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
def build_opt(args):
    """Load the config for evaluation: force inference mode and point the model at
    the checkpoint to test (CLI ``--checkpoint`` or the experiment's ``final.pth``)."""
    opt = load_yaml(args.config)
    for override in (args.set or []):
        apply_override(opt, override)
    if args.device is not None:
        opt['device'] = args.device

    opt['is_train'] = False  # no optimizers/schedulers/losses for a pure eval pass

    # NOTE: -n/--name is NOT applied here. It names the results/ SUBDIR (leaf) only (see evaluate);
    # the experiment paths -- and thus the results ROOT and the default checkpoint -- stay derived
    # from the config's own name, so -n picks the leaf, never the model directory.
    resolve_experiment_paths(opt)
    ckpt = args.checkpoint or os.path.join(opt['path']['models'], 'final.pth')
    opt['path']['resume_state'] = ckpt
    opt['path']['resume'] = False  # net-only load; don't restore optimizer/epoch state
    return opt, ckpt


def apply_override(opt, spec):
    """Apply one ``dotted.key=value`` override to the loaded config in place, so eval knobs can
    be swept from the CLI without editing/copying the YAML (e.g. ``densifier.k_fm=160``).

    The value is parsed as YAML so ``160`` -> int, ``0.5`` -> float, ``true`` -> bool, ``1e-3``
    -> float (via the strict loader), and anything else stays a string. Intermediate dict keys
    are created as needed; pair with ``--eval_tag`` so swept runs don't overwrite each other."""
    if '=' not in spec:
        raise ValueError(f"--set expects 'dotted.key=value', got {spec!r}")
    key, raw = spec.split('=', 1)
    node = opt
    parts = key.split('.')
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.load(raw, Loader=_YamlLoader)


def eval_tag_for(opt, override=None):
    """Directory name for one evaluation, so evaluating a model on several datasets
    keeps each result set separate under ``results/<tag>/`` instead of overwriting.

    Defaults to the test dataset's ``name`` (plus phase when not ``test``); pass
    ``override`` (CLI ``--eval_tag``) to name it explicitly, e.g. to compare two eval
    settings on the same dataset."""
    if override:
        tag = override
    else:
        test = opt['datasets']['test']
        tag = str(test.get('name', 'test'))
        phase = test.get('phase')
        if phase and phase != 'test':
            tag = f'{tag}_{phase}'
    # keep it a safe single path segment
    return re.sub(r'[^A-Za-z0-9._-]+', '_', tag).strip('_') or 'test'


def results_root_for(opt, args, ckpt):
    """The ``results/`` root this eval writes under. It must belong to the TRAINED MODEL, so that
    one model directory owns all of its evaluations. The old code rooted results at the config's
    experiment dir (opt['path']['results']), so a cross-dataset run -- target-dataset config plus
    ``--checkpoint <other model>`` -- leaked the result into the eval dataset's folder, orphaning it
    from the model that produced it. The root is decided purely by the checkpoint:

      * a ``--checkpoint``  -> <checkpoint's experiment root>/results  (the model owns it)
      * else                -> opt['path']['results']  (config's own model == checkpoint)

    ``-n/--name`` deliberately does NOT enter here: it names the results/ LEAF (the eval_tag), not
    the root, so the model dir always owns the output. The checkpoint root is two levels up from
    ``.../models/<file>.pth``; a non-standard layout falls back to the config's dir with a warning.
    Under whichever root, the leaf is the ``eval_tag`` (-n / --eval_tag / dataset name) so evaluating
    one model on several datasets never clobbers."""
    if args.checkpoint is not None:
        ckpt_abs = os.path.abspath(ckpt)
        if os.path.basename(os.path.dirname(ckpt_abs)) == 'models':
            return os.path.join(os.path.dirname(os.path.dirname(ckpt_abs)), 'results')
        get_root_logger().warning(
            f'--checkpoint {ckpt} is not in a <experiment>/models/ layout; cannot infer its model '
            f"directory, so results go under the config's dir.")
    return opt['path']['results']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained shape-matching model on the test set.')
    parser.add_argument('-c', '--config', required=True, help='path to the YAML config used for training')
    parser.add_argument('-n', '--name', default=None,
                        help='name of the results/ subdir to write into (default: the test dataset '
                             "name); the parent always follows the checkpoint's model, giving "
                             '<model>/results/<name>. To evaluate a different model, use --checkpoint.')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint to evaluate (default: experiments/<config-name>/models/'
                             'final.pth); results file under THIS checkpoint\'s own experiment dir')
    parser.add_argument('--eval_tag', default=None,
                        help='subdir of results/ to write this evaluation into '
                             '(default: the test dataset name); lets one model be '
                             'evaluated on several datasets without overwriting')
    parser.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    parser.add_argument('--set', action='append', metavar='KEY=VALUE', default=None,
                        help='override a config value by dotted key, repeatable '
                             '(e.g. --set densifier.k_fm=160); pair with --eval_tag to avoid '
                             'overwriting the default run')
    parser.add_argument('--no_sparse', action='store_true',
                        help='skip the [2/2] bijective sparse-stats pass (dense MGE only)')
    parser.add_argument('--num_workers', type=int, default=0, help='dataloader workers')
    parser.add_argument('--num_qual', type=int, default=10,
                        help='number of random test pairs to render texture-transfer '
                             'figures for (results/qual/); 0 disables')
    parser.add_argument('--qual_seed', type=int, default=0,
                        help='RNG seed for picking the qualitative pairs')
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# qualitative texture-transfer figures
# --------------------------------------------------------------------------- #
def generate_qualitative(model, test_set, out_dir, num_pairs=10, seed=0):
    """Render texture-transfer figures (source / hard p2p / smoothed) for a few
    random test pairs into ``out_dir`` for quick visual inspection."""
    logger = get_root_logger()
    if num_pairs <= 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    model.eval()  # validation() flips back to train mode when it finishes

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_set), size=min(num_pairs, len(test_set)), replace=False)
    flip_up = bool(getattr(test_set, 'flip_up', False))

    # texture transfer needs a dense whole-shape p2p; a sparse matcher's validate_single only
    # returns the sparse FPS-point map, so lift it through the densifier first.
    use_dense = getattr(model, 'densifier', None) is not None

    for idx in indices:
        data = test_set[int(idx)]
        data_x, data_y = data['first'], data['second']
        p2p = to_numpy(model.densify_single(data) if use_dense
                       else model.validate_single(data))  # Y -> X, [Vy]

        # per-pair mean geodesic error for the title (same normalization as validation)
        title_err = ''
        if 'geo_error' in model.metrics and 'dist' in data_x:
            geo_err = model.metrics['geo_error'](
                to_numpy(data_x['dist']), to_numpy(data_x['corr']),
                to_numpy(data_y['corr']), p2p, return_mean=False)
            if 'mass' in data_x:
                geo_err = geo_err / to_numpy(sqrt_surface_area(data_x['mass']))
            title_err = f'  |  mean geo err: {geo_err.mean():.4f}'

        name_x = data_x.get('name', f'shape_{idx}_x')
        name_y = data_y.get('name', f'shape_{idx}_y')
        common = dict(
            evecs_x=to_numpy(data_x['evecs']) if 'evecs' in data_x else None,
            evecs_y=to_numpy(data_y['evecs']) if 'evecs' in data_y else None,
            evecs_trans_x=to_numpy(data_x['evecs_trans']) if 'evecs_trans' in data_x else None,
            evecs_trans_y=to_numpy(data_y['evecs_trans']) if 'evecs_trans' in data_y else None,
            flip_up=flip_up,
            title=f'{name_x} → {name_y}{title_err}')
        geom = (to_numpy(data_x['verts']), to_numpy(data_x['faces']),
                to_numpy(data_y['verts']), to_numpy(data_y['faces']), p2p)

        # texture transfer + position-coded vertex-color correspondence
        for kind, render in (('texture', render_texture_transfer_figure),
                             ('color', render_color_transfer_figure)):
            out_file = os.path.join(out_dir, f'{name_x}-{name_y}_{kind}.png')
            render(*geom, **common, out_file=out_file)
            logger.info(f'Wrote qualitative figure: {out_file}')


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(opt, ckpt, args):
    logger = get_root_logger()

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f'checkpoint not found: {ckpt}\nTrain first, or pass --checkpoint <path>.')

    test_set = build_dataset(opt['datasets']['test'])
    # Sparse matcher (has independent_fps): the two eval stats live in different sampling
    # regimes, so we report them in two passes (see below). Non-sparse datasets lack the
    # flag and take the plain single pass in the else branch.
    sparse_matcher = hasattr(test_set, 'independent_fps')
    # _single_collate ships sparse operators (gradX/gradY) as IPC-safe tuples; _RestoringLoader
    # rebuilds them in the main process (as in train.py). Required for the DiffusionNet extractor,
    # which reads those operators; harmless otherwise.
    test_loader = _RestoringLoader(DataLoader(test_set, batch_size=1, shuffle=False,
                             collate_fn=_single_collate, num_workers=args.num_workers))

    # match each network's input dim to the actual per-vertex feature dim (as in train.py)
    autofill_feat_dim(opt, int(test_set[0]['first']['feat'].shape[-1]))

    # the constructor loads `ckpt` (net-only, since is_train is False)
    model = build_model(opt)
    sparse_matcher = sparse_matcher and hasattr(model, 'report_sparse')

    # Results are owned by the trained model's own experiment dir (results_root_for), keyed by the
    # test dataset (eval_tag). So evaluating THIS model on another dataset adds a sibling subdir,
    # and evaluating ANOTHER model here (cross-dataset, via --checkpoint) lands under that model's
    # dir -- not this config's -- so every result sits beside the checkpoint that produced it.
    # leaf name precedence: --eval_tag, then -n/--name, then the test dataset name. The root is the
    # model's dir, so -n gives <model>/results/<name>.
    eval_tag = eval_tag_for(opt, args.eval_tag or args.name)
    results_root = results_root_for(opt, args, ckpt)
    model_name = os.path.basename(os.path.dirname(results_root))    # the model dir that owns this
    results_dir = os.path.join(results_root, eval_tag)
    os.makedirs(results_dir, exist_ok=True)

    logger.info(f'Evaluating model "{model_name}" on {len(test_set)} test pairs -> results/{eval_tag}'
                f'  (checkpoint: {ckpt}, device: {model.device}).')

    if sparse_matcher:
        # Two passes, because the sparse and dense stats need different sparse sampling:
        #   1) dense whole-shape MGE under honest independent FPS (each shape's FPS points
        #      chosen on its own geometry, no GT) -- the reporting metric.
        #   2) sparse FPS-point avg_error + acc under GT-consistent bijective FPS, where
        #      sparse Y point j is built to match sparse X point j (identity gt_perm), so
        #      the diagonal error/accuracy are defined. This uses GT in point selection and
        #      is a dev diagnostic, not an honest number -- keys are prefixed 'sparse_'.
        metrics = {}

        if getattr(model, 'densifier', None) is not None:
            test_set.independent_fps = True
            model.report_sparse, model.report_dense = False, True
            logger.info('[1/2] dense MGE (honest independent FPS)')
            metrics.update(model.validation(test_loader, out_dir=results_dir))
        else:
            logger.info('[1/2] dense MGE skipped (no densifier configured)')

        if getattr(args, 'no_sparse', False):
            logger.info('[2/2] sparse stats skipped (--no_sparse)')
        else:
            test_set.independent_fps = False
            model.report_sparse, model.report_dense = True, False
            logger.info('[2/2] sparse stats (bijective FPS)')
            sparse_metrics = model.validation(test_loader, out_dir=None)
            for k, v in sparse_metrics.items():
                # avg_error/acc are the sparse stats; keep them explicit about the regime.
                metrics[f'sparse_{k}' if k in ('avg_error', 'acc') else k] = v
    else:
        # pure evaluation pass. out_dir sends pck.png / pck.npy to results/ (the same
        # dir stats.json is written to).
        metrics = model.validation(test_loader, out_dir=results_dir)

    # qualitative texture-transfer figures on a few random pairs (results/qual/)
    generate_qualitative(model, test_set, os.path.join(results_dir, 'qual'),
                         num_pairs=args.num_qual, seed=args.qual_seed)

    stats = {
        'name': model_name,            # the model experiment dir that owns this result
        'eval_tag': eval_tag,          # the results/ leaf (-n / --eval_tag / dataset name)
        'checkpoint': ckpt,
        'dataset': opt['datasets']['test'],
        'num_test_pairs': len(test_set),
        **metrics,
    }
    with open(os.path.join(results_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f'Test result: {metrics}')
    logger.info(f'Wrote results to {results_dir}')
    return stats


def main():
    args = parse_args()
    opt, ckpt = build_opt(args)
    evaluate(opt, ckpt, args)


if __name__ == '__main__':
    main()
