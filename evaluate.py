import argparse
import json
import os

from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from train import _single_collate
from utils.logger import get_root_logger
from utils.options import load_yaml, resolve_experiment_paths


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
def build_opt(args):
    """Load the config for evaluation: force inference mode and point the model at
    the checkpoint to test (CLI ``--checkpoint`` or the experiment's ``final.pth``)."""
    opt = load_yaml(args.config)
    if args.name is not None:
        opt['name'] = args.name
    if args.device is not None:
        opt['device'] = args.device

    opt['is_train'] = False  # no optimizers/schedulers/losses for a pure eval pass

    resolve_experiment_paths(opt)
    ckpt = args.checkpoint or os.path.join(opt['path']['models'], 'final.pth')
    opt['path']['resume_state'] = ckpt
    opt['path']['resume'] = False  # net-only load; don't restore optimizer/epoch state
    return opt, ckpt


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained shape-matching model on the test set.')
    parser.add_argument('-c', '--config', required=True, help='path to the YAML config used for training')
    parser.add_argument('-n', '--name', default=None, help='override experiment name (subdir of experiments/)')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint to evaluate (default: experiments/<name>/models/final.pth)')
    parser.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    parser.add_argument('--num_workers', type=int, default=0, help='dataloader workers')
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(opt, ckpt, args):
    logger = get_root_logger()

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f'checkpoint not found: {ckpt}\nTrain first, or pass --checkpoint <path>.')

    test_set = build_dataset(opt['datasets']['test'])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             collate_fn=_single_collate, num_workers=args.num_workers)

    # match the encoder input dim to the actual per-vertex feature dim (as in train.py)
    opt['networks']['encoder']['in_dim'] = int(test_set[0]['first']['feat'].shape[-1])

    # the constructor loads `ckpt` (net-only, since is_train is False)
    model = build_model(opt)
    logger.info(f'Evaluating "{opt["name"]}" on {len(test_set)} test pairs '
                f'(checkpoint: {ckpt}, device: {model.device}).')

    results_dir = opt['path']['results']
    os.makedirs(results_dir, exist_ok=True)

    # update=False: pure evaluation, no best-model tracking. out_dir sends pck.png /
    # pck.npy to results/ (the same dir stats.json is written to).
    metrics = model.validation(test_loader, update=False, out_dir=results_dir)

    stats = {
        'name': opt['name'],
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
