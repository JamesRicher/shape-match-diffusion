import argparse

import torch
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.logger import get_root_logger
from utils.options import load_yaml, resolve_experiment_paths


# --------------------------------------------------------------------------- #
# options
# --------------------------------------------------------------------------- #
def build_opt(args):
    """Load the ``opt`` dict from a YAML config and apply CLI overrides.

    The YAML holds the layout the model package expects (see models/base_model.py):
    ``networks`` / ``train`` (optims, schedulers, losses) / ``val`` (metrics), plus a
    ``datasets`` block that ``train.py`` consumes and the model ignores. Experiment
    output paths are resolved from ``name`` under experiments/.
    """
    opt = load_yaml(args.config)

    # CLI overrides (only when provided)
    if args.name is not None:
        opt['name'] = args.name
    if args.epochs is not None:
        opt['train']['total_epochs'] = args.epochs
    if args.device is not None:
        opt['device'] = args.device

    # keep the cosine schedule length tied to the run length: T_max always follows
    # total_epochs (we step CosineAnnealingLR once per epoch), so the two can't drift.
    total_epochs = opt['train']['total_epochs']
    for sched_cfg in opt['train'].get('schedulers', {}).values():
        if sched_cfg.get('type') == 'CosineAnnealingLR':
            sched_cfg['T_max'] = total_epochs

    # resolve experiment output paths (models/ results/) from the name
    resolve_experiment_paths(opt)
    path = opt['path']
    path['resume_state'] = args.resume if args.resume is not None else path.get('resume_state')

    return opt


def parse_args():
    parser = argparse.ArgumentParser(description='Train a shape-matching model.')
    parser.add_argument('-c', '--config', required=True, help='path to a YAML config file')
    parser.add_argument('-n', '--name', default=None, help='override experiment name (subdir of experiments/)')
    parser.add_argument('-e', '--epochs', type=int, default=None, help='override number of training epochs')
    parser.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    parser.add_argument('--resume', default=None, help='path to a checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=0, help='dataloader workers')
    parser.add_argument('--debug', action='store_true', help='run a couple of iterations for a quick smoke test')
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def _single_collate(batch):
    """batch_size=1 collate that returns the sample untouched.

    The shape pairs hold variable-size and sparse tensors (operators), which the
    default collate cannot stack, so we train one pair at a time.
    """
    return batch[0]


def move_to_device(obj, device):
    """Recursively move tensors in a (possibly nested) dict to ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def build_dataloaders(opt, num_workers):
    train_set = build_dataset(opt['datasets']['train'])
    val_set = build_dataset(opt['datasets']['val'])
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              collate_fn=_single_collate, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            collate_fn=_single_collate, num_workers=num_workers)
    return train_set, train_loader, val_loader


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def train(opt, args):
    logger = get_root_logger()

    train_set, train_loader, val_loader = build_dataloaders(opt, args.num_workers)

    # match the encoder input dim to the actual per-vertex feature dim
    opt['networks']['encoder']['in_dim'] = int(train_set[0]['first']['feat'].shape[-1])

    model = build_model(opt)
    logger.info(f'Start training "{opt["name"]}" for {opt["train"]["total_epochs"]} epochs '
                f'on {len(train_set)} pairs (device: {model.device}).')

    log_freq = opt['train']['log_freq']
    for epoch in range(model.curr_epoch, opt['train']['total_epochs']):
        model.curr_epoch = epoch
        model.train()

        for i, data in enumerate(train_loader):
            model.curr_iter += 1
            data = move_to_device(data, model.device)

            model.feed_data(data)
            model.optimize_parameters()
            model.update_model_per_iteration()

            if model.curr_iter % log_freq == 0:
                losses = model.get_loss_metrics()
                loss_str = ' '.join(f'{k}:{v.item():.4f}' for k, v in losses.items())
                lr = model.get_current_learning_rate()[0]
                logger.info(f'[epoch {epoch:03d}][iter {model.curr_iter:06d}] lr:{lr:.2e} {loss_str}')

            if args.debug and i >= 2:
                break

        model.update_model_per_epoch()

        # end-of-epoch validation + checkpoint
        model.validation(val_loader, update=True)
        model.save_model()

        if args.debug:
            break

    # save the best-by-validation weights as the final model
    model.save_model(net_only=True, best=True)
    # dump a self-describing summary (config + network stats) to the experiment root
    info_path = model.save_experiment_info()
    logger.info(f'Training done. Best avg geodesic error: {model.best_metric}')
    logger.info(f'Wrote experiment info to {info_path}')


def main():
    args = parse_args()
    opt = build_opt(args)
    train(opt, args)


if __name__ == '__main__':
    main()
