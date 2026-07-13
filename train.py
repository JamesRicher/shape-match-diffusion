import argparse
import random
import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

import os.path as osp

from datasets import build_dataset
from models import build_model
from utils.logger import get_root_logger
from utils.metric_logger import MetricLogger
from utils.options import load_yaml, resolve_experiment_paths
from vis.plot_training import plot_training_curves


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
    if getattr(args, 'seed', None) is not None:
        opt['seed'] = args.seed

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
    parser.add_argument('--seed', type=int, default=None, help='global RNG seed (overrides config "seed")')
    parser.add_argument('--debug', action='store_true', help='run a couple of iterations for a quick smoke test')
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int):
    """Seed python/numpy/torch global RNGs. The sparse dataset draws its FPS start
    from the global numpy RNG, so this makes the sampled points reproducible too
    (with num_workers=0; see _seed_worker for the multi-worker case)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int):
    """Re-seed numpy/random per DataLoader worker. Forked workers share the parent's
    numpy seed otherwise, so every worker would draw identical FPS starts."""
    seed = torch.initial_seed() % 2**32   # torch gives each worker a distinct base seed
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------- #
# data helpers
# --------------------------------------------------------------------------- #
def _single_collate(batch):
    """batch_size=1 collate that returns the sample untouched.

    The shape pairs hold variable-size and sparse tensors (operators), which the
    default collate cannot stack, so we train one pair at a time.
    """
    return batch[0]


def autofill_feat_dim(opt, feat_dim):
    """Fill any network config field left null (`in_dim` for the encoder, `feat_dim` for
    the matrix denoiser) with the actual per-vertex feature dim from the data. Networks
    that hardcode the dim are left untouched."""
    for net_cfg in opt['networks'].values():
        for key in ('in_dim', 'feat_dim'):
            if key in net_cfg and net_cfg[key] is None:
                net_cfg[key] = feat_dim


def move_to_device(obj, device):
    """Recursively move tensors in a (possibly nested) dict to ``device``."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj


def _maybe_subset(dataset, n):
    """Deterministic evenly-spaced subset of `n` items, for cheap mid-training validation.
    Returns the dataset unchanged if `n` is falsy or >= its length. Full test evaluation
    (evaluate.py) is unaffected — it builds the test set directly."""
    if not n or n >= len(dataset):
        return dataset
    stride = max(1, len(dataset) // n)
    indices = list(range(0, len(dataset), stride))[:n]
    return Subset(dataset, indices)


def build_dataloaders(opt, num_workers):
    train_set = build_dataset(opt['datasets']['train'])
    val_set = build_dataset(opt['datasets']['val'])

    # optional val subset (opt['val']['subset']): validation runs a sampler per pair, so
    # the full val set can be slow; a fixed subset keeps epoch-to-epoch numbers comparable.
    val_set = _maybe_subset(val_set, (opt.get('val') or {}).get('subset'))

    # overfit knobs (opt['train']): 'subset' picks a few fixed pairs (use a deterministic
    # dataset phase so the sparse FPS points are fixed too), 'repeat' inflates one epoch to
    # many iterations of those pairs so validation still runs once per epoch, not per step.
    train_cfg = opt.get('train') or {}
    train_set = _maybe_subset(train_set, train_cfg.get('subset'))
    repeat = train_cfg.get('repeat')
    if repeat and repeat > 1:
        train_set = ConcatDataset([train_set] * int(repeat))

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              collate_fn=_single_collate, num_workers=num_workers,
                              worker_init_fn=_seed_worker)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            collate_fn=_single_collate, num_workers=num_workers,
                            worker_init_fn=_seed_worker)
    return train_set, train_loader, val_loader


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def train(opt, args):
    logger = get_root_logger()

    train_set, train_loader, val_loader = build_dataloaders(opt, args.num_workers)

    # match each network's input dim to the actual per-vertex feature dim
    autofill_feat_dim(opt, int(train_set[0]['first']['feat'].shape[-1]))

    model = build_model(opt)
    logger.info(f'Start training "{opt["name"]}" for {opt["train"]["total_epochs"]} epochs '
                f'on {len(train_set)} pairs (device: {model.device}).')

    # scalar logging -> results/metrics.csv (+ TensorBoard under experiments/<name>/tb/)
    results_dir = opt['path']['results']
    mlogger = MetricLogger(results_dir, tb_dir=osp.join(opt['path']['experiment_root'], 'tb'))

    log_freq = opt['train']['log_freq']
    # rough ETA: iters/sec since training started, projected over the remaining iters
    total_iters = opt['train']['total_epochs'] * len(train_loader)
    start_time = time.time()
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
                elapsed = time.time() - start_time
                eta = elapsed / model.curr_iter * (total_iters - model.curr_iter)  # excludes val time
                logger.info(f'[epoch {epoch:03d}][iter {model.curr_iter:06d}] lr:{lr:.2e} {loss_str} '
                            f'eta:{timedelta(seconds=int(eta))}')
                mlogger.log_many({f'Loss/{k}': v.item() for k, v in losses.items()},
                                 step=model.curr_iter, epoch=epoch)
                mlogger.log('LR', lr, step=model.curr_iter, epoch=epoch)

            if args.debug and i >= 2:
                break

        model.update_model_per_epoch()

        # end-of-epoch validation + checkpoint
        val_metrics = model.validation(val_loader)
        mlogger.log_many({f'Val/{k}': v for k, v in val_metrics.items()},
                         step=model.curr_iter, epoch=epoch)
        model.save_model()
        # refresh the curve PNGs each epoch so partial results are viewable mid-run
        plot_training_curves(results_dir)

        if args.debug:
            break

    mlogger.close()
    # save the final-epoch weights as the model to evaluate. We deliberately do NOT
    # select a "best" checkpoint by validation error: the val set coincides with the
    # test set, so cherry-picking on it would leak test labels into model selection.
    model.save_model(net_only=True)
    # dump a self-describing summary (config + network stats) to the experiment root
    info_path = model.save_experiment_info()
    logger.info('Training done (reporting final-epoch checkpoint).')
    logger.info(f'Wrote experiment info to {info_path}')


def main():
    args = parse_args()
    opt = build_opt(args)
    seed = opt.get('seed')
    if seed is not None:
        seed_everything(int(seed))
        get_root_logger().info(f'Global seed set to {seed}.')
    train(opt, args)


if __name__ == '__main__':
    main()
