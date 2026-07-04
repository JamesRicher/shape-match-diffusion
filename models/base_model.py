import os
import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim

from networks import build_network
from metrics import build_metric
from utils.logger import get_root_logger, AvgTimer


def to_numpy(x):
    """Detach a tensor to a numpy array; pass through anything already numpy-like."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class BaseModel:
    """
    Base model to be inherited by concrete models.

    A model owns one or more registered networks and the training machinery around
    them (optimizers, schedulers, losses) plus a metrics-driven validation loop.
    Subclasses typically override ``feed_data``, ``optimize_parameters`` and
    ``validate_single``.

    The single ``opt`` dict drives everything:
        opt['is_train']  (bool)
        opt['device']    (str, e.g. 'cuda' / 'cpu'); optional, auto-detected otherwise
        opt['networks']  ({name: {'type': ..., **kwargs}})
        opt['train']     ({'optims': {...}, 'schedulers': {...}, 'losses': {...}})
        opt['val']       ({'metrics': {name: {'type': ..., **kwargs}}})
        opt['path']      ({'models': ..., 'visualization': ..., 'resume_state': ..., 'resume': bool})
    """

    def __init__(self, opt):
        self.opt = opt
        self.is_train = opt.get('is_train', False)
        self.device = torch.device(
            opt.get('device') or ('cuda' if torch.cuda.is_available() else 'cpu'))

        # build networks and move them to the device
        self.networks = OrderedDict()
        self._setup_networks()
        for name, net in self.networks.items():
            self.networks[name] = net.to(self.device)
        self.print_networks()

        # validation metrics
        self.metrics = OrderedDict()
        self._setup_metrics()

        # best-model tracking (used by validation in both train and inference modes)
        self.best_metric = None
        self.best_networks_state_dict = None

        # training machinery
        if self.is_train:
            self.train()
            self._init_training_setting()

        # optionally resume from a checkpoint
        load_path = self.opt.get('path', {}).get('resume_state')
        if load_path and os.path.isfile(load_path):
            state_dict = torch.load(load_path, map_location='cpu')
            resume = self.is_train and self.opt.get('path', {}).get('resume', True)
            self.resume_model(state_dict, net_only=not resume)

    # ------------------------------------------------------------------ #
    # methods a concrete model overrides
    # ------------------------------------------------------------------ #
    def feed_data(self, data):
        """Run the forward pass and populate ``self.loss_metrics``."""
        raise NotImplementedError

    def validate_single(self, data):
        """Run inference on a single (batch-of-one) pair and return a point-to-point
        map ``p2p`` (shape y -> shape x), consumed by the validation metrics."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # training step
    # ------------------------------------------------------------------ #
    def optimize_parameters(self):
        """Sum the loss terms, back-prop and step every optimizer."""
        loss = 0.0
        for k, v in self.loss_metrics.items():
            if k != 'l_total':
                loss = loss + v
        self.loss_metrics['l_total'] = loss

        for name in self.optimizers:
            self.optimizers[name].zero_grad()
        loss.backward()
        for key in self.networks:
            torch.nn.utils.clip_grad_norm_(self.networks[key].parameters(), 1.0)
        for name in self.optimizers:
            self.optimizers[name].step()

    def update_model_per_iteration(self):
        for name in self.schedulers:
            if isinstance(self.schedulers[name], optim.lr_scheduler.OneCycleLR):
                self.schedulers[name].step()

    def update_model_per_epoch(self):
        per_epoch = (optim.lr_scheduler.StepLR, optim.lr_scheduler.MultiStepLR,
                     optim.lr_scheduler.ExponentialLR, optim.lr_scheduler.CosineAnnealingLR,
                     optim.lr_scheduler.CosineAnnealingWarmRestarts)
        for name in self.schedulers:
            if isinstance(self.schedulers[name], per_epoch):
                self.schedulers[name].step()

    def get_current_learning_rate(self):
        return [opt.param_groups[0]['lr'] for opt in self.optimizers.values()]

    def get_loss_metrics(self):
        return self.loss_metrics

    # ------------------------------------------------------------------ #
    # validation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validation(self, dataloader, update=True):
        """Evaluate geodesic error / PCK over a validation dataloader.

        Relies on metrics named ``geo_error`` and (optionally) ``plot_pck`` being
        registered via ``opt['val']['metrics']``.
        """
        self.eval()
        logger = get_root_logger()

        geo_errors = []
        timer = AvgTimer()
        pbar = tqdm(dataloader)
        for data in pbar:
            p2p = to_numpy(self.validate_single(data))
            timer.record()

            if 'geo_error' in self.metrics:
                data_x, data_y = data['first'], data['second']
                dist_x = data_x['dist'] if 'dist' in data_x else torch.cdist(data_x['verts'], data_x['verts'])
                geo_err = self.metrics['geo_error'](
                    to_numpy(dist_x), to_numpy(data_x['corr']), to_numpy(data_y['corr']),
                    p2p, return_mean=False)
                pbar.set_description(f'geo error: {geo_err.mean():.4f}')
                geo_errors.append(geo_err)

        logger.info(f'Avg inference time: {timer.get_avg_time():.4f}s')

        metrics_result = {}
        if geo_errors:
            geo_errors = np.concatenate(geo_errors)
            avg_geo_error = float(geo_errors.mean())
            metrics_result['avg_error'] = avg_geo_error
            logger.info(f'Val avg error: {avg_geo_error:.4f}')

            if 'plot_pck' in self.metrics:
                auc, fig, pcks = self.metrics['plot_pck'](geo_errors)
                metrics_result['auc'] = float(auc)
                logger.info(f'Val auc: {auc:.4f}')
                vis_dir = self.opt.get('path', {}).get('visualization')
                if vis_dir:
                    os.makedirs(vis_dir, exist_ok=True)
                    fig.savefig(os.path.join(vis_dir, 'pck.png'), bbox_inches='tight')
                    np.save(os.path.join(vis_dir, 'pck.npy'), pcks)
                plt.close(fig)

            if update and (self.best_metric is None or avg_geo_error < self.best_metric):
                self.best_metric = avg_geo_error
                self.best_networks_state_dict = self._get_networks_state_dict()
                logger.info(f'Best model updated, avg geodesic error: {self.best_metric:.4f}')

        self.train()
        return metrics_result

    # ------------------------------------------------------------------ #
    # setup helpers
    # ------------------------------------------------------------------ #
    def _init_training_setting(self):
        self.curr_epoch = 0
        self.curr_iter = 0
        self.optimizers = OrderedDict()
        self.schedulers = OrderedDict()
        self._setup_optimizers()
        self._setup_schedulers()
        self.losses = OrderedDict()
        self._setup_losses()
        self.loss_metrics = OrderedDict()
        self.best_networks_state_dict = self._get_networks_state_dict()
        self.best_metric = None

    def _setup_networks(self):
        for name, network_opt in deepcopy(self.opt['networks']).items():
            self.networks[name] = build_network(network_opt)

    def _setup_metrics(self):
        val_opt = deepcopy(self.opt.get('val') or {})
        for name, metric_opt in val_opt.get('metrics', {}).items():
            self.metrics[name] = build_metric(metric_opt)
        if not self.metrics:
            get_root_logger().info('No metric is registered.')

    def _setup_optimizers(self):
        optim_map = {'Adam': optim.Adam, 'AdamW': optim.AdamW,
                     'RMSprop': optim.RMSprop, 'SGD': optim.SGD}
        train_opt = deepcopy(self.opt['train'])
        for name, net in self.networks.items():
            params = [p for p in net.parameters() if p.requires_grad]
            if not params:
                get_root_logger().info(f'Network {name} has no trainable params. Ignore it.')
                continue
            if name not in train_opt.get('optims', {}):
                get_root_logger().warning(f'Network {name} will not be optimized.')
                continue
            optim_cfg = train_opt['optims'][name]
            optim_type = optim_cfg.pop('type')
            if optim_type not in optim_map:
                raise NotImplementedError(f'optimizer {optim_type} is not supported.')
            self.optimizers[name] = optim_map[optim_type](params, **optim_cfg)

    def _setup_schedulers(self):
        sched_map = {
            'StepLR': optim.lr_scheduler.StepLR,
            'MultiStepLR': optim.lr_scheduler.MultiStepLR,
            'ExponentialLR': optim.lr_scheduler.ExponentialLR,
            'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
            'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
            'OneCycleLR': optim.lr_scheduler.OneCycleLR,
        }
        scheduler_opts = deepcopy(self.opt['train']).get('schedulers', {})
        for name, optimizer in self.optimizers.items():
            if name not in scheduler_opts or scheduler_opts[name].get('type', 'none') == 'none':
                self.schedulers[name] = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1)
                continue
            sched_cfg = scheduler_opts[name]
            sched_type = sched_cfg.pop('type')
            if sched_type not in sched_map:
                raise NotImplementedError(f'Scheduler {sched_type} is not implemented.')
            self.schedulers[name] = sched_map[sched_type](optimizer, **sched_cfg)

    def _setup_losses(self):
        train_opt = deepcopy(self.opt['train'])
        if 'losses' not in train_opt:
            get_root_logger().info('No loss is registered.')
            return
        # Lazy import: the losses package is optional until training objectives exist.
        from losses import build_loss
        for name, loss_opt in train_opt['losses'].items():
            self.losses[name] = build_loss(loss_opt).to(self.device)

    # ------------------------------------------------------------------ #
    # state / mode
    # ------------------------------------------------------------------ #
    def _get_networks_state_dict(self):
        return {name: deepcopy(net.state_dict()) for name, net in self.networks.items()}

    def print_networks(self):
        logger = get_root_logger()
        for name, net in self.networks.items():
            n_params = sum(p.numel() for p in net.parameters())
            logger.info(f'Network [{name}] {net.__class__.__name__}, params: {n_params:,d}')

    def train(self):
        self.is_train = True
        for net in self.networks.values():
            net.train()

    def eval(self):
        self.is_train = False
        for net in self.networks.values():
            net.eval()

    def save_model(self, net_only=False, best=False):
        networks_state_dict = self.best_networks_state_dict if best else self._get_networks_state_dict()
        if net_only:
            state_dict = {'networks': networks_state_dict}
            save_filename = 'final.pth'
        else:
            state_dict = {
                'networks': networks_state_dict,
                'epoch': self.curr_epoch,
                'iter': self.curr_iter,
                'optimizers': {name: o.state_dict() for name, o in self.optimizers.items()},
                'schedulers': {name: s.state_dict() for name, s in self.schedulers.items()},
            }
            save_filename = f'{self.curr_iter}.pth'

        models_dir = self.opt['path']['models']
        os.makedirs(models_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(models_dir, save_filename))

    def resume_model(self, resume_state, net_only=False, verbose=True):
        logger = get_root_logger()
        for name in self.networks:
            if name not in resume_state['networks']:
                if verbose:
                    logger.warning(f'Network {name} cannot be resumed.')
                continue
            net_state_dict = {k.replace('module.', ''): v
                              for k, v in resume_state['networks'][name].items()}
            self.networks[name].load_state_dict(net_state_dict)
            if verbose:
                logger.info(f'Resuming network: {name}')

        if not net_only:
            for name in self.optimizers:
                if name in resume_state.get('optimizers', {}):
                    self.optimizers[name].load_state_dict(resume_state['optimizers'][name])
            for name in self.schedulers:
                if name in resume_state.get('schedulers', {}):
                    self.schedulers[name].load_state_dict(resume_state['schedulers'][name])
            self.curr_iter = resume_state['iter']
            self.curr_epoch = resume_state['epoch']
            if verbose:
                logger.info(f'Resuming training from epoch {self.curr_epoch}, iter {self.curr_iter}.')
