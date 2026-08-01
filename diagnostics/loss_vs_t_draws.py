"""DIAGNOSTIC -- where on the t axis does the training loss actually live?

Motivation: feed_data draws t ~ U[0,1] with batch size 1, and training logs show many
~0 losses. Before reweighting the t sampler we map the loss landscape: per-draw row-CE
over a dense t grid for a few TRAIN pairs (bijective FPS, the exact training regime),
in both conditioning regimes a train step can be in:

  * feats-on  -- the ordinary step (feature conditioning present)
  * feats-off -- the CFG feature-dropout step (feature block zeroed), which is
                 feature_dropout of the steps in the snrfd configs

Each (t, draw) is one fresh forward-noising + denoiser pass, i.e. exactly what one train
step at that t would see (minus the dropout coin-flip, which the two regimes cover).

Outputs (figures/loss_vs_t/):
  <name>_pair{i}_draws.png -- per-draw loss scatter + median curve vs t, both regimes,
                              log-y with reference lines (uniform-CE ceiling, eps floors).
  <name>_zero_frac.png     -- fraction of draws with loss < eps vs t (across pairs).
Printed: the fraction of uniform-t train steps expected to land below eps in each
regime, and the mixture at the config's feature_dropout rate.

  python -m diagnostics.loss_vs_t_draws -c configs/joint_diffusionnet/faust_diffusionnet_512_FMD_snrfd.yaml
"""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets import build_dataset
from models import build_model
from train import autofill_feat_dim
from utils.options import load_yaml, resolve_experiment_paths
from utils.sinkhorn import logit_target, q_sample, log_sinkhorn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG_DIR = os.path.join(ROOT, 'figures', 'loss_vs_t')

C_ON, C_OFF = '#2a78d6', '#008300'   # feats-on / feats-off series hues
GRID = dict(color='0.85', lw=0.6)


def _build_train(config_path, checkpoint, device, split='train'):
    """Load a trained checkpoint + the TRAIN-regime dataset (bijective FPS -> gt_perm),
    mirroring nn_baseline_dense._build minus the eval-only forcing (no independent FPS,
    no densifier requirement)."""
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
    ds_opt = dict(opt['datasets'][split])
    # the train configs keep self-pairs (trivially ~0 loss at every t -- they are part of the
    # zero-loss story, reported separately); the curves here want genuine cross-shape pairs.
    ds_opt['exclude_self'] = True
    dataset = build_dataset(ds_opt)
    feat = dataset[0]['first'].get('feat')
    if feat is not None:
        autofill_feat_dim(opt, int(feat.shape[-1]))
    model = build_model(opt)
    model.eval()
    return model, dataset, opt


@torch.no_grad()
def per_draw_losses(model, F_x, F_y, D_x, D_y, P0, u0, ts, repeats, chunk=16):
    """Row-CE for every (t, draw) pair, one fresh noising each. Returns (T, repeats).
    Draws are batched into chunks of denoiser forwards for speed; the per-sample loss is
    the row-mean CE (what _forward_ce averages over the batch)."""
    jobs = [(i, r) for i in range(len(ts)) for r in range(repeats)]
    out = torch.empty(len(ts), repeats)
    exp = lambda z, B: z.expand(B, *z.shape[1:]).contiguous()
    for lo in tqdm(range(0, len(jobs), chunk), desc='t-grid', leave=False):
        batch = jobs[lo:lo + chunk]
        B = len(batch)
        t = torch.tensor([ts[i] for i, _ in batch], device=model.device, dtype=torch.float32)
        u_t = q_sample(exp(u0, B), t, s=model.schedule_s, logsnr_shift=model.logsnr_shift)
        P_t = log_sinkhorn(u_t, n_iters=model.proj_iters).exp()
        u0_hat = model.networks['denoiser'](P_t, exp(F_x, B), exp(F_y, B), exp(D_x, B), exp(D_y, B), t)
        logP = model._row_logprob(u0_hat)
        loss = -(exp(P0, B) * logP).sum(-1).mean(-1)               # (B,) per-draw row-CE
        for (i, r), lv in zip(batch, loss.cpu()):
            out[i, r] = lv
    return out


def _draws_plot(ts, losses, name, pair_idx, eps_lines, n, path):
    """Per-draw scatter + median curves for one pair, both regimes, log-y."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    floor = 1e-6                                                    # display clamp for log axis
    for key, color, label in (('on', C_ON, 'features on'), ('off', C_OFF, 'features dropped')):
        L = losses[key].clamp_min(floor).numpy()                    # (T, R)
        for r in range(L.shape[1]):
            ax.scatter(ts, L[:, r], s=7, color=color, alpha=0.3, linewidths=0)
        ax.plot(ts, np.median(L, axis=1), color=color, lw=2, label=f'{label} (median)')
    ax.axhline(np.log(n), color='0.55', lw=1, ls=':', label=f'uniform-CE ceiling log(n)={np.log(n):.2f}')
    for eps in eps_lines:
        ax.axhline(eps, color='0.55', lw=1, ls='--')
        ax.annotate(f'eps={eps:g}', (0.005, eps), fontsize=7, color='0.4',
                    va='bottom', ha='left')
    ax.set_yscale('log')
    ax.set_xlabel('diffusion time t')
    ax.set_ylabel('row-CE loss (log scale)')
    ax.set_title(f'{name} -- pair {pair_idx}: per-draw training loss vs t')
    ax.grid(True, which='major', **GRID)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc='center right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _zero_frac_plot(ts, all_losses, eps_list, name, path):
    """Fraction of draws (pooled over pairs) with loss < eps, vs t, per regime."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for key, color, label in (('on', C_ON, 'features on'), ('off', C_OFF, 'features dropped')):
        pooled = torch.cat([l[key] for l in all_losses], dim=1).numpy()   # (T, pairs*R)
        for eps, ls in zip(eps_list, ('-', '--')):
            ax.plot(ts, (pooled < eps).mean(axis=1), color=color, lw=2, ls=ls,
                    label=f'{label}, eps={eps:g}')
    ax.set_xlabel('diffusion time t')
    ax.set_ylabel('fraction of draws with loss < eps')
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f'{name}: near-zero-loss fraction vs t')
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(config_path, checkpoint, device, pairs, t_bins, repeats, eps_list, chunk):
    model, dataset, opt = _build_train(config_path, checkpoint, device)
    name = opt['name']
    os.makedirs(FIG_DIR, exist_ok=True)
    ts = np.linspace(0.0, 1.0, t_bins).tolist()
    all_losses = []

    for idx in pairs:
        data = dataset[idx]
        F_x, F_y, D_x, D_y, P0 = model._sparse_inputs(data)
        u0 = logit_target(P0, model.eta)
        n = P0.shape[-1]
        losses = {'on': per_draw_losses(model, F_x, F_y, D_x, D_y, P0, u0, ts, repeats, chunk),
                  'off': per_draw_losses(model, torch.zeros_like(F_x), torch.zeros_like(F_y),
                                         D_x, D_y, P0, u0, ts, repeats, chunk)}
        all_losses.append(losses)
        med_on = np.median(losses['on'].numpy(), axis=1)
        print(f'pair {idx:>3}: feats-on median loss  t=0: {med_on[0]:.4f}  '
              f't=0.5: {med_on[t_bins // 2]:.4f}  t=1: {med_on[-1]:.4f}')
        _draws_plot(ts, losses, name, idx, eps_list, n,
                    os.path.join(FIG_DIR, f'{name}_pair{idx}_draws.png'))

    _zero_frac_plot(ts, all_losses, eps_list, name,
                    os.path.join(FIG_DIR, f'{name}_zero_frac.png'))

    # uniform-t expectation: the t grid is uniform, so pooling over it estimates the
    # fraction of ordinary train steps whose loss lands below eps.
    p_drop = model.feature_dropout
    print(f'\n{name}: expected near-zero-loss fraction under t ~ U[0,1]')
    for eps in eps_list:
        f_on = float(torch.cat([l['on'] for l in all_losses], 1).lt(eps).float().mean())
        f_off = float(torch.cat([l['off'] for l in all_losses], 1).lt(eps).float().mean())
        mix = (1 - p_drop) * f_on + p_drop * f_off
        print(f'  eps={eps:g}: feats-on {f_on * 100:5.1f}%   feats-off {f_off * 100:5.1f}%   '
              f'train mixture (p_drop={p_drop:g}) {mix * 100:5.1f}%')
    print(f'\nfigures -> {FIG_DIR}/')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--pairs', default='0,3,7', help='comma list of TRAIN pair indices')
    p.add_argument('--t-bins', type=int, default=33, help='t grid resolution over [0,1]')
    p.add_argument('--repeats', type=int, default=6, help='independent noise draws per t')
    p.add_argument('--eps', default='1e-2,1e-3', help='comma list of near-zero thresholds')
    p.add_argument('--chunk', type=int, default=16, help='denoiser forwards per batch')
    p.add_argument('--device', default=None)
    args = p.parse_args()
    run(args.config, args.checkpoint, args.device,
        [int(x) for x in args.pairs.split(',')], args.t_bins, args.repeats,
        [float(x) for x in args.eps.split(',')], args.chunk)


if __name__ == '__main__':
    main()
