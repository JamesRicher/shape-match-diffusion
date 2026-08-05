"""DIAGNOSTIC -- does the assignment matrix genuinely evolve over the reverse process,
or is the match decided in one shot?

Runs the DDIM reverse sampler (faithfully re-implemented so the full doubly-stochastic
P_t at every step can be captured -- model.sample only returns hard argmax snaps) on a
FAUST-trained model applied to SCAPE pairs, and looks at how uncertain / how mobile the
assignment matrix is along t=1 -> 0.

For each pair it saves:
  * a strip of P_t heatmaps at several timesteps, columns reordered so the GT match is the
    diagonal: a sharp diagonal = confident + correct, off-diagonal mass = uncertainty/error;
  * a per-pair curve of three trajectory statistics vs t:
      - mean row entropy of P_t (uncertainty; 0 = one-hot, log n = uniform),
      - correct-diagonal mass (fraction of row mass on the GT column),
      - argmax-churn: fraction of rows whose hard match CHANGED from the previous step
        (the direct "is there a trajectory or is it frozen?" signal -- near 0 after the
        first steps means one-shot). See the reverse-trajectory-inert / loss-vs-t memories.

  python -m diagnostics.assignment_trajectory \
      -c configs/joint_diffusionnet/scape_diffusionnet_512_FMD_snrfd_tw_gt.yaml \
      --checkpoint experiments/diffusionnet/faust_diffusionnet_512_FMD_snrfd_tw_gt/models/final.pth \
      --pairs 0,1
"""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

from datasets import build_dataset
from models import build_model
from train import autofill_feat_dim
from utils.options import load_yaml, resolve_experiment_paths
from utils.sinkhorn import cosine_alpha_bar, log_sinkhorn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG_DIR = os.path.join(ROOT, 'figures', 'assignment_trajectory')
GRID = dict(color='0.85', lw=0.6)


def _build(config_path, checkpoint, device, split='test'):
    """Load a checkpoint into the config's model + the config's dataset (cross-dataset when
    the checkpoint is from another dataset). Mirrors loss_vs_t_draws._build_train."""
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
    ds_opt['exclude_self'] = True                     # genuine cross-shape pairs only
    dataset = build_dataset(ds_opt)
    feat = dataset[0]['first'].get('feat')
    if feat is not None:
        autofill_feat_dim(opt, int(feat.shape[-1]))
    model = build_model(opt)
    model.eval()
    return model, dataset, opt, ckpt


@torch.no_grad()
def sample_with_matrices(model, F_x, F_y, D_x, D_y, steps):
    """DDIM reverse process (deterministic, eta=0), capturing the read-in DS matrix P_t at
    every step plus the final converged P0. Faithful copy of MatrixDiffusionModel.sample's
    loop. Returns (P_list [(n_y,n_x)] length steps+1, ts [steps+1] with the final 0.0)."""
    net = model.networks['denoiser']
    B, n = F_x.shape[0], F_x.shape[1]
    u = torch.randn(B, n, n, device=model.device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=model.device)
    mats, keep_t = [], []
    for i in range(steps):
        t_i, t_prev = ts[i], ts[i + 1]
        P_t = log_sinkhorn(u, n_iters=model.proj_iters).exp()
        mats.append(P_t[0].cpu().numpy())
        keep_t.append(float(t_i))
        u0_hat = net(P_t, F_x, F_y, D_x, D_y, t_i.reshape(1).expand(B))
        ab_t = cosine_alpha_bar(t_i, model.schedule_s, model.logsnr_shift)
        ab_p = cosine_alpha_bar(t_prev, model.schedule_s, model.logsnr_shift)
        eps_hat = (u - ab_t.sqrt() * u0_hat) / (1.0 - ab_t).clamp_min(1e-8).sqrt()
        u = ab_p.sqrt() * u0_hat + (1.0 - ab_p).clamp_min(0.0).sqrt() * eps_hat
    mats.append(log_sinkhorn(u, n_iters=model.final_iters).exp()[0].cpu().numpy())
    keep_t.append(float(ts[-1]))
    return mats, keep_t


def row_entropy(P):
    """Mean over rows of the row-normalised Shannon entropy (nats). P (n_y, n_x)."""
    r = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
    return float(-(r * np.log(np.clip(r, 1e-12, None))).sum(1).mean())


def trajectory_stats(mats, gt_col):
    """Per-step (entropy, correct-diagonal mass, argmax-churn-from-prev-step). gt_col (n_y,)
    is the GT source column per target row."""
    ent, mass, churn = [], [], []
    rows = np.arange(len(gt_col))
    prev = None
    for P in mats:
        r = P / np.clip(P.sum(1, keepdims=True), 1e-12, None)
        ent.append(row_entropy(P))
        mass.append(float(r[rows, gt_col].mean()))
        am = P.argmax(1)
        churn.append(np.nan if prev is None else float((am != prev).mean()))
        prev = am
    return np.array(ent), np.array(mass), np.array(churn)


def heatmap_strip(mats, ts, gt_col, snap_idx, name, pair_idx, acc, path):
    """Strip of GT-diagonalised P_t heatmaps at the chosen step indices."""
    order = np.argsort(gt_col)                         # sort rows so the diagonal reads cleanly
    gcol = gt_col[order]
    fig, axes = plt.subplots(1, len(snap_idx), figsize=(3.0 * len(snap_idx), 3.2))
    if len(snap_idx) == 1:
        axes = [axes]
    for ax, si in zip(axes, snap_idx):
        M = mats[si][order][:, gcol]                   # correct match now on the diagonal
        vmax = np.percentile(M, 99.5)
        ax.imshow(M, cmap='magma', vmin=0, vmax=max(vmax, 1e-6), interpolation='nearest')
        ax.set_title(f't={ts[si]:.2f}\nH={row_entropy(mats[si]):.2f}  diag={M.diagonal().mean():.2f}',
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{name} -- pair {pair_idx}: P_t along reverse process '
                 f'(GT = diagonal, final sparse-acc {acc:.3f})', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def animate_matrix(mats, ts, gt_col, crop, name, pair_idx, path, fps=8):
    """Animated GIF of P_t denoising, cropped to a crop x crop GT-diagonalised block so
    individual assignment cells are visible (the diagonal = the correct match)."""
    order = np.argsort(gt_col)
    gcol = gt_col[order]
    c = min(crop, len(order))
    ri, ci = order[:c], gcol[:c]                       # same c points on both axes
    frames = [P[ri][:, ci] for P in mats]              # each (c, c), diagonal = correct
    # per-frame vmax: a DS matrix at t=1 is near-uniform ~1/n, so a fixed (final-frame) scale
    # renders it near-black. Normalising each frame to its own 99.5 pct shows the noise texture
    # at every t while still letting the diagonal sharpen (values printed in the title stay raw).
    vmaxes = [max(np.percentile(f, 99.5), 1e-6) for f in frames]

    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    im = ax.imshow(frames[0], cmap='magma', vmin=0, vmax=vmaxes[0], interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel('source (GT-ordered)'); ax.set_ylabel('target (GT-ordered)')
    ttl = ax.set_title('', fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.86)  # reserve room for 2-line title

    def upd(k):
        im.set_data(frames[k])
        im.set_clim(0, vmaxes[k])
        ttl.set_text(f'{name} pair {pair_idx} -- top {c}x{c} (per-frame scaled)\n'
                     f't={ts[k]:.2f}   H={row_entropy(mats[k]):.2f}   diag={frames[k].diagonal().mean():.2f}')
        return im, ttl

    anim = FuncAnimation(fig, upd, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def stats_plot(ts, ent, mass, churn, n, name, pair_idx, path):
    """Entropy / correct-mass / argmax-churn vs t for one pair."""
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.plot(ts, ent, color='#8a2be2', lw=2, marker='o', ms=3, label='mean row entropy (nats)')
    ax.axhline(np.log(n), color='#8a2be2', lw=1, ls=':', label=f'uniform log(n)={np.log(n):.2f}')
    ax.set_xlabel('diffusion time t  (reverse process runs 1 -> 0)')
    ax.set_ylabel('entropy (nats)', color='#8a2be2')
    ax.invert_xaxis()
    ax2 = ax.twinx()
    ax2.plot(ts, mass, color='#008300', lw=2, marker='s', ms=3, label='correct-diagonal mass')
    ax2.plot(ts, churn, color='#d62728', lw=2, marker='^', ms=3, label='argmax churn vs prev step')
    ax2.set_ylabel('fraction', color='0.2')
    ax2.set_ylim(-0.02, 1.02)
    ax.set_title(f'{name} -- pair {pair_idx}: trajectory activity vs t')
    ax.grid(True, **GRID); ax.set_axisbelow(True)
    for s in ('top',):
        ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    lines = ax.get_lines()[:2] + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc='center left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(config_path, checkpoint, device, pairs, steps, n_snaps, crop):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device)
    name = f"{opt['name']}__on__{opt['datasets']['test']['name']}"
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f'model: {opt["name"]}   checkpoint: {ckpt}')
    print(f'eval dataset: {opt["datasets"]["test"]["name"]}   steps={steps}\n')

    for idx in tqdm(pairs, desc='pairs'):
        data = dataset[idx]
        F_x, F_y, D_x, D_y, P0 = model._sparse_inputs(data)
        assert P0 is not None, 'need bijective sparse GT (P0); use the test split, not independent FPS'
        gt_col = P0[0].detach().cpu().numpy().argmax(1)          # (n_y,) target row -> GT source col
        n = P0.shape[-1]

        mats, ts = sample_with_matrices(model, F_x, F_y, D_x, D_y, steps)
        ts = np.array(ts)
        final_acc = float((mats[-1].argmax(1) == gt_col).mean())  # sparse argmax accuracy

        ent, mass, churn = trajectory_stats(mats, gt_col)
        snap_idx = np.unique(np.linspace(0, len(mats) - 1, n_snaps).round().astype(int))
        heatmap_strip(mats, ts, gt_col, snap_idx, name, idx, final_acc,
                      os.path.join(FIG_DIR, f'{name}_pair{idx}_heatmaps.png'))
        stats_plot(ts, ent, mass, churn, n, name, idx,
                   os.path.join(FIG_DIR, f'{name}_pair{idx}_stats.png'))
        animate_matrix(mats, ts, gt_col, crop, name, idx,
                       os.path.join(FIG_DIR, f'{name}_pair{idx}_denoise.gif'))

        active = np.nanmean(churn[max(1, len(churn) // 3):])     # churn after the first third
        print(f'pair {idx:>3}: final argmax-acc {final_acc:.3f}   entropy t=1 {ent[0]:.2f} -> '
              f't=0 {ent[-1]:.2f}   mean late-churn {active:.3f}  '
              f'({"one-shot" if active < 0.02 else "trajectory active"})')

    print(f'\nfigures -> {FIG_DIR}/')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', default='configs/joint_diffusionnet/scape_diffusionnet_512_FMD_snrfd_tw_gt.yaml',
                   help='config supplying the EVAL dataset + model arch (default: SCAPE)')
    p.add_argument('--checkpoint',
                   default='experiments/diffusionnet/faust_diffusionnet_512_FMD_snrfd_tw_gt/models/final.pth',
                   help='trained weights to load (default: FAUST-trained tw_gt model)')
    p.add_argument('--pairs', default='0,1', help='comma list of eval pair indices')
    p.add_argument('--steps', type=int, default=50, help='reverse-process steps')
    p.add_argument('--snaps', type=int, default=6, help='timestep heatmaps per pair')
    p.add_argument('--crop', type=int, default=48, help='GIF shows the top crop x crop GT block')
    p.add_argument('--device', default=None)
    args = p.parse_args()
    run(args.config, args.checkpoint, args.device,
        [int(x) for x in args.pairs.split(',')], args.steps, args.snaps, args.crop)


if __name__ == '__main__':
    main()
