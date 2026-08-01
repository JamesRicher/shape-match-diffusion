"""MODEL-FREE noise-schedule chooser: how well does the GROUND-TRUTH map survive the forward
process, as a function of t, for a sweep of log-SNR shifts b? No denoiser, no checkpoint -- this
measures only the forward pipeline (noise the GT logit target -> Sinkhorn project -> compare to
GT), so it tells you which schedule to TRAIN with before spending a single training run.

WHY
---
The reverse trajectory is inert: the map is a feature one-shot decided at the top, and the
genuinely-ambiguous 'work band' is jammed into a narrow high-t sliver because the sharp logit
target (spike ~ log n_sparse) keeps GT recoverable until high t. A downward log-SNR shift b<0
(cosine_alpha_bar's logsnr_shift) lowers SNR at every interior t, sliding that band toward
mid-trajectory and widening it (Hoogeboom 'simple diffusion' 2023; Chen 2023).

WHAT IT MEASURES  (for each shift b, at each t, averaged over pairs and noise draws)
  * survival accuracy : fraction of rows where argmax(Pi_S(q_sample(u0))) == GT match
  * median row-CE     : robust CE of the noised map vs GT (mean is swamped by the sharpening tail)
This is the ORACLE ceiling -- the best any P_t-reading denoiser could do -- so a schedule whose
survival declines GRADUALLY across [0,1] (rather than sitting at ~1 then cliffing near t=1) is the
one that gives the trajectory work to do. Pick that b; then train with diffusion.logsnr_shift: b.

USAGE
-----
  python -m diagnostics.gt_survival_sweep -c configs/joint_diffusionnet/scape_diffusionnet_512_FMD.yaml
  python -m diagnostics.gt_survival_sweep -c <cfg> --shifts 0 -1 -1.75 -2.5 --num-pairs 15 --draws 6

Optional: --num-pairs N (default 10), --draws D per (b,t) (default 4), --n-t (t-grid size, 26),
--device, --out DIR (default figures/), --seed.
"""
import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from datasets import build_dataset
from utils.options import load_yaml
from utils.sinkhorn import logit_target, gaussian_target, safe_log, q_sample, log_sinkhorn

INK, GRID, FAINT, REF = '#1B2027', '#E3E6E4', '#8A929C', '#A65A12'
C_LIGHT, C_DARK = '#A9CFCB', '#0E4A46'          # sequential teal ramp (shift magnitude)


def _ramp(n):
    a, b = np.array(mcolors.to_rgb(C_LIGHT)), np.array(mcolors.to_rgb(C_DARK))
    return [mcolors.to_hex((1 - x) * a + x * b) for x in np.linspace(0, 1, max(n, 1))]


def _dataset(config_path):
    """Model-free: just the test dataset in bijective mode so gt_perm exists, plus the config's
    diffusion knobs (eta, schedule_s, proj_iters). No model, no checkpoint."""
    opt = load_yaml(config_path)
    opt['is_train'] = False
    ds_opt = dict(opt['datasets']['test'])
    ds_opt['ret_evecs'] = False                 # we only need the sparse GT correspondence
    dataset = build_dataset(ds_opt)
    dataset.independent_fps = False             # bijective sampling -> data['gt_perm']
    diff = opt.get('diffusion', {})
    return (dataset, opt.get('name', 'model'),
            diff.get('eta', 0.1), diff.get('schedule_s', 0.008), diff.get('proj_iters', 6),
            diff.get('target', {}))


@torch.no_grad()
def _curves(dataset, idxs, shifts, tgrid, draws, eta, s, proj, device, gen, target=None):
    """Returns acc[b, t] and medce[b, t], averaged over pairs and noise draws."""
    nb, nt = len(shifts), len(tgrid)
    acc = np.zeros((nb, nt))
    medce = np.zeros((nb, nt))
    m_ref = None
    for i in tqdm(idxs, desc='pairs'):
        data = dataset[i]
        P0 = data['gt_perm']
        P0 = P0.float().to(device)
        if P0.dim() == 2:
            P0 = P0.unsqueeze(0)                # (1, n_y, n_x)
        m_ref = P0.shape[-1]
        gt = P0.argmax(-1)                      # (1, n_y)
        if target and target.get('type') == 'gaussian':    # config's diffusion.target block
            D_x = data['first']['sparse']['dist'].float().to(device).unsqueeze(0)
            sigma = target.get('sigma', 0.03)
            u0 = safe_log(gaussian_target(P0, D_x, sigma,
                                          target.get('cutoff', 3.0 * sigma),
                                          target.get('floor', 2e-4)))
        else:
            u0 = logit_target(P0, eta)
        for bi, b in enumerate(shifts):
            for ti, tv in enumerate(tgrid):
                t = torch.full((1,), float(tv), device=device)
                a_acc, a_med = 0.0, 0.0
                for _ in range(draws):
                    noise = torch.randn(u0.shape, generator=gen, device=device)
                    u_t = q_sample(u0, t, noise=noise, s=s, logsnr_shift=b)
                    logP = log_sinkhorn(u_t, n_iters=proj)
                    logP = logP - torch.logsumexp(logP, dim=-1, keepdim=True)   # row-normalise
                    a_acc += (logP.argmax(-1) == gt).float().mean().item()
                    a_med += (-(P0 * logP).sum(-1)).median().item()
                acc[bi, ti] += a_acc / draws
                medce[bi, ti] += a_med / draws
    return acc / len(idxs), medce / len(idxs), m_ref


def _cross50(tgrid, acc_row):
    """t where survival accuracy drops through 50% (linear interp); None if it never does."""
    below = np.where(acc_row < 0.5)[0]
    if not len(below) or below[0] == 0:
        return None
    j = below[0]
    t0, t1, a0, a1 = tgrid[j - 1], tgrid[j], acc_row[j - 1], acc_row[j]
    return float(t0 + (t1 - t0) * (a0 - 0.5) / (a0 - a1 + 1e-12))


def run(config_path, shifts, num_pairs, draws, n_t, device, out_dir, seed):
    dataset, name, eta, s, proj, target = _dataset(config_path)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    gen = torch.Generator(device=device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = min(num_pairs, len(dataset))
    idxs = sorted(rng.choice(len(dataset), size=n, replace=False).tolist())
    tgrid = np.linspace(0.02, 0.98, n_t)

    acc, medce, m = _curves(dataset, idxs, shifts, tgrid, draws, eta, s, proj, device, gen, target)
    colors = _ramp(len(shifts))
    logm = float(np.log(m))

    # ---- figure: two panels, one line per shift ----
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7.4, 6.8), dpi=150,
                                   gridspec_kw={'hspace': 0.16})
    for ax in (ax1, ax2):
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        for sp in ('left', 'bottom'):
            ax.spines[sp].set_color(FAINT)
        ax.tick_params(colors=FAINT, labelsize=9)

    ax1.axhline(50, color=REF, lw=1.1, ls='--', zorder=1)
    for bi, b in enumerate(shifts):
        ax1.plot(tgrid, acc[bi] * 100, color=colors[bi], lw=2.3, zorder=3 + bi,
                 label=f'b = {b:g}')
    ax1.set_ylim(0, 103)
    ax1.set_ylabel('GT-survival accuracy  (%)', color=INK, fontsize=10)
    ax1.set_title(f'{name}: GT survival vs t  (model-free, {len(idxs)} pairs, {draws} draws)  '
                  f'— pick b that declines gradually', color=INK, fontsize=11.5, loc='left', pad=10)
    ax1.legend(frameon=False, fontsize=9, loc='lower left', labelcolor=INK, title='log-SNR shift',
               title_fontsize=8.5)

    ax2.axhline(logm, color=REF, lw=1.1, ls='--', zorder=1)
    ax2.text(0.015, logm - 0.2, f'uniform (ln {m})', color=REF, fontsize=8, ha='left', va='top')
    for bi, b in enumerate(shifts):
        ax2.plot(tgrid, medce[bi], color=colors[bi], lw=2.3, zorder=3 + bi)
    ax2.set_ylim(bottom=-0.1)
    ax2.set_ylabel('median row-CE of $P_t$ vs GT', color=INK, fontsize=10)
    ax2.set_xlabel('diffusion time  $t$   (forward: 0 = clean $\\rightarrow$ 1 = noise)',
                   color=INK, fontsize=10)
    ax2.set_xlim(0, 1)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f'gt_survival_{name}')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', facecolor='white')
    plt.close(fig)

    summary = {'name': name, 'n_pairs': len(idxs), 'draws': draws, 'eta': eta, 's': s,
               'proj_iters': proj, 'n_sparse': int(m), 'shifts': list(map(float, shifts)),
               't_survival_50pct': {f'{b:g}': _cross50(tgrid, acc[bi]) for bi, b in enumerate(shifts)}}
    with open(f'{stem}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    np.savez(f'{stem}.npz', t=tgrid, shifts=np.array(shifts), acc=acc, median_ce=medce)
    return summary, stem


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True, help='config (for dataset + diffusion knobs)')
    p.add_argument('--shifts', type=float, nargs='+', default=[0.0, -1.0, -1.75, -2.5],
                   help='log-SNR shifts b to sweep (nats); 0 = current cosine')
    p.add_argument('--num-pairs', type=int, default=10)
    p.add_argument('--draws', type=int, default=4, help='noise draws averaged per (b, t)')
    p.add_argument('--n-t', type=int, default=26, help='t-grid resolution')
    p.add_argument('--device', default=None, help="'cuda'/'cpu'; auto when omitted")
    p.add_argument('--out', default='figures', help='output dir for the figure + json (default: figures/)')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    summary, stem = run(args.config, args.shifts, args.num_pairs, args.draws, args.n_t,
                        args.device, args.out, args.seed)
    print(f"\n{summary['name']}: n_sparse={summary['n_sparse']}  eta={summary['eta']}  s={summary['s']}")
    print(f"{'shift b':>9} {'t @ 50% survival':>18}")
    print('-' * 30)
    for b, tc in summary['t_survival_50pct'].items():
        print(f"{b:>9} {('%.2f' % tc) if tc is not None else 'never':>18}")
    print(f"\nfigure + arrays: {stem}.png / .pdf / .json / .npz")


if __name__ == '__main__':
    main()
