"""DIAGNOSTIC -- would soft-transporting anchor deltas (P0-weighted) beat the hard argmax
in the densifier? Only worth doing if the FINAL DS matrix P0 carries real, spatially-local
uncertainty. Motivated by the assignment_trajectory finding that cross-dataset P0 keeps a
high row entropy (~3.2 nats), unlike the ~one-hot in-distribution map.

Every densifier consumes the HARD sparse_p2p (each Y anchor -> one X anchor). The proposal
is to instead transport each anchor by its P0 row (a distribution over X anchors). That helps
ONLY when a row's competing mass sits on X anchors CLOSE to the argmax (a within-cell nudge);
if the competitors are far apart (symmetry / limb flip) a soft blend lands between them --
worse than committing. So per Y-anchor row of P0 we report:

  entropy / perplexity      how many X anchors the row effectively spreads over
  top-1 mass, top1-top2 gap how committed the argmax is
  top-k mass                is the spread a few real candidates or a uniform noise floor?
  spread/spacing            E_k[ d_geoX(match_argmax, match_k) ] / median anchor spacing --
                            the DECISIVE metric: <~0.3 = local (soft helps), >~1 = far
                            competitor (soft hurts). Split out for the genuinely-soft rows.

Runs a checkpoint on its config's dataset (cross-dataset when the checkpoint is foreign) so
you can contrast a hard set (FAUST->SCAPE) against the easy in-distribution one.

  python -m diagnostics.soft_transport_impact                       # FAUST model on SCAPE
  python -m diagnostics.soft_transport_impact \
      -c configs/joint_diffusionnet/faust_diffusionnet_512_FMD_snrfd_tw_gt.yaml \
      --checkpoint experiments/diffusionnet/faust_diffusionnet_512_FMD_snrfd_tw_gt/models/final.pth
"""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from diagnostics.assignment_trajectory import _build, FIG_DIR as _TRAJ_DIR  # noqa: F401
from utils.sinkhorn import cosine_alpha_bar, log_sinkhorn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG_DIR = os.path.join(ROOT, 'figures', 'soft_transport')
GRID = dict(color='0.85', lw=0.6)
SOFT_THRESH = 0.9        # a row is "genuinely soft" if top-1 mass < this
LOCAL_FRAC = 0.3         # spread/spacing below this = a within-cell nudge


@torch.no_grad()
def reverse_logits(model, F_x, F_y, D_x, D_y, steps=None):
    """Run the deterministic DDIM reverse process and return the FINAL logits u (B,n,n),
    which model.sample discards. Faithful copy of MatrixDiffusionModel.sample's loop -- read
    only, touches no training code. Re-projecting u at different tau is how we test sharpening
    the final read-out WITHOUT changing the trained model or its sampler."""
    steps = steps or model.sample_steps
    net = model.networks['denoiser']
    B, n = F_x.shape[0], F_x.shape[1]
    u = torch.randn(B, n, n, device=model.device)
    ts = torch.linspace(1.0, 0.0, steps + 1, device=model.device)
    for i in range(steps):
        t_i, t_prev = ts[i], ts[i + 1]
        P_t = log_sinkhorn(u, n_iters=model.proj_iters).exp()
        u0_hat = net(P_t, F_x, F_y, D_x, D_y, t_i.reshape(1).expand(B))
        ab_t = cosine_alpha_bar(t_i, model.schedule_s, model.logsnr_shift)
        ab_p = cosine_alpha_bar(t_prev, model.schedule_s, model.logsnr_shift)
        eps_hat = (u - ab_t.sqrt() * u0_hat) / (1.0 - ab_t).clamp_min(1e-8).sqrt()
        u = ab_p.sqrt() * u0_hat + (1.0 - ab_p).clamp_min(0.0).sqrt() * eps_hat
    return u


@torch.no_grad()
def row_stats(P0, D_x, gt_col, top_c):
    """Per-row concentration + spatial-spread stats for one pair. P0 (n_y,n_x) DS matrix,
    D_x (n_x,n_x) sparse-X geodesic, gt_col (n_y,) GT source column.

    A Sinkhorn DS row has a near-uniform floor smeared over all n_x columns, so full-row
    top-1 mass / entropy are floor-dominated and misleading. A soft densifier only ever uses
    the top few candidates (row_stochastic keeps top_c=8), so commitment and spatial spread
    are measured on the top-c columns RENORMALISED to sum to 1 -- that is what soft transport
    would actually blend. `floor_mass` (mass outside top-c) is reported separately so the
    floor is visible, not hidden."""
    P = P0 / P0.sum(1, keepdim=True).clamp_min(1e-12)
    n = P.shape[0]
    ent = -(P * P.clamp_min(1e-12).log()).sum(1)                   # (n_y,) full-row nats (floor-inflated)
    srt, idx = P.sort(1, descending=True)
    am = idx[:, 0]                                                 # argmax X anchor per row
    correct = (am.cpu().numpy() == gt_col)

    c_mass = srt[:, :top_c]                                        # (n_y, top_c) real candidates
    floor_mass = 1.0 - c_mass.sum(1)                              # mass in the uniform noise floor
    r = c_mass / c_mass.sum(1, keepdim=True).clamp_min(1e-12)     # renormalised over top-c
    top1_c, top2_c = r[:, 0], r[:, 1]
    perp_c = torch.exp(-(r * r.clamp_min(1e-12).log()).sum(1))    # eff. real candidates (<= top_c)
    # geodesic spread of the real candidates around the argmax match, floor removed
    cand_d = torch.gather(D_x[am], 1, idx[:, :top_c])            # (n_y, top_c) d(argmax, cand)
    spread = (r * cand_d).sum(1)                                  # X-geodesic units
    spacing = D_x[D_x > 0].reshape(n, n - 1).min(1).values.median()  # median nearest-anchor gap
    return dict(ent=ent.cpu().numpy(), floor=floor_mass.cpu().numpy(),
                top1=top1_c.cpu().numpy(), gap=(top1_c - top2_c).cpu().numpy(),
                perp=perp_c.cpu().numpy(), spread=(spread / spacing).cpu().numpy(),
                correct=correct)


def summarize(tag, S, top_c):
    """Print the headline numbers deciding whether soft transport is worth wiring in.
    All commitment/spread stats are on the top-c renormalised candidates (floor removed)."""
    n = len(S['top1'])
    soft = S['top1'] < SOFT_THRESH
    soft_local = soft & (S['spread'] < LOCAL_FRAC)
    soft_far = soft & (S['spread'] >= LOCAL_FRAC)
    print(f'\n=== {tag}  ({n} anchor rows over {S["n_pairs"]} pairs, top_c={top_c}) ===')
    print(f'  noise floor      median {100*np.median(S["floor"]):.1f}% of row mass outside top-{top_c} '
          f'(full-row entropy {np.median(S["ent"]):.2f} nats is floor-inflated)')
    print(f'  among top-{top_c} candidates (renormalised, what soft transport would blend):')
    print(f'    commitment top-1  median {np.median(S["top1"]):.3f}   '
          f'eff. real candidates median {np.median(S["perp"]):.1f}')
    print(f'    hard rows (top1>{SOFT_THRESH})     {100*np.mean(~soft):5.1f}%  -> soft == hard here')
    print(f'    genuinely-soft rows         {100*np.mean(soft):5.1f}%')
    print(f'      LOCAL (<{LOCAL_FRAC} spacing) {100*np.mean(soft_local):5.1f}% of all  '
          f'-> useful within-cell nudge (soft HELPS)')
    print(f'      FAR  (>= {LOCAL_FRAC})       {100*np.mean(soft_far):5.1f}% of all  '
          f'-> competitors far apart -> blend between modes (soft HURTS)')
    if soft.any():
        hard = ~soft
        hacc = 100*np.mean(S['correct'][hard]) if hard.any() else float('nan')
        print(f'    argmax-acc: soft rows {100*np.mean(S["correct"][soft]):.1f}%  '
              f'vs committed rows {hacc:.1f}%  (is uncertainty where the errors are?)')


def plots(tag, S, path_prefix):
    # top-1 mass histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(S['top1'], bins=40, range=(0, 1), color='#2a78d6', alpha=0.85)
    ax.axvline(SOFT_THRESH, color='#d62728', ls='--', lw=1.5, label=f'soft threshold {SOFT_THRESH}')
    ax.set_xlabel('top-1 mass among top-c candidates (1.0 = committed -> soft==hard)')
    ax.set_ylabel('anchor rows'); ax.set_title(f'{tag}: how committed is each anchor?')
    ax.legend(frameon=False); ax.grid(True, **GRID); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f'{path_prefix}_top1.png', dpi=150); plt.close(fig)

    # decisive 2D view: commitment vs spatial spread of the competing mass
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(S['top1'], S['spread'], s=6, c=S['correct'], cmap='RdYlGn',
                    alpha=0.4, linewidths=0, vmin=0, vmax=1)
    ax.axhline(LOCAL_FRAC, color='0.4', ls='--', lw=1, label=f'local/far split ({LOCAL_FRAC})')
    ax.axvline(SOFT_THRESH, color='0.4', ls=':', lw=1)
    ax.set_yscale('symlog', linthresh=0.1)
    ax.set_xlabel('top-1 mass (commitment)')
    ax.set_ylabel('competitor spread / anchor spacing  (log)')
    ax.set_title(f'{tag}: soft-transport impact map\n'
                 'bottom-left = soft & local (helps); upper-left = soft & far (hurts)')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, **GRID); ax.set_axisbelow(True)
    cb = fig.colorbar(sc, ax=ax, ticks=[0, 1]); cb.ax.set_yticklabels(['wrong', 'correct'])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f'{path_prefix}_impactmap.png', dpi=150); plt.close(fig)


def tau_sweep_plot(tag, taus, rows, path):
    """How the top-c row picture moves as the final read-out is sharpened (tau down)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(taus))
    ax.plot(x, [r['top1'] for r in rows], color='#2a78d6', marker='o', lw=2, label='top-1 commitment (median)')
    ax.plot(x, [r['floor'] for r in rows], color='#8a2be2', marker='v', lw=2, label='noise-floor mass (median)')
    ax.plot(x, [r['local'] for r in rows], color='#008300', marker='s', lw=2, label='rows LOCAL (soft helps)')
    ax.plot(x, [r['acc'] for r in rows], color='#d62728', marker='^', lw=2, ls='--', label='argmax accuracy')
    ax.set_xticks(x); ax.set_xticklabels([f'{t:g}' for t in taus])
    ax.set_xlabel('final-projection temperature tau  (baseline 1.0 -> sharper)')
    ax.set_ylabel('fraction'); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f'{tag}: does sharpening the read-out make rows local & confident?')
    ax.grid(True, **GRID); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_tau_sweep(config_path, checkpoint, device, pairs, top_c, taus):
    """Re-project each pair's final logits u at several taus and track whether sharpening
    yields genuinely LOCAL low-entropy rows (soft transport becomes useful) or just collapses
    to one-hot (soft == hard again). The trained model / sampler are untouched."""
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device)
    tag = f"{opt['name'].split('_')[0]}model_on_{opt['datasets']['test']['name']}"
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f'model: {opt["name"]}   eval: {opt["datasets"]["test"]["name"]}   tau sweep {taus}')

    us, D_xs, gts = [], [], []
    for idx in tqdm(pairs, desc='sample'):
        data = dataset[idx]
        F_x, F_y, D_x, D_y, P0gt = model._sparse_inputs(data)
        assert P0gt is not None, 'need bijective sparse GT; use the test split'
        us.append(reverse_logits(model, F_x, F_y, D_x, D_y))
        D_xs.append(D_x[0]); gts.append(P0gt[0].cpu().numpy().argmax(1))

    print(f'\n{tag}  (top_c={top_c}, {len(pairs)} pairs)')
    print(f'{"tau":>6} {"top1_c":>8} {"floor%":>8} {"eff_cand":>9} {"%local":>8} {"argmax_acc":>11}')
    print('-' * 54)
    rows = []
    for tau in taus:
        agg = {k: [] for k in ('top1', 'floor', 'perp', 'spread', 'correct')}
        for u, D_x, gt in zip(us, D_xs, gts):
            P = log_sinkhorn(u, n_iters=model.final_iters, tau=tau).exp()[0]
            S = row_stats(P, D_x, gt, top_c)
            for k in agg:
                agg[k].append(S[k])
        S = {k: np.concatenate(v) for k, v in agg.items()}
        local = float(np.mean((S['top1'] < SOFT_THRESH) & (S['spread'] < LOCAL_FRAC)))
        rec = dict(top1=float(np.median(S['top1'])), floor=float(np.median(S['floor'])),
                   perp=float(np.median(S['perp'])), local=local,
                   acc=float(np.mean(S['correct'])))
        rows.append(rec)
        print(f'{tau:>6g} {rec["top1"]:>8.3f} {100*rec["floor"]:>7.1f}% {rec["perp"]:>9.1f} '
              f'{100*local:>7.1f}% {100*rec["acc"]:>10.1f}%')
    tau_sweep_plot(tag, taus, rows, os.path.join(FIG_DIR, f'{tag}_tausweep.png'))
    print(f'\nfigure -> {FIG_DIR}/{tag}_tausweep.png')


def run(config_path, checkpoint, device, pairs, draws, top_c):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device)
    tag = f"{opt['name'].split('_')[0]}model_on_{opt['datasets']['test']['name']}"
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f'model: {opt["name"]}   checkpoint: {ckpt}\neval dataset: {opt["datasets"]["test"]["name"]}')

    acc = {k: [] for k in ('ent', 'floor', 'perp', 'top1', 'gap', 'spread', 'correct')}
    for idx in tqdm(pairs, desc='pairs'):
        data = dataset[idx]
        F_x, F_y, D_x, D_y, P0gt = model._sparse_inputs(data)
        assert P0gt is not None, 'need bijective sparse GT; use the test split'
        gt_col = P0gt[0].cpu().numpy().argmax(1)
        for _ in range(draws):
            P0 = model.sample(F_x, F_y, D_x, D_y)[0]               # (n_y, n_x) final DS matrix
            S = row_stats(P0, D_x[0], gt_col, top_c)
            for k in acc:
                acc[k].append(S[k])
    S = {k: np.concatenate(v) for k, v in acc.items()}
    S['n_pairs'] = len(pairs) * draws
    summarize(tag, S, top_c)
    plots(tag, S, os.path.join(FIG_DIR, tag))
    print(f'\nfigures -> {FIG_DIR}/{tag}_*.png')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', default='configs/joint_diffusionnet/scape_diffusionnet_512_FMD_snrfd_tw_gt.yaml',
                   help='config supplying the eval dataset + arch (default: SCAPE = the hard set)')
    p.add_argument('--checkpoint',
                   default='experiments/diffusionnet/faust_diffusionnet_512_FMD_snrfd_tw_gt/models/final.pth')
    p.add_argument('--pairs', default='0,1,2,3,4,5', help='comma list of eval pair indices')
    p.add_argument('--draws', type=int, default=1, help='independent prior draws per pair (DDIM is det. per init)')
    p.add_argument('--top-c', type=int, default=8, help='candidates a soft densifier keeps per row (row_stochastic=8)')
    p.add_argument('--tau-sweep', default=None,
                   help='comma taus to re-project the final logits at (e.g. 1.0,0.5,0.25,0.1,0.05); '
                        'runs the sharpening test instead of the baseline analysis')
    p.add_argument('--device', default=None)
    args = p.parse_args()
    pairs = [int(x) for x in args.pairs.split(',')]
    if args.tau_sweep:
        run_tau_sweep(args.config, args.checkpoint, args.device, pairs, args.top_c,
                      [float(x) for x in args.tau_sweep.split(',')])
    else:
        run(args.config, args.checkpoint, args.device, pairs, args.draws, args.top_c)


if __name__ == '__main__':
    main()
