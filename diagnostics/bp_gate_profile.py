"""Plot the learned per-block BP gate schedules from a trained checkpoint.

Every BPBlock owns three learned quantities, all driven by the conditioning spine's c
and therefore all functions of the diffusion timestep t (networks/mpnn/bp_block.py):

    beta(t) = softplus(beta_head(c))        unary scale on the residual field
    g(t)    = g_scale * g_head(c)           output gate on the BP write
    v(t), b(t) = gate_head(c)               agreement gate w = 1 + tanh(0.5*(v*z + b))

Within one denoiser forward all depth blocks receive the SAME c, so whatever they have
learned to differentiate is specialisation by depth position only. This script draws the
curves and scores how redundant they are, which is the evidence for or against collapsing
the stack to a single BP call per denoiser forward (one per DDIM step at inference).

Read it as: similar g(t) curves across blocks => the applications repeat each other and
collapsing loses nothing; distinct curves => depth specialisation is real and an eval
ablation should come before discarding it.

Usage:
    python -m diagnostics.bp_gate_profile experiments/mpnn_diffusion/<run>/models/final.pth
    python -m diagnostics.bp_gate_profile <ckpt> --g-scale 4.0 --out /tmp/gates.png
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from networks.denoiser_conditioning import ConditioningSpine


def load_denoiser_sd(ckpt_path):
    """Denoiser state dict from a saved joint checkpoint."""
    return torch.load(ckpt_path, map_location='cpu')['networks']['denoiser']


def load_run_config(ckpt_path):
    """The run's config from experiment_info.json beside the models/ dir, or None."""
    info = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), 'experiment_info.json')
    if not os.path.isfile(info):
        return None
    with open(info) as fh:
        return json.load(fh).get('config')


def spine_from_sd(sd):
    """Rebuild the conditioning spine alone, so no data or geometry is needed."""
    dim = sd['spine.t_mlp.0.weight'].shape[0]
    spine = ConditioningSpine(dim)
    spine.load_state_dict({k[len('spine.'):]: v for k, v in sd.items() if k.startswith('spine.')})
    return spine.eval()


def block_prefixes(sd):
    """BP parameter prefixes, one per BP call, oldest-layout-first.

    Three layouts exist: the current per-site stage (`bp.sites.{i}.*`, one entry per block
    named in at_block -- usually just site 0), the single-site stage it replaced
    (`bp.block.*`), and the superseded per-block stack (`bp.blocks.{i}.*`), which older
    trained checkpoints still use. Site index i is a position in at_blocks, NOT a trunk
    block index; the run's config says which block each maps to.
    """
    idx = sorted({int(k.split('.')[2]) for k in sd if k.startswith('bp.sites.')})
    if idx:
        return [f'bp.sites.{i}' for i in idx]
    if any(k.startswith('bp.block.') for k in sd):
        return ['bp.block']
    idx = sorted({int(k.split('.')[2]) for k in sd if k.startswith('bp.blocks.')})
    return [f'bp.blocks.{i}' for i in idx]


@torch.no_grad()
def gate_curves(sd, t_grid, g_scale, beta_bounds=None):
    """Evaluate every BP call's beta, g and agreement-gate coefficients over t.

    Each head is Sequential(SiLU, Linear), so the linear layer reads SiLU(c) and the
    stored weights alone reproduce the schedule exactly.

    beta has two parameterisations: the current bounded log-space sigmoid
    (`beta_min·(beta_max/beta_min)^sigmoid`, used when `beta_bounds` is given) and the
    superseded unbounded softplus. The agreement gate likewise has two: the current
    slope-only head (out=1, no bias — the bias was a global off-switch), and the old
    (v, b) head. `b` is reported as zero for the current layout, which is exactly what
    it now is.
    """
    c = F.silu(spine_from_sd(sd)(t_grid))                       # (T, dim)
    out = {k: [] for k in ('beta', 'g', 'v', 'b')}
    for pre in block_prefixes(sd):
        p = lambda h, s: sd[f'{pre}.{h}.1.{s}']
        raw = c @ p('beta_head', 'weight').T + p('beta_head', 'bias')
        if beta_bounds is None:
            beta = F.softplus(raw)
        else:
            lo, hi = beta_bounds
            beta = lo * (hi / lo) ** torch.sigmoid(raw)
        g = g_scale * (c @ p('g_head', 'weight').T + p('g_head', 'bias'))
        out['beta'].append(beta.squeeze(-1))
        out['g'].append(g.squeeze(-1))
        if f'{pre}.gate_head.1.weight' in sd:
            vb = c @ p('gate_head', 'weight').T + p('gate_head', 'bias')
            out['v'].append(vb[:, 0])
            out['b'].append(vb[:, 1] if vb.shape[-1] > 1 else torch.zeros_like(vb[:, 0]))
        else:                                    # cycle_gate off: w ≡ 1 identically
            out['v'].append(torch.zeros_like(g.squeeze(-1)))
            out['b'].append(torch.zeros_like(g.squeeze(-1)))
    return {k: torch.stack(v).numpy() for k, v in out.items()}


def effective_g(cur):
    """The write that actually lands: g scaled by the agreement gate at the average vertex.

    Reading `g` alone overstates the write wherever the old gate's bias `b` was driving
    w = 1 + tanh(b/2) toward zero — which it was, in every checkpoint of the per-block
    stack (b in [-8, -2], removing 29-66% of the raw write). Always judge writes by this,
    never by `g`. For the current layout b ≡ 0 and this reduces to `g`.
    """
    return cur['g'] * (1.0 + np.tanh(0.5 * cur['b']))


def redundancy(g):
    """How alike the per-block g(t) curves are: pairwise correlation and magnitude spread.

    Correlation compares SHAPE (does every block ramp the same way in t); the magnitude
    ratio compares how much each block actually writes. Both must be high/tight for the
    blocks to count as repetitions of each other. Returns the full (n, n) correlation
    matrix so redundant SUBSETS are visible, not just the global mean.
    """
    n = g.shape[0]
    cen = g - g.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(cen, axis=1)
    C = (cen @ cen.T) / np.clip(np.outer(nrm, nrm), 1e-12, None)
    return C, np.abs(g).max(axis=1)


def cancellation(g):
    """Per-t ratio |sum_i g_i| / sum_i |g_i|: 1 = blocks pull together, 0 = they cancel.

    The write magnitudes are not directly comparable across blocks (each multiplies its
    own Delta), so this is an indicator, not the net write. A persistently low value
    still means the stack holds large opposing gates whose near-cancellation nothing
    constrains -- the fragile configuration.
    """
    return np.abs(g.sum(axis=0)) / np.clip(np.abs(g).sum(axis=0), 1e-12, None)


def plot(t, cur, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    n = cur['g'].shape[0]
    colors = plt.cm.viridis(np.linspace(0, 0.9, n))

    for i in range(n):
        axes[0].plot(t, cur['g'][i], color=colors[i], label=f'block {i}')
        axes[1].plot(t, cur['beta'][i], color=colors[i])
        axes[2].plot(t, cur['v'][i], color=colors[i])
        axes[2].plot(t, cur['b'][i], color=colors[i], ls='--', alpha=0.6)

    axes[0].set_title('output gate  g(t) = g_scale * g_head(c)\n(how much BP is written)')
    axes[0].set_ylabel('g')
    axes[0].axhline(0, color='k', lw=0.6, ls=':')
    axes[1].set_title('unary scale  beta(t) = softplus(beta_head(c))\n(how much BP trusts the state)')
    axes[1].set_ylabel('beta')
    axes[2].set_title('agreement gate coeffs (solid v, dashed b)\nw = 1 + tanh(0.5*(v*z + b))')
    axes[2].set_ylabel('v, b')
    axes[2].axhline(0, color='k', lw=0.6, ls=':')
    for ax in axes:
        ax.set_xlabel('diffusion time t   (1 = noise, 0 = clean)')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def report(t, cur):
    """Print the g(t) table plus the depth-redundancy verdict."""
    g = cur['g']
    show = np.linspace(0, len(t) - 1, 6).astype(int)
    print('\n=== output gate g(t) per block ===')
    print('block | ' + ' | '.join(f't={t[k]:.2f}' for k in show) + ' |    peak |g|')
    for i in range(g.shape[0]):
        print(f'{i:>5} | ' + ' | '.join(f'{g[i, k]:+7.3f}' for k in show)
              + f' | {np.abs(g[i]).max():9.3f}')

    print('\n=== beta(t) per block ===')
    print('block | ' + ' | '.join(f't={t[k]:.2f}' for k in show))
    for i in range(cur['beta'].shape[0]):
        print(f'{i:>5} | ' + ' | '.join(f'{cur["beta"][i, k]:7.3f}' for k in show))

    eff = effective_g(cur)
    if g.shape[0] == 1:                 # current layout: one BP call, nothing to compare
        print('\n=== single BP call (bp.at_block) ===')
        print(f'peak |g| {np.abs(g[0]).max():.3f}   peak effective write '
              f'{np.abs(eff[0]).max():.3f}   sign: '
              f"{'positive' if g[0].max() > 0 else 'NEGATIVE (BP is being subtracted)'}")
        print(f'beta range over t: [{cur["beta"][0].min():.3f}, {cur["beta"][0].max():.3f}]')
        return

    C, peak = redundancy(g)
    n = g.shape[0]
    off = C[~np.eye(n, dtype=bool)]
    print('\n=== pairwise shape correlation of g(t) ===')
    print('      ' + ' '.join(f'{j:>6}' for j in range(n)))
    for i in range(n):
        print(f'{i:>5} ' + ' '.join(f'{C[i, j]:+6.2f}' for j in range(n)))

    print('\n=== depth redundancy ===')
    print(f'off-diagonal correlation:  mean {off.mean():+.3f}   '
          f'min {off.min():+.3f}   max {off.max():+.3f}')
    print('peak |g| per block: ' + '  '.join(f'{v:.3f}' for v in peak))
    print(f'  spread max/min = {peak.max() / max(peak.min(), 1e-9):.2f}x   '
          f'sum over blocks = {peak.sum():.3f}')

    canc = cancellation(g)
    print(f'cancellation |sum g| / sum |g|:  mean {canc.mean():.3f}   '
          f'min {canc.min():.3f}  (1 = aligned, 0 = opposing)')
    n_neg = int((g.max(axis=1) < 0).sum())
    print(f'blocks with g < 0 at every t: {n_neg}  (these SUBTRACT the BP write)')

    # The write that actually lands. Judging by g alone is what hid the real story on
    # the per-block stack: the agreement gate's bias had switched most blocks off, so
    # both the "specialisation" and the apparent opposition sat on blocks writing ~0.
    peak_eff = np.abs(eff).max(axis=1)
    top = peak_eff.argmax()
    print('\n=== effective write  g*(1 + tanh(b/2))  [judge by THIS, not g] ===')
    print('peak per block: ' + '  '.join(f'{v:.3f}' for v in peak_eff))
    print(f'  sum {peak_eff.sum():.3f} vs raw {peak.sum():.3f} '
          f'({100 * (1 - peak_eff.sum() / max(peak.sum(), 1e-9)):.0f}% removed by the gate)   '
          f'blocks < 0.05: {int((peak_eff < 0.05).sum())}/{n}')
    print(f'  top block {top} carries {100 * peak_eff[top] / max(peak_eff.sum(), 1e-9):.0f}%'
          ' of the effective write')

    canc_eff = cancellation(eff)
    print(f'  effective cancellation: mean {canc_eff.mean():.3f} (vs {canc.mean():.3f} raw)')

    if peak_eff[top] / max(peak_eff.sum(), 1e-9) > 0.5:
        print(f'VERDICT: the stack has collapsed itself onto block {top} -> one BP call '
              'there loses little. This is what bp.at_block now does.')
    elif off.mean() > 0.9 and peak.max() / max(peak.min(), 1e-9) < 3.0:
        print('VERDICT: blocks agree in shape and magnitude -> largely redundant, '
              'collapsing to one BP call per forward should lose little.')
    else:
        print('VERDICT: blocks differ in EFFECTIVE write -> depth specialisation may be '
              'real; check whether it REPLICATES across runs before treating it as signal.')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('ckpt', help='saved checkpoint (final.pth / latest.pth)')
    ap.add_argument('--g-scale', type=float, default=None,
                    help='override the config g_scale (default: read from experiment_info.json, else 4.0)')
    ap.add_argument('--steps', type=int, default=201, help='points on the t grid')
    ap.add_argument('--out', default=None, help='figure path (default: diagnostics/results/...)')
    args = ap.parse_args()

    sd = load_denoiser_sd(args.ckpt)
    if not block_prefixes(sd):
        raise SystemExit('no bp.block/bp.blocks.* in this checkpoint -- trained without BP')

    cfg = load_run_config(args.ckpt)
    bp = ((cfg or {}).get('networks', {}).get('denoiser', {}) or {}).get('bp') or {}
    g_scale = args.g_scale if args.g_scale is not None else bp.get('g_scale', 4.0)
    # bounded beta only if the run configured it; older runs used the unbounded softplus
    beta_bounds = ((bp['beta_min'], bp['beta_max'])
                   if 'beta_min' in bp and 'beta_max' in bp else None)
    name = (cfg or {}).get('name') or os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))

    t = torch.linspace(1.0, 0.0, args.steps)
    cur = gate_curves(sd, t, g_scale, beta_bounds)
    report(t.numpy(), cur)

    out = args.out or os.path.join('diagnostics', 'results', f'bp_gate_profile_{name}.png')
    print('\nwrote ' + plot(t.numpy(), cur, f'{name}  (g_scale={g_scale})', out))


if __name__ == '__main__':
    main()
