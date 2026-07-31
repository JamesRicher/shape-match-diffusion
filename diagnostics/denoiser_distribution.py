"""DIAGNOSTIC / VIS -- what distribution does the trained denoiser actually represent?

Draws K samples from the reverse process for a FIXED pair (varying the initial noise, and,
with --eta>0, injecting per-step DDIM noise so independent draws can diverge into different
modes). For each Y-anchor we look at the spread of X-anchors it lands on ACROSS the K samples:

  * unimodal / feature-parroting  -> every row is a spike (one X-anchor always wins);
    across-sample entropy ~0 everywhere. This is the "learned distribution is a
    feature-conditioned delta" outcome (cf. reverse-trajectory-inert): not a failure, the
    crispest confirmation that the denoiser adds no generative prior.
  * multimodal                    -> some rows split mass between competing X-anchors. If the
    competing modes are the SYMMETRIC counterparts (large geodesic separation on X), the prior
    has learned the intrinsic symmetry group -- the interpretable, publishable outcome.

Outputs (figures/denoiser_dist/<name>_pair{idx}_*.png):
  entropy_mesh  -- Y anchors at their mesh positions, coloured by across-sample entropy
                   (ambiguous points light up; check whether they cluster on symmetric regions).
  row_hists     -- X-anchor pick histograms for the highest-entropy rows.
  mode_sep      -- for ambiguous rows, geodesic distance on X between the top-2 modes. A bimodal
                   histogram (a spike near 0 = jitter, a spike at large dist = symmetric flip)
                   tells jitter from genuine symmetry multimodality.

  python -m diagnostics.denoiser_distribution -c configs/joint_diffusionnet/faust_diffusionnet_512_FMD.yaml \
      --pairs 0,5 --k 64 --eta 1.0
  python -m diagnostics.denoiser_distribution -c configs/joint_diffusionnet/smal_diffusionnet_512_FMD_xyz.yaml \
      --pairs 0,3 --k 64 --eta 1.0        # SMAL: where real multimodality is most expected
"""
import argparse
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.base_model import to_numpy
from diagnostics.nn_baseline_dense import _build

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FIG_DIR = os.path.join(ROOT, 'figures', 'denoiser_dist')


def _row_entropy(picks, n_x):
    """Across-sample entropy (nats) per Y-anchor. picks (K, n_y) int X-anchor indices."""
    K, n_y = picks.shape
    ent = np.zeros(n_y)
    for j in range(n_y):
        counts = np.bincount(picks[:, j], minlength=n_x).astype(np.float64)
        p = counts[counts > 0] / K
        ent[j] = -(p * np.log(p)).sum()
    return ent


@torch.no_grad()
def sample_picks(model, data, k, eta):
    """K sampled sparse maps for one pair. Returns picks (K, n_y) X-anchor argmax indices."""
    F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)               # (1, n, .) each
    exp = lambda z: z.expand(k, *z.shape[1:]).contiguous()
    P0 = model.sample(exp(F_x), exp(F_y), exp(D_x), exp(D_y), sample_eta=eta)   # (K, n_y, n_x)
    return to_numpy(P0.argmax(-1))                                   # (K, n_y)


def _entropy_mesh(verts_y, idx_y, ent, path, title):
    """Y anchors at mesh positions, coloured by across-sample entropy over a faint point cloud."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(verts_y[:, 0], verts_y[:, 1], verts_y[:, 2], s=1, c='lightgrey', alpha=0.15)
    a = verts_y[idx_y]
    sc = ax.scatter(a[:, 0], a[:, 1], a[:, 2], s=28, c=ent, cmap='inferno', vmin=0.0)
    fig.colorbar(sc, ax=ax, shrink=0.6, label='across-sample entropy (nats)')
    ax.set_title(title); ax.set_axis_off()
    try: ax.set_box_aspect(np.ptp(verts_y, axis=0))
    except Exception: pass
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _row_hists(picks, ent, n_x, path, title, top=6):
    """X-anchor pick histograms for the highest-entropy rows."""
    rows = np.argsort(ent)[::-1][:top]
    ncol = 3; nrow = int(np.ceil(top / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow), squeeze=False)
    for ax, j in zip(axes.ravel(), rows):
        counts = np.bincount(picks[:, j], minlength=n_x)
        hit = np.nonzero(counts)[0]
        ax.bar(range(len(hit)), counts[hit], color='steelblue')
        ax.set_xticks(range(len(hit))); ax.set_xticklabels(hit, rotation=45, fontsize=7)
        ax.set_title(f'Y-anchor {j}  (H={ent[j]:.2f}, {len(hit)} modes)', fontsize=9)
        ax.set_xlabel('X-anchor'); ax.set_ylabel('count')
    for ax in axes.ravel()[len(rows):]:
        ax.set_axis_off()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _mode_separation(picks, ent, idx_x, dist_x, thresh):
    """For ambiguous rows (entropy>thresh), geodesic dist on X between the two most-picked
    X-anchors. Large => the modes are distant regions (symmetric flip); ~0 => local jitter."""
    seps = []
    for j in np.nonzero(ent > thresh)[0]:
        counts = np.bincount(picks[:, j])
        if np.count_nonzero(counts) < 2:
            continue
        a, b = np.argsort(counts)[::-1][:2]
        seps.append(dist_x[idx_x[a], idx_x[b]])
    return np.array(seps)


def _mode_sep_plot(seps, path, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(seps, bins=30, color='indianred')
    ax.set_xlabel('geodesic separation of top-2 modes on X (area-normalised)')
    ax.set_ylabel('ambiguous rows'); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _involution_test(picks, ent, idx_x, dist_x, ent_thresh, sep_thresh, return_r, seed=0):
    """Is the FAR-mode structure a self-inverse map (a learned intrinsic symmetry) rather than
    off-manifold spread? A symmetry S obeys S(S(a))=a; noise doesn't. The histogram can't tell
    these apart (a true symmetry's partner-distance is itself broad -- small on the axis, large
    at the extremities), so we test the algebra directly.

    Restricted to rows whose top-2 modes are GENUINELY distant (sep>sep_thresh) -- the candidate
    symmetric flips, not local jitter (which passes involution trivially since b~=a). Each such
    row gives a directed pair g: mode1-anchor a -> mode2-anchor b. We then follow g twice, g(g(a)),
    and measure its geodesic return distance to a: ~0 => involution (symmetry); large => not.

    Returns dict with the median return distance, the fraction returning within `return_r`, the
    coverage (how many far-anchors have g(b) defined), and a random-pairing null for scale. None
    when there are too few far-mode rows to test."""
    n_y = picks.shape[1]
    g = {}                                                          # X-anchor a(mode1) -> b(mode2)
    for j in np.nonzero(ent > ent_thresh)[0]:
        counts = np.bincount(picks[:, j])
        if np.count_nonzero(counts) < 2:
            continue
        a, b = np.argsort(counts)[::-1][:2]
        if dist_x[idx_x[a], idx_x[b]] > sep_thresh:                # far modes only (candidate flips)
            g[int(a)] = int(b)
    if len(g) < 5:
        return None

    ret, covered = [], 0                                            # g(g(a)) return distance to a
    for a, b in g.items():
        if b in g:                                                 # need g defined at b to compose
            covered += 1
            ret.append(dist_x[idx_x[g[b]], idx_x[a]])
    if not ret:
        return None
    ret = np.array(ret)

    # null: pair the same far-anchor set at random; an accidental "involution" return distance.
    rng = np.random.default_rng(seed)
    keys = np.array(list(g.keys()))
    perm = rng.permutation(keys)
    rnd = {int(k): int(v) for k, v in zip(keys, perm)}
    rret = [dist_x[idx_x[rnd[rnd[a]]], idx_x[a]] for a in rnd if rnd[a] in rnd]
    return {'n_far': len(g), 'coverage': covered / len(g),
            'median_return': float(np.median(ret)),
            'frac_within_r': float(np.mean(ret < return_r)),
            'null_median_return': float(np.median(rret)) if rret else float('nan')}


def run(config_path, checkpoint, device, fps_metric, pairs, k, eta, ent_thresh):
    model, dataset, opt, ckpt = _build(config_path, checkpoint, device, fps_metric)
    name = opt['name']
    os.makedirs(FIG_DIR, exist_ok=True)
    n_x = dataset[0]['first']['sparse']['idx'].shape[0]

    for idx in pairs:
        data = dataset[idx]
        picks = sample_picks(model, data, k, eta)                   # (K, n_y)
        ent = _row_entropy(picks, n_x)
        verts_y = to_numpy(data['second']['verts'])
        idx_y = to_numpy(data['second']['sparse']['idx'])
        idx_x = to_numpy(data['first']['sparse']['idx'])
        dist_x = to_numpy(data['first']['dist'])
        seps = _mode_separation(picks, ent, idx_x, dist_x, ent_thresh)

        frac_amb = float(np.mean(ent > ent_thresh))
        inv = _involution_test(picks, ent, idx_x, dist_x, ent_thresh, sep_thresh=0.2, return_r=ent_thresh)
        tag = f'{name}_pair{idx}_eta{eta:g}'
        cap = f'{name}  pair {idx}  (K={k}, eta={eta:g})'
        print(f'pair {idx:>3}: mean H={ent.mean():.3f}  max H={ent.max():.3f}  '
              f'ambiguous rows (H>{ent_thresh})={frac_amb*100:.1f}%  '
              f'mode-sep median={np.median(seps) if len(seps) else float("nan"):.3f}')
        if inv is None:
            print('          involution: too few far-mode rows to test (no symmetric-scale multimodality)')
        else:
            print(f'          involution: far-rows={inv["n_far"]} cov={inv["coverage"]*100:.0f}%  '
                  f'g(g(a))->a return={inv["median_return"]:.3f} (within {ent_thresh}: '
                  f'{inv["frac_within_r"]*100:.0f}%)  vs random null={inv["null_median_return"]:.3f}')

        _entropy_mesh(verts_y, idx_y, ent, os.path.join(FIG_DIR, f'{tag}_entropy_mesh.png'),
                      cap + '\nY-anchor across-sample entropy')
        _row_hists(picks, ent, n_x, os.path.join(FIG_DIR, f'{tag}_row_hists.png'),
                   cap + '  -- highest-entropy rows')
        if len(seps):
            _mode_sep_plot(seps, os.path.join(FIG_DIR, f'{tag}_mode_sep.png'),
                           cap + '  -- top-2 mode separation on X')
    print(f'\nfigures -> {FIG_DIR}/')
    if eta == 0.0:
        print("NOTE eta=0: draws differ only in initial noise; near-zero entropy everywhere means "
              "the map is feature-decided (a delta). Re-run with --eta 1.0 to probe true modes.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', required=True)
    p.add_argument('--checkpoint', default=None, help='checkpoint override (e.g. cross-dataset)')
    p.add_argument('--pairs', default='0', help='comma list of dataset pair indices, e.g. 0,5,12')
    p.add_argument('--k', type=int, default=64, help='samples per pair')
    p.add_argument('--eta', type=float, default=1.0,
                   help='DDIM stochasticity: 0=deterministic (delta probe), 1=full DDPM (mode probe)')
    p.add_argument('--ent-thresh', type=float, default=0.1, help='entropy (nats) for an "ambiguous" row')
    p.add_argument('--fps-metric', choices=('config', 'geodesic', 'euclidean'), default='config')
    p.add_argument('--device', default=None)
    args = p.parse_args()
    pairs = [int(x) for x in args.pairs.split(',')]
    run(args.config, args.checkpoint, args.device, args.fps_metric,
        pairs, args.k, args.eta, args.ent_thresh)


if __name__ == '__main__':
    main()
