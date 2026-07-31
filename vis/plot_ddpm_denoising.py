"""Figure for the DDPM background section: a pointwise map being progressively denoised.

Renders a left-to-right strip of the reverse process. Top row is the assignment matrix
P_t = Pi_S(u_t); bottom row is the same thing drawn as a bipartite pointwise map, edge
opacity proportional to P_t[i, j]. Noise on the left (t=1), clean map on the right (t=0).

Two modes:

  schematic (default) -- no model. Evaluates the forward marginal
    u_t = sqrt(ab)*u0 + sqrt(1-ab)*eps at decreasing t with a single fixed eps, so the
    panels are visually coherent and converge to the ground-truth permutation. That is
    what a PERFECT denoiser would trace out. Use it as a background-section illustration
    of DDPM; do not caption it as model output.

  real (--config ...) -- loads a trained checkpoint and snapshots P_t during the actual
    DDIM reverse process of MatrixDiffusionModel.sample. Genuine model output, so it
    shows real intermediate errors. The full P_t is n_sparse x n_sparse (e.g. 512); only
    an --n-point random subset of the FPS points is displayed, with rows renormalised
    over that subset (mass leaving the subset is not drawn). Sparse points are indexed
    consistently across the pair, so the CORRECT map is the diagonal.

Usage:
    python -m vis.plot_ddpm_denoising                               # schematic
    python -m vis.plot_ddpm_denoising --n 20 --perm identity --rows matrix
    python -m vis.plot_ddpm_denoising --config configs/<...>.yaml   # trained model
    python -m vis.plot_ddpm_denoising --config configs/<...>.yaml \
        --checkpoint experiments/<group>/<run>/models/final.pth --pair 3 --sample-eta 1.0
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from paths import REPO_ROOT
from utils.sinkhorn import cosine_alpha_bar, log_sinkhorn, logit_target, q_sample

OUT_DIR = Path(REPO_ROOT) / "figures"

# single-hue sequential ramp, light -> dark (magnitude encoding)
CMAP = LinearSegmentedColormap.from_list("ddpm_blue", ["#ffffff", "#c6dbef", "#4a90c4", "#123c5e"])
INK = "#1b2733"
MUTED = "#7a8794"


def build_matrices(n: int, times, perm: str, eta: float, tau: float, seed: int):
    """P_t for each t in times, sharing one noise draw. Returns (T, n, n) numpy."""
    g = torch.Generator().manual_seed(seed)

    idx = torch.arange(n) if perm == "identity" else torch.randperm(n, generator=g)
    P0 = torch.zeros(n, n)
    P0[torch.arange(n), idx] = 1.0

    u0 = logit_target(P0, eta=eta)
    eps = torch.randn(n, n, generator=g)
    # match the noise scale to the target's, else the clean end dominates every panel
    eps = eps * u0.std()

    mats = []
    for t in times:
        u_t = q_sample(u0, torch.tensor(float(t)), noise=eps)
        mats.append(log_sinkhorn(u_t, n_iters=20, tau=tau).exp().numpy())
    return np.stack(mats)


def load_model(config_path, checkpoint, split, device):
    """Load a trained MatrixDiffusionModel + its dataset, as evaluate.py / sample_select do."""
    # imported lazily so the schematic mode needs nothing but torch + matplotlib
    from datasets import build_dataset
    from models import build_model
    from train import autofill_feat_dim
    from utils.options import load_yaml, resolve_experiment_paths

    opt = load_yaml(config_path)
    if device is not None:
        opt['device'] = device
    opt['is_train'] = False
    resolve_experiment_paths(opt)
    ckpt = checkpoint or str(Path(opt['path']['models']) / 'final.pth')
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f'checkpoint not found: {ckpt}\nTrain first, or pass --checkpoint.')
    opt['path']['resume_state'] = ckpt
    opt['path']['resume'] = False
    dataset = build_dataset(opt['datasets'][split])
    autofill_feat_dim(opt, int(dataset[0]['first']['feat'].shape[-1]))
    model = build_model(opt)
    model.eval()
    return model, dataset, ckpt


@torch.no_grad()
def sample_trajectory(model, data, times, sample_eta=0.0):
    """Run the model's DDIM reverse process, snapshotting P_t at the requested times.

    Mirrors MatrixDiffusionModel.sample (which returns only hard maps for its trajectory).
    Each requested t is served by the sampler step whose t_i is nearest; t=0 is the final
    converged DS matrix. Returns (mats (T,n,n) numpy, actual t values).
    """
    F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)
    net = model.networks['denoiser']
    B, n = F_x.shape[0], F_x.shape[1]
    u = torch.randn(B, n, n, device=model.device)
    ts = torch.linspace(1.0, 0.0, model.sample_steps + 1, device=model.device)

    # nearest sampler step for each requested t; t<=0 is served by the final DS matrix
    grid = ts[:-1].cpu().numpy()
    wanted = {}
    for t in times:
        step = None if t <= 0.0 else int(np.abs(grid - t).argmin())
        wanted.setdefault(step, []).append(float(t))

    snaps, actual = {}, {}
    for i in range(model.sample_steps):
        t_i, t_prev = ts[i], ts[i + 1]
        P_t = log_sinkhorn(u, n_iters=model.proj_iters).exp()   # the read-in the denoiser sees
        if i in wanted:
            snaps[i] = P_t[0].cpu().numpy()
            actual[i] = float(t_i)
        u0_hat = net(P_t, F_x, F_y, D_x, D_y, t_i.reshape(1).expand(B))

        # honour the model's log-SNR shift so the reverse dynamics match training/sample() exactly
        # (unshifted here would mis-schedule a snrfd model, e.g. logsnr_shift=-1.75)
        shift = getattr(model, 'logsnr_shift', 0.0)
        ab_t = cosine_alpha_bar(t_i, model.schedule_s, shift)
        ab_p = cosine_alpha_bar(t_prev, model.schedule_s, shift)
        eps_hat = (u - ab_t.sqrt() * u0_hat) / (1.0 - ab_t).clamp_min(1e-8).sqrt()
        sigma = sample_eta * ((1.0 - ab_p) / (1.0 - ab_t).clamp_min(1e-8)).sqrt() \
                * (1.0 - ab_t / ab_p.clamp_min(1e-8)).clamp_min(0.0).sqrt()
        u = ab_p.sqrt() * u0_hat + ((1.0 - ab_p) - sigma ** 2).clamp_min(0.0).sqrt() * eps_hat
        if sample_eta > 0.0:
            u = u + sigma * torch.randn_like(u)

    snaps[None] = log_sinkhorn(u, n_iters=model.final_iters).exp()[0].cpu().numpy()
    actual[None] = 0.0

    mats, out_t = [], []
    for t in times:
        key = next(k for k, v in wanted.items() if float(t) in v)
        mats.append(snaps[key])
        out_t.append(actual[key])
    return np.stack(mats), np.asarray(out_t)


def subset_matrices(mats, n_show, seed):
    """Cut an n_show x n_show block of FPS points out of the full P_t and renormalise rows.

    Sparse points are index-aligned across the pair, so keeping the same index set on rows
    and columns preserves the diagonal = correct-match reading. Rows are renormalised over
    the block; probability mapped outside the block is not shown.
    """
    n = mats.shape[-1]
    if n_show >= n:
        return mats
    sel = np.sort(np.random.default_rng(seed).choice(n, size=n_show, replace=False))
    sub = mats[:, sel][:, :, sel]
    return sub / sub.sum(axis=-1, keepdims=True).clip(1e-12)


def draw_matrix(ax, P, title, subtitle):
    ax.imshow(P, cmap=CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#d5dbe1")
        s.set_linewidth(0.8)
    ax.set_title(title, fontsize=11, color=INK, pad=5)
    ax.set_xlabel(subtitle, fontsize=8.5, color=MUTED, labelpad=4)


def draw_map(ax, P, thresh=0.04):
    """Bipartite pointwise map: source column left, target column right."""
    n = P.shape[0]
    ys = np.linspace(1.0, 0.0, n)
    rows, cols = np.nonzero(P > thresh)
    order = np.argsort(P[rows, cols])  # strong edges drawn last
    for i, j in zip(rows[order], cols[order]):
        w = float(P[i, j])
        ax.plot([0.0, 1.0], [ys[i], ys[j]], color="#2a6ea6",
                alpha=min(1.0, 0.05 + 0.95 * w ** 0.8), linewidth=0.3 + 1.5 * w, zorder=1,
                solid_capstyle="round")
    ax.scatter(np.zeros(n), ys, s=11, color=INK, zorder=3)
    ax.scatter(np.ones(n), ys, s=11, color=INK, zorder=3)
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=14, help="number of points in the map")
    ap.add_argument("--steps", type=int, default=5, help="number of panels")
    ap.add_argument("--times", type=float, nargs="+", default=None,
                    help="explicit t values, noisy first (overrides --steps)")
    ap.add_argument("--perm", choices=["random", "identity"], default="random")
    ap.add_argument("--rows", choices=["both", "matrix", "map"], default="both")
    ap.add_argument("--eta", type=float, default=0.1, help="logit_target smoothing")
    ap.add_argument("--tau", type=float, default=1.0, help="Sinkhorn temperature")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--name", default="ddpm_denoising")
    ap.add_argument("--caption", default=None, help="arrow caption override")
    # real-model mode
    ap.add_argument("--config", default=None,
                    help="training config; switches to real sampler output from a checkpoint")
    ap.add_argument("--checkpoint", default=None, help="checkpoint override (default: <run>/models/final.pth)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pair", type=int, default=0, help="dataset pair index to sample")
    ap.add_argument("--sample-eta", type=float, default=0.0,
                    help="DDIM<->DDPM stochasticity (0 = deterministic DDIM, 1 = full DDPM)")
    ap.add_argument("--device", default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    args = ap.parse_args()

    times = np.asarray(args.times) if args.times else np.linspace(1.0, 0.0, args.steps)
    n_panels = len(times)

    if args.config:
        torch.manual_seed(args.seed)
        model, dataset, ckpt = load_model(args.config, args.checkpoint, args.split, args.device)
        print(f"sampling pair {args.pair}/{len(dataset)} from {ckpt}")
        mats, times = sample_trajectory(model, dataset[args.pair], times, args.sample_eta)
        # sanity: how right is each panel? (argmax vs the index-aligned sparse GT diagonal)
        gt = np.arange(mats.shape[-1])
        accs = " ".join(f"t={t:.2f}:{float((m.argmax(-1) == gt).mean()):.2f}"
                        for t, m in zip(times, mats))
        print(f"sparse argmax accuracy per panel — {accs}")
        mats = subset_matrices(mats, args.n, args.seed)
    else:
        mats = build_matrices(args.n, times, args.perm, args.eta, args.tau, args.seed)
    abar_shift = getattr(model, 'logsnr_shift', 0.0) if args.config else 0.0
    abars = [float(cosine_alpha_bar(torch.tensor(float(t)), 0.008, abar_shift)) for t in times]

    n_rows = 2 if args.rows == "both" else 1
    heights = [1.0, 0.72][:n_rows]
    fig, axes = plt.subplots(n_rows, n_panels,
                             figsize=(1.65 * n_panels, 2.5 if n_rows == 2 else 1.9),
                             gridspec_kw={"height_ratios": heights, "hspace": 0.12, "wspace": 0.22})
    axes = np.atleast_2d(axes)

    for k, (t, ab) in enumerate(zip(times, abars)):
        title = rf"$t={t:.2f}$"
        sub = rf"$\bar\alpha={ab:.2f}$"
        if args.rows == "map":
            draw_map(axes[0, k], mats[k])
            axes[0, k].set_title(title, fontsize=11, color=INK, pad=6)
        else:
            draw_matrix(axes[0, k], mats[k], title, sub)
            if n_rows == 2:
                draw_map(axes[1, k], mats[k])

    # reverse-process arrow under the strip
    fig.subplots_adjust(bottom=0.16 if n_rows == 2 else 0.32)
    y = 0.055
    fig.patches.append(plt.matplotlib.patches.FancyArrow(
        0.10, y, 0.80, 0.0, transform=fig.transFigure, figure=fig,
        width=0.0015, head_width=0.02, head_length=0.012,
        color=MUTED, length_includes_head=True))
    caption = args.caption or (r"reverse process $p_\theta(u_{t-1}\mid u_t)$" if args.config
                               else r"reverse process $p_\theta(u_{t-1}\mid u_t)$ (schematic)")
    fig.text(0.5, y + 0.035, caption, ha="center", va="bottom", fontsize=9.5, color=INK)
    fig.text(0.10, y - 0.045, "noise", ha="left", va="top", fontsize=9, color=MUTED)
    fig.text(0.90, y - 0.045, "pointwise map", ha="right", va="top", fontsize=9, color=MUTED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        out = OUT_DIR / f"{args.name}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
