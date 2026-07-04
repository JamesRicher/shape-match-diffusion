import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import build_dataset
from metrics import build_metric
from paths import FROZEN_BASELINES_ROOT


OUTPUT_ROOT = Path(FROZEN_BASELINES_ROOT)

# metrics pulled from the registry (see metrics/geo_metric.py, metrics/map_metric.py)
calculate_geodesic_error = build_metric({"type": "calculate_geodesic_error"})
plot_pck = build_metric({"type": "plot_pck"})
plot_pck_multi = build_metric({"type": "plot_pck_multi"})
geodesic_distortion = build_metric({"type": "geodesic_distortion"})
dirichlet_energy = build_metric({"type": "dirichlet_energy"})


# ret_evecs pulls in the cached DiffusionNet-style operators (L, mass, ...) that the
# distortion / Dirichlet-energy metrics consume.
DATASET_OPTS = {
    "FAUST_r": {"name": "Faust_r", "type": "PairFaustDataset", "phase": "test", "ret_evecs": True},
    "SMAL_r":  {"name": "Smal_r",  "type": "PairSmalDataset",  "phase": "test", "ret_evecs": True},
    "SCAPE_r": {"name": "Scape_r", "type": "PairScapeDataset", "phase": "test", "ret_evecs": True},
}

PCK_MAX = 0.25
PCK_N = 100


def cosine_similarity_matrix(feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
    """|V_a| x |V_b| cosine similarity between per-vertex feature tensors."""
    a = F.normalize(feat_a, dim=-1)
    b = F.normalize(feat_b, dim=-1)
    return a @ b.T


def nn_assignment(sim: torch.Tensor) -> torch.Tensor:
    """For each vertex in A, index of its nearest-neighbour vertex in B. Shape [V_a]."""
    return sim.argmax(dim=1)


def sqrt_surface_area(mass: torch.Tensor) -> torch.Tensor:
    """sqrt of total surface area, read off the lumped-mass (per-vertex area) vector."""
    return mass.sum().sqrt()


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature-based NN for a baseline assignment quality")
    parser.add_argument('-d', '--datasets', nargs='+', default=['all'],
                        choices=list(DATASET_OPTS.keys()) + ['all'],
                        help="dataset name(s); 'all' runs every registered dataset (test phase)")
    return parser.parse_args()


@torch.no_grad()
def run_dataset(dataset_key: str):
    opts = DATASET_OPTS[dataset_key]
    ds = build_dataset(opts)
    print(f"{dataset_key}: {len(ds)} pairs")

    all_errs = []
    distortion_sums = []
    distortion_means = []
    energies = []
    for i in tqdm(range(len(ds)), desc=dataset_key):
        i_a, i_b = ds.combinations[i]
        if i_a == i_b:
            continue
        pair = ds[i]
        first, second = pair['first'], pair['second']

        sim = cosine_similarity_matrix(first['feat'], second['feat'])
        p2p = nn_assignment(sim)
        sqrt_area_a = sqrt_surface_area(first['mass'])
        sqrt_area_b = sqrt_surface_area(second['mass'])

        # Geodesic error on shape B: x = B (second), y = A (first), p2p maps A -> B.
        # The registry metric is un-normalized, so scale by sqrt(area(B)) ourselves.
        err = calculate_geodesic_error(
            dist_x=second['dist'].numpy(),
            corr_x=second['corr'].numpy(),
            corr_y=first['corr'].numpy(),
            p2p=p2p.numpy(),
            return_mean=False,
        ) / sqrt_area_b.item()
        all_errs.append(err)

        d_sum, d_mean = geodesic_distortion(p2p, first['dist'], second['dist'],
                                            sqrt_area_a, sqrt_area_b)
        distortion_sums.append(d_sum.item())
        distortion_means.append(d_mean.item())
        energies.append(dirichlet_energy(p2p, first['L'],
                                         second['verts'], sqrt_area_b).item())

    all_errs = np.concatenate(all_errs)
    thresholds = np.linspace(0, PCK_MAX, PCK_N)
    auc, fig, pck = plot_pck(all_errs, threshold=PCK_MAX, steps=PCK_N,
                             label=f"{dataset_key} — frozen feature NN")
    mge = float(all_errs.mean())
    mean_distortion_sum = float(np.mean(distortion_sums))
    mean_distortion_mean = float(np.mean(distortion_means))
    mean_dirichlet = float(np.mean(energies))

    print(f"MGE: {mge:.4f}")
    print(f"AUC (0-{PCK_MAX}): {auc:.4f}")
    print(f"Geodesic distortion (sum per pair, avg over pairs):  {mean_distortion_sum:.4f}")
    print(f"Geodesic distortion (mean per vertex-pair, avg):     {mean_distortion_mean:.4f}")
    print(f"Dirichlet energy (avg over pairs):                   {mean_dirichlet:.4f}")

    out_dir = OUTPUT_ROOT / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / "pck.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        out_dir / "pck_data.npz",
        thresholds=thresholds,
        pck=pck,
        errors=all_errs,
    )

    stats = {
        "dataset": dataset_key,
        "dataset_opts": opts,
        "num_pairs": len(ds),
        "num_correspondences": int(all_errs.size),
        "pck_max": PCK_MAX,
        "pck_n": PCK_N,
        "mge": mge,
        "auc": auc,
        "geodesic_distortion_sum": mean_distortion_sum,
        "geodesic_distortion_mean": mean_distortion_mean,
        "dirichlet_energy": mean_dirichlet,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"wrote outputs to {out_dir}")

    return all_errs


def main():
    args = parse_args()
    if 'all' in args.datasets:
        dataset_keys = list(DATASET_OPTS.keys())
    else:
        dataset_keys = args.datasets

    errors_by_dataset = {key: run_dataset(key) for key in dataset_keys}

    # Overlay the PCK curves of every dataset run for side-by-side comparison.
    if len(errors_by_dataset) > 1:
        aucs, fig, _ = plot_pck_multi(errors_by_dataset, threshold=PCK_MAX, steps=PCK_N,
                                      title="Frozen feature NN — PCK by dataset")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_ROOT / "pck_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote comparison plot to {OUTPUT_ROOT / 'pck_comparison.png'} "
              f"(AUCs: {', '.join(f'{k}={v:.3f}' for k, v in aucs.items())})")


if __name__ == "__main__":
    main()
