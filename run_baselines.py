import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import build_dataset
from paths import FROZEN_BASELINES_ROOT


OUTPUT_ROOT = Path(FROZEN_BASELINES_ROOT)


DATASET_OPTS = {
    "FAUST_r": {"name": "Faust_r", "type": "PairFaustDataset", "phase": "test"},
    "SMAL_r":  {"name": "Smal_r",  "type": "PairSmalDataset",  "phase": "test"},
    "SCAPE_r": {"name": "Scape_r", "type": "PairScapeDataset", "phase": "test"},
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


def sqrt_surface_area(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return (0.5 * torch.linalg.cross(v1 - v0, v2 - v0).norm(dim=1).sum()).sqrt()


def geodesic_error(p2p: torch.Tensor,
                   dist_b: torch.Tensor,
                   corr_a: torch.Tensor,
                   corr_b: torch.Tensor,
                   sqrt_area_b: torch.Tensor) -> torch.Tensor:
    """Per-correspondence geodesic error on B, normalized by sqrt(area(B))."""
    return dist_b[p2p[corr_a], corr_b] / sqrt_area_b


def geodesic_distortion(p2p: torch.Tensor,
                        dist_a: torch.Tensor,
                        dist_b: torch.Tensor,
                        sqrt_area_a: torch.Tensor,
                        sqrt_area_b: torch.Tensor):
    """Global-isometry distortion of the map A -> B: |d_A(i,j) - d_B(p2p(i), p2p(j))|,
    with both distance matrices normalized by sqrt of surface area. Returns (sum, mean)
    over all V_a^2 vertex pairs."""
    d_a = dist_a / sqrt_area_a
    d_b_mapped = dist_b[p2p][:, p2p] / sqrt_area_b
    diff = (d_a - d_b_mapped).abs()
    return diff.sum(), diff.mean()


def dirichlet_energy(p2p: torch.Tensor,
                     verts_a: torch.Tensor,
                     faces_a: torch.Tensor,
                     verts_b: torch.Tensor,
                     sqrt_area_b: torch.Tensor) -> torch.Tensor:
    """Cotangent Dirichlet energy E = (1/2) f^T L_A f of the mapping
    f(v) = verts_B[p2p(v)] / sqrt(area_B), with L_A the cot Laplacian of mesh A.
    Assembled face-wise: E = (1/4) sum_faces (cot_i ||f_j-f_k||^2 + cyc.)."""
    v0 = verts_a[faces_a[:, 0]]
    v1 = verts_a[faces_a[:, 1]]
    v2 = verts_a[faces_a[:, 2]]

    e01 = v1 - v0
    e02 = v2 - v0
    e12 = v2 - v1

    two_area = torch.linalg.cross(e01, e02).norm(dim=1).clamp_min(1e-12)

    cot0 = (e01 * e02).sum(dim=1) / two_area
    cot1 = -(e01 * e12).sum(dim=1) / two_area
    cot2 = (e02 * e12).sum(dim=1) / two_area

    f = verts_b[p2p] / sqrt_area_b
    f0 = f[faces_a[:, 0]]
    f1 = f[faces_a[:, 1]]
    f2 = f[faces_a[:, 2]]

    d12_sq = ((f1 - f2) ** 2).sum(dim=1)
    d02_sq = ((f0 - f2) ** 2).sum(dim=1)
    d01_sq = ((f0 - f1) ** 2).sum(dim=1)

    return 0.25 * (cot0 * d12_sq + cot1 * d02_sq + cot2 * d01_sq).sum()


def pck_and_auc(errs: torch.Tensor, thresholds: torch.Tensor):
    pck = (errs.unsqueeze(0) <= thresholds.unsqueeze(1)).float().mean(dim=1)
    auc = torch.trapz(pck, thresholds) / (thresholds[-1] - thresholds[0])
    return pck, auc


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
        sqrt_area_a = sqrt_surface_area(first['verts'], first['faces'])
        sqrt_area_b = sqrt_surface_area(second['verts'], second['faces'])
        err = geodesic_error(p2p, second['dist'], first['corr'], second['corr'], sqrt_area_b)
        all_errs.append(err)

        d_sum, d_mean = geodesic_distortion(p2p, first['dist'], second['dist'],
                                            sqrt_area_a, sqrt_area_b)
        distortion_sums.append(d_sum.item())
        distortion_means.append(d_mean.item())
        energies.append(dirichlet_energy(p2p, first['verts'], first['faces'],
                                         second['verts'], sqrt_area_b).item())

    all_errs = torch.cat(all_errs)
    thresholds = torch.linspace(0, PCK_MAX, PCK_N)
    pck, auc = pck_and_auc(all_errs, thresholds)
    mge = all_errs.mean().item()
    mean_distortion_sum = float(np.mean(distortion_sums))
    mean_distortion_mean = float(np.mean(distortion_means))
    mean_dirichlet = float(np.mean(energies))

    print(f"MGE: {mge:.4f}")
    print(f"AUC (0-{PCK_MAX}): {auc.item():.4f}")
    print(f"Geodesic distortion (sum per pair, avg over pairs):  {mean_distortion_sum:.4f}")
    print(f"Geodesic distortion (mean per vertex-pair, avg):     {mean_distortion_mean:.4f}")
    print(f"Dirichlet energy (avg over pairs):                   {mean_dirichlet:.4f}")

    out_dir = OUTPUT_ROOT / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(thresholds.numpy(), pck.numpy())
    plt.xlabel("geodesic error / sqrt(area)")
    plt.ylabel("PCK")
    plt.title(f"{dataset_key} — frozen feature NN")
    plt.xlim(0, PCK_MAX)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / "pck.png", dpi=150, bbox_inches="tight")
    plt.close()

    np.savez(
        out_dir / "pck_data.npz",
        thresholds=thresholds.numpy(),
        pck=pck.numpy(),
        errors=all_errs.numpy(),
    )

    stats = {
        "dataset": dataset_key,
        "dataset_opts": opts,
        "num_pairs": len(ds),
        "num_correspondences": int(all_errs.numel()),
        "pck_max": PCK_MAX,
        "pck_n": PCK_N,
        "mge": mge,
        "auc": auc.item(),
        "geodesic_distortion_sum": mean_distortion_sum,
        "geodesic_distortion_mean": mean_distortion_mean,
        "dirichlet_energy": mean_dirichlet,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"wrote outputs to {out_dir}")


def main():
    args = parse_args()
    if 'all' in args.datasets:
        dataset_keys = list(DATASET_OPTS.keys())
    else:
        dataset_keys = args.datasets

    for key in dataset_keys:
        run_dataset(key)


if __name__ == "__main__":
    main()
