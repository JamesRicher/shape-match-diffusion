"""Under GT correspondence, how consistent is a geodesic patch decomposition across shapes?

Coarse-to-fine matchers assume a patch on shape X maps to *the* corresponding patch on
shape Y. This measures how true that is for the geodesic patches used elsewhere in the
repo (matched FPS centres + nearest-centre assignment = geodesic Voronoi cells).

For a pair (X, Y) we pick bijective FPS centres on the shared ULRSSM template, so centre
cx[k]=corr_x[K[k]] on X is a GT match of cy[k]=corr_y[K[k]] on Y. Each vertex is assigned
to its geodesically nearest centre, giving a patch id per vertex on each shape. Then over
every GT-matched template pair (corr_x[t], corr_y[t]) we ask whether both endpoints land in
the same patch id:

    strict     : patch(corr_x[t]) == patch(corr_y[t])
    within-1   : patch(corr_y[t]) is the correct patch OR one adjacent to it (a boundary
                 slip, not a limb-to-limb jump) -- the error a fine stage can still fix.

Reported as a mean over sampled pairs, swept over the number of patches (coarseness).

    python -m diagnostics.patch_consistency --dataset faust \
        --num-patches 8,16,32,64,128 --num-pairs 40
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_utils import (consistent_bijective_fps, load_corres,   # noqa: E402
                              load_dist, load_off)
from vis.blender.export_lbo_blender import DATASET_DIRS, sorted_off_files  # noqa: E402


def voronoi_labels(dist, centres):
    """Assign every vertex to its geodesically nearest centre. dist (V,V), centres (n,) -> (V,)."""
    return np.argmin(dist[centres], axis=0)


def patch_adjacency(faces, labels, n):
    """Symmetric (n,n) bool: patches sharing a mesh edge (diagonal True). Used for within-1."""
    adj = np.zeros((n, n), dtype=bool)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = labels[faces[:, i]], labels[faces[:, j]]
        adj[a, b] = True
        adj[b, a] = True
    np.fill_diagonal(adj, True)
    return adj


def eval_pair(root, off_x, off_y, n_patches, start):
    """(strict, within1) patch-consistency accuracy for one shape pair, over all GT pairs."""
    def stem(p): return os.path.splitext(os.path.basename(p))[0]
    verts_x = load_off(off_x)[0].astype(np.float32)
    faces_y = load_off(off_y)[1].astype(np.int64)
    corr_x = load_corres(os.path.join(root, "corres", stem(off_x) + ".vts"))
    corr_y = load_corres(os.path.join(root, "corres", stem(off_y) + ".vts"))
    dist_x = load_dist(os.path.join(root, "dist", stem(off_x) + ".mat"))
    dist_y = load_dist(os.path.join(root, "dist", stem(off_y) + ".mat"))

    K = consistent_bijective_fps(verts_x, corr_x, corr_y, n_patches, start=start, dist_x=dist_x)
    n = len(K)
    cx, cy = corr_x[K], corr_y[K]

    lab_x = voronoi_labels(dist_x, cx)
    lab_y = voronoi_labels(dist_y, cy)

    px = lab_x[corr_x]                                    # patch of each template point on X
    py = lab_y[corr_y]                                    # patch of its GT match on Y
    strict = float(np.mean(px == py))

    adj_y = patch_adjacency(faces_y, lab_y, n)
    within1 = float(np.mean(adj_y[px, py]))               # Y-endpoint in the right patch or a neighbour
    return strict, within1, n


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=DATASET_DIRS, default="faust")
    p.add_argument("--num-patches", default="8,16,32,64,128",
                   help="comma list of patch counts to sweep")
    p.add_argument("--num-pairs", type=int, default=40, help="random shape pairs to average over")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=int, default=0, help="FPS start position")
    return p.parse_args()


def main():
    args = parse_args()
    root = DATASET_DIRS[args.dataset]
    offs = sorted_off_files(root)
    counts = [int(x) for x in args.num_patches.split(",")]

    rng = np.random.default_rng(args.seed)
    pairs = []
    while len(pairs) < args.num_pairs:
        i, j = rng.integers(len(offs), size=2)
        if i != j:
            pairs.append((offs[i], offs[j]))

    print(f"dataset={args.dataset}  pairs={len(pairs)}  (GT template pairs per shape-pair "
          f"evaluated in full)\n")
    print(f"{'patches':>8} {'strict acc':>12} {'within-1 acc':>13} {'chance':>8}")
    print("-" * 44)
    for n_req in counts:
        s, w, ns = [], [], []
        for off_x, off_y in pairs:
            strict, within1, n = eval_pair(root, off_x, off_y, n_req, args.start)
            s.append(strict); w.append(within1); ns.append(n)
        n_eff = int(np.round(np.mean(ns)))
        note = "" if n_eff == n_req else f" (n->{n_eff}, bijective cap)"
        print(f"{n_req:>8} {np.mean(s):>10.3f}   {np.mean(w):>11.3f}   {1.0/n_eff:>6.3f}{note}")


if __name__ == "__main__":
    main()
