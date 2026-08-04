"""Build and visualise a sparse shape graph: 512 geodesic-FPS vertices + geodesic kNN edges.

Untracked scratch script. For one shape from a dataset it:
  1. loads the mesh (.off) and its precomputed geodesic distance matrix (dist/*.mat),
  2. picks n_sparse points by geodesic farthest-point sampling (repo default),
  3. connects each sparse point to its k geodesically nearest sparse points,
  4. shows the mesh, the sampled points, and the graph edges in polyscope.

    python shape_graph_vis.py --dataset faust --index 0 --k 6
    python shape_graph_vis.py --dataset scape --index 3 --n-sparse 512 --k 8
"""
import argparse
import os

import numpy as np

from paths import DEFAULT_DATA_ROOTS
from utils.data_utils import fps, load_dist, load_off
from vis.blender.export_lbo_blender import sorted_off_files

# dataset name -> data root (from paths.py, keyed lower-case for the CLI)
DATASETS = {name.split("_")[0].lower(): root for name, root in DEFAULT_DATA_ROOTS.items()}


def dist_path(root, off_file):
    stem = os.path.splitext(os.path.basename(off_file))[0]
    return os.path.join(root, "dist", stem + ".mat")


def knn_edges(dist_sub, k):
    """Undirected edge list (E, 2) linking each row to its k nearest others (self excluded)."""
    n = dist_sub.shape[0]
    k = min(k, n - 1)
    nbrs = np.argsort(dist_sub, axis=1)[:, 1:k + 1]           # (n, k), drop self (col 0)
    edges = {tuple(sorted((i, int(j)))) for i in range(n) for j in nbrs[i]}
    return np.array(sorted(edges), dtype=np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(DATASETS), default="faust")
    p.add_argument("--index", type=int, default=0, help="shape index (numeric .off order)")
    p.add_argument("--n-sparse", type=int, default=512, help="number of FPS points")
    p.add_argument("--k", type=int, default=6, help="neighbours per point in the geodesic kNN graph")
    p.add_argument("--start", type=int, default=0, help="FPS start vertex")
    args = p.parse_args()

    root = DATASETS[args.dataset]
    offs = sorted_off_files(root)
    off_file = offs[args.index]
    stem = os.path.splitext(os.path.basename(off_file))[0]
    print(f"dataset={args.dataset}  shape={stem}  n_sparse={args.n_sparse}  k={args.k}")

    verts, faces = load_off(off_file)
    verts = verts.astype(np.float32)
    faces = faces.astype(np.int64)
    dist = load_dist(dist_path(root, off_file))                # (V, V) geodesic

    n = min(args.n_sparse, verts.shape[0])
    sel = fps(verts, n, args.start, dist=dist)                 # geodesic FPS (repo default)
    dist_sub = dist[sel][:, sel]
    edges = knn_edges(dist_sub, args.k)
    print(f"sampled {n} points, {len(edges)} edges")

    import polyscope as ps
    ps.init()
    ps.register_surface_mesh("mesh", verts, faces, enabled=True,
                             color=(0.85, 0.85, 0.9), transparency=0.5)
    graph = ps.register_curve_network("shape_graph", verts[sel], edges,
                                      color=(0.9, 0.2, 0.2), radius=0.0015)
    graph.add_scalar_quantity("fps_order", np.arange(n), enabled=True)
    ps.register_point_cloud("sparse_points", verts[sel], color=(0.1, 0.3, 0.9), radius=0.004)
    ps.show()


if __name__ == "__main__":
    main()
