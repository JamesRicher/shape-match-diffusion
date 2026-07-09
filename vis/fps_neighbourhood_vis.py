"""Visualise FPS samples and their Dijkstra neighbourhoods on a couple of meshes.

Two shapes are drawn side by side as polyscope surface meshes. On each shape a
number of FPS points is sampled (same `fps()` as the sparse pipeline) and shown
as a point cloud in a fixed color. Around every FPS point, its Dijkstra
neighbourhood — the `nbr size` graph-geodesically nearest vertices, computed on
the mesh edge graph with Euclidean edge lengths — is painted on the surface,
one color per FPS point. Contested vertices go to the closer FPS point.

Example:
    python -m vis.fps_neighbourhood_vis --type SingleFaustDataset --name Faust_r --phase test --num-fps 128 --nbr-size 40
"""
import argparse
import sys
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import polyscope as ps
import polyscope.imgui as psim

from datasets import build_dataset
from utils.data_utils import fps


BASE_COLOR = np.array([0.85, 0.85, 0.85], dtype=np.float32)   # unclaimed surface
FPS_COLOR = np.array([0.05, 0.05, 0.05], dtype=np.float32)    # the FPS points


def _hsv_palette(n: int) -> np.ndarray:
    """Return n RGB colors evenly spaced around the HSV wheel."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    hues = (np.arange(n, dtype=np.float32) + 0.5) / n
    h6 = hues * 6.0
    c = np.ones(n, dtype=np.float32)
    x = 1.0 - np.abs((h6 % 2.0) - 1.0)
    z = np.zeros(n, dtype=np.float32)
    rgb = np.zeros((n, 3), dtype=np.float32)
    seg = h6.astype(np.int32) % 6
    for s, cols in enumerate([(c, x, z), (x, c, z), (z, c, x),
                              (z, x, c), (x, z, c), (c, z, x)]):
        rgb[seg == s] = np.stack(cols, axis=1)[seg == s]
    return rgb


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _edge_graph(verts: np.ndarray, faces: np.ndarray) -> scipy.sparse.csr_matrix:
    """Symmetric sparse graph over mesh vertices, weighted by Euclidean edge length."""
    ii = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    jj = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    w = np.linalg.norm(verts[ii] - verts[jj], axis=1)
    n = verts.shape[0]
    g = scipy.sparse.coo_matrix((w, (ii, jj)), shape=(n, n)).tocsr()
    return g.maximum(g.T)


class FPSNeighbourhoodVis:
    def __init__(self, dataset, num_fps: int, nbr_size: int, seed):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        self.num_fps = int(num_fps)
        self.nbr_size = int(nbr_size)

        # per-shape state: dicts with verts (offset applied), faces, graph, mesh, dists
        self.shapes = []
        self._resample_shapes()

    # ------------------------------------------------------------------ shapes

    def _resample_shapes(self):
        n = len(self.dataset)
        if n < 2:
            raise RuntimeError(f"Dataset has only {n} shapes; need at least 2.")
        idx_a, idx_b = self.rng.choice(n, size=2, replace=False)

        ps.remove_all_structures()
        self.shapes = []
        offset = np.zeros(3, dtype=np.float32)
        for tag, idx in (("A", int(idx_a)), ("B", int(idx_b))):
            item = self.dataset[idx]
            verts = _to_numpy(item["verts"]).astype(np.float32) + offset
            faces = _to_numpy(item["faces"]).astype(np.int64)
            extent = float(verts[:, 0].max() - verts[:, 0].min())
            offset = offset + np.array([extent * 1.3, 0.0, 0.0], dtype=np.float32)

            mesh = ps.register_surface_mesh(
                f"shape_{tag} [{item.get('name', idx)}]", verts, faces, smooth_shade=True)
            self.shapes.append({
                "tag": tag, "verts": verts, "faces": faces, "mesh": mesh,
                "graph": _edge_graph(verts, faces),
                "fps_idx": None, "dists": None,   # filled by _recompute_fps
            })
        self._recompute_fps()

    # ------------------------------------------------------------ fps + dijkstra

    def _recompute_fps(self):
        """Resample FPS points and rerun Dijkstra from them (num_fps changed)."""
        for s in self.shapes:
            n_verts = s["verts"].shape[0]
            k = int(np.clip(self.num_fps, 1, n_verts))
            s["fps_idx"] = fps(s["verts"], k, start=0)
            # (num_fps, N) geodesic-ish distances along mesh edges from each FPS point
            s["dists"] = scipy.sparse.csgraph.dijkstra(
                s["graph"], directed=False, indices=s["fps_idx"])
        self._repaint()

    def _repaint(self):
        """Recolor neighbourhoods (cheap: reuses cached Dijkstra distances)."""
        for s in self.shapes:
            dists, fps_idx, verts = s["dists"], s["fps_idx"], s["verts"]
            n_verts = verts.shape[0]
            nbr = int(np.clip(self.nbr_size, 1, n_verts))

            # each FPS point claims its nbr nearest vertices; ties/overlaps go to
            # the FPS point that is closer
            owner = np.full(n_verts, -1, dtype=np.int64)
            best = np.full(n_verts, np.inf)
            for i, row in enumerate(dists):
                take = np.argpartition(row, min(nbr, n_verts) - 1)[:nbr]
                closer = row[take] < best[take]
                owner[take[closer]] = i
                best[take[closer]] = row[take[closer]]

            palette = _hsv_palette(len(fps_idx))
            colors = np.tile(BASE_COLOR, (n_verts, 1))
            claimed = owner >= 0
            colors[claimed] = palette[owner[claimed]]
            s["mesh"].add_color_quantity("nbr_color", colors, enabled=True)

            extent = float(verts.max(axis=0).max() - verts.min(axis=0).min())
            pc = ps.register_point_cloud(
                f"fps_{s['tag']}", verts[fps_idx],
                radius=0.006 if extent > 0 else 0.006)
            pc.add_color_quantity(
                "fps_color", np.tile(FPS_COLOR, (len(fps_idx), 1)), enabled=True)

    # ------------------------------------------------------------------- UI

    def ui_callback(self):
        psim.TextUnformatted("FPS + Dijkstra neighbourhood viewer")
        psim.Separator()

        changed_fps, new_fps = psim.SliderInt("num fps points", self.num_fps, 2, 1024)
        if changed_fps:
            self.num_fps = int(new_fps)
            self._recompute_fps()

        changed_nbr, new_nbr = psim.SliderInt("nbr size (verts)", self.nbr_size, 1, 500)
        if changed_nbr:
            self.nbr_size = int(new_nbr)
            self._repaint()

        if psim.Button("resample shapes"):
            self._resample_shapes()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", default="SingleFaustDataset",
                        help="Dataset class name registered in DATASET_REGISTRY")
    parser.add_argument("--name", default="Faust_r",
                        help="Dataset name key for the default data root")
    parser.add_argument("--phase", default="test", help="train/test/full (default: test)")
    parser.add_argument("--num-fps", type=int, default=128,
                        help="Number of FPS points per shape")
    parser.add_argument("--nbr-size", type=int, default=40,
                        help="Dijkstra neighbourhood size in vertices per FPS point")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = parser.parse_args(argv)

    dataset_opt = {
        "type": args.type,
        "name": args.name,
        "phase": args.phase,
        "ret_faces": True,
        "ret_feats": False,
        "ret_corr": False,
        "ret_dist": False,
        "ret_evecs": False,
    }
    dataset = build_dataset(dataset_opt)

    ps.init()
    ps.set_up_dir("neg_y_up" if getattr(dataset, "flip_up", False) else "y_up")

    viewer = FPSNeighbourhoodVis(dataset, num_fps=args.num_fps,
                                 nbr_size=args.nbr_size, seed=args.seed)
    ps.set_user_callback(viewer.ui_callback)
    ps.show()


if __name__ == "__main__":
    main(sys.argv[1:])
