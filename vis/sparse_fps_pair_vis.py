"""Visualise the consistent bijective FPS sampling on a pair of meshes.

Two shapes are drawn side by side (A = first/X, B = second/Y). n sparse points are
chosen by consistent bijective FPS on A and pushed through the shared template to B;
matched points share a colour, so a correct bijection is visible at a glance. The
point count is a live polyscope slider.

Because the .vts correspondence is many-to-one per shape, only so many points can be
distinct on BOTH shapes at once. The viewer computes that maximum for the current pair
and start, reports it, and flags (in red) when the requested count exceeds it — beyond
that, a bijective sparse GT is impossible and the count is clamped.

Example:
    python -m vis.sparse_fps_pair_vis --name Faust_r --phase test --num-points 128
"""
import argparse
import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from datasets import build_dataset
from utils.data_utils import bijective_fps_order
from vis.fps_neighbourhood_vis import _hsv_palette, _to_numpy


class SparseFPSPairVis:
    def __init__(self, dataset, num_points: int, seed):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        self.num_points = int(num_points)
        self.start = 0
        self.order = None          # full bijective FPS order (len = max achievable)
        self.pair = None           # (A dict, B dict) with verts/faces/corr/offset
        self._resample_pair()

    # ----------------------------------------------------------------- sampling

    def _resample_pair(self):
        n = len(self.dataset)
        if n < 2:
            raise RuntimeError(f"Dataset has only {n} shapes; need at least 2.")
        idx_a, idx_b = self.rng.choice(n, size=2, replace=False)

        ps.remove_all_structures()
        shapes, offset = [], np.zeros(3, dtype=np.float32)
        for tag, idx in (("A", int(idx_a)), ("B", int(idx_b))):
            item = self.dataset[idx]
            verts = _to_numpy(item["verts"]).astype(np.float32) + offset
            faces = _to_numpy(item["faces"]).astype(np.int64)
            corr = _to_numpy(item["corr"]).astype(np.int64)
            extent = float(verts[:, 0].max() - verts[:, 0].min())
            offset = offset + np.array([extent * 1.3, 0.0, 0.0], dtype=np.float32)
            ps.register_surface_mesh(f"shape_{tag} [{item.get('name', idx)}]",
                                     verts, faces, smooth_shade=True)
            shapes.append({"tag": tag, "verts": verts, "faces": faces, "corr": corr})
        self.pair = shapes
        self._recompute_order()

    def _recompute_order(self):
        """Exhaustive bijective FPS order for the current pair + start (cheap enough
        for interactive use; the slider then just takes a prefix)."""
        a, b = self.pair
        # order indexes the shared template; A is the FPS shape (X), B is pushed (Y).
        self.order = bijective_fps_order(a["verts"], a["corr"], b["corr"], self.start)
        self._repaint()

    # -------------------------------------------------------------------- draw

    def max_points(self) -> int:
        return len(self.order)

    def _repaint(self):
        n = min(self.num_points, self.max_points())
        K = self.order[:n]
        palette = _hsv_palette(n)
        for s in self.pair:
            pts = s["verts"][s["corr"][K]]                 # matched sparse points
            pc = ps.register_point_cloud(f"fps_{s['tag']}", pts, radius=0.008)
            pc.add_color_quantity("match", palette, enabled=True)

    # --------------------------------------------------------------------- UI

    def ui_callback(self):
        psim.TextUnformatted("Consistent bijective FPS — matched points share a colour")
        psim.Separator()

        max_n = self.max_points()
        changed, new_n = psim.SliderInt("num points", self.num_points, 1, max(2 * max_n, 8))
        if changed:
            self.num_points = int(new_n)
            self._repaint()

        psim.TextUnformatted(f"max bijective points (this pair/start): {max_n}")
        if self.num_points > max_n:
            psim.TextColored((1.0, 0.3, 0.3, 1.0),
                             f"requested {self.num_points} > {max_n}: bijective FPS "
                             f"impossible — clamped to {max_n}")
        else:
            psim.TextColored((0.3, 1.0, 0.3, 1.0),
                             f"showing {self.num_points} bijective matched points")

        if psim.Button("new FPS start"):
            self.start = int(self.rng.integers(self.pair[0]["corr"].shape[0]))
            self._recompute_order()
        psim.SameLine()
        if psim.Button("resample pair"):
            self.start = 0
            self._resample_pair()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", default="SingleFaustDataset",
                        help="Dataset class name registered in DATASET_REGISTRY")
    parser.add_argument("--name", default="Faust_r", help="Dataset name key for the data root")
    parser.add_argument("--phase", default="test", help="train/test/full (default: test)")
    parser.add_argument("--num-points", type=int, default=128, help="Initial sparse point count")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for pair selection")
    args = parser.parse_args(argv)

    dataset = build_dataset({
        "type": args.type, "name": args.name, "phase": args.phase,
        "ret_faces": True, "ret_feats": False, "ret_corr": True,
        "ret_dist": False, "ret_evecs": False,
    })

    ps.init()
    ps.set_up_dir("neg_y_up" if getattr(dataset, "flip_up", False) else "y_up")
    viewer = SparseFPSPairVis(dataset, num_points=args.num_points, seed=args.seed)
    ps.set_user_callback(viewer.ui_callback)
    ps.show()


if __name__ == "__main__":
    main(sys.argv[1:])
