"""Visualise which mesh vertices are covered by the ground-truth template match.

The GT correspondence for a shape is stored as a ``.vts`` list of vertex indices
(one per template point). Only those vertices have a known ground-truth
correspondence — the rest of the surface is uncovered. This script draws the mesh
as a polyscope surface and overlays the covered vertices as a point cloud on top,
so you can see how much of the shape the template actually reaches.

Example:
    python -m vis.gt_coverage_vis --type SingleFaustDataset --name Faust_r --phase test --index 0
"""
import argparse
import sys
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from datasets import build_dataset


BASE_COLOR = np.array([0.85, 0.85, 0.85], dtype=np.float32)   # the mesh surface
COVERED_COLOR = np.array([0.90, 0.20, 0.20], dtype=np.float32)  # covered vertices


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class GTCoverageVis:
    def __init__(self, dataset, index: int):
        self.dataset = dataset
        self.index = int(np.clip(index, 0, len(dataset) - 1))
        self._show_shape()

    def _show_shape(self):
        ps.remove_all_structures()
        item = self.dataset[self.index]

        verts = _to_numpy(item["verts"]).astype(np.float32)
        faces = _to_numpy(item["faces"]).astype(np.int64)
        # unique vertices reached by the template (corr may list a vertex more than once)
        covered = np.unique(_to_numpy(item["corr"]).astype(np.int64))
        name = item.get("name", self.index)

        ps.register_surface_mesh(f"mesh [{name}]", verts, faces, smooth_shade=True,
                                 color=BASE_COLOR)

        # radius is relative to the scene bounding box, so a fixed value reads on any dataset
        pc = ps.register_point_cloud("gt_covered", verts[covered], radius=0.005)
        pc.add_color_quantity("covered",
                              np.tile(COVERED_COLOR, (len(covered), 1)), enabled=True)

        self._stats = (name, len(covered), verts.shape[0])
        print(f"[{name}] covered {len(covered)} / {verts.shape[0]} vertices "
              f"({100.0 * len(covered) / verts.shape[0]:.1f}%)")

    def ui_callback(self):
        name, n_cov, n_tot = self._stats
        psim.TextUnformatted("GT template-coverage viewer")
        psim.TextUnformatted(f"shape: {name}")
        psim.TextUnformatted(f"covered: {n_cov} / {n_tot} "
                             f"({100.0 * n_cov / n_tot:.1f}%)")
        psim.Separator()

        changed, new_idx = psim.SliderInt("shape index", self.index,
                                          0, len(self.dataset) - 1)
        if changed:
            self.index = int(new_idx)
            self._show_shape()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", default="SingleFaustDataset",
                        help="Dataset class name registered in DATASET_REGISTRY")
    parser.add_argument("--name", default="Faust_r",
                        help="Dataset name key for the default data root")
    parser.add_argument("--phase", default="test", help="train/test/full (default: test)")
    parser.add_argument("--index", type=int, default=0,
                        help="Index of the shape to display")
    args = parser.parse_args(argv)

    dataset_opt = {
        "type": args.type,
        "name": args.name,
        "phase": args.phase,
        "ret_faces": True,
        "ret_feats": False,
        "ret_corr": True,   # we need the GT correspondence indices
        "ret_dist": False,
        "ret_evecs": False,
    }
    dataset = build_dataset(dataset_opt)

    ps.init()
    ps.set_up_dir("neg_y_up" if getattr(dataset, "flip_up", False) else "y_up")

    viewer = GTCoverageVis(dataset, index=args.index)
    ps.set_user_callback(viewer.ui_callback)
    ps.show()


if __name__ == "__main__":
    main(sys.argv[1:])
