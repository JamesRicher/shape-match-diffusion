"""Visualise a random pair of shapes from a dataset with sampled correspondences.

Two shapes are drawn side by side as polyscope surface meshes. A user-controlled
number of correspondence points is drawn on each shape as a point cloud, with a
shared per-index color linking the point on shape A to its match on shape B.
The first 5 feature channels are exposed as scalar quantities on both meshes.

Example:
    python -m vis.random_pair_vis --type SingleFaustDataset --name Faust_r --phase test
"""
import argparse
import sys
import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from datasets import build_dataset


MAX_FEAT_CHANNELS = 5


def _hsv_palette(n: int) -> np.ndarray:
    """Return n RGB colors evenly spaced around the HSV wheel."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    hues = (np.arange(n, dtype=np.float32) + 0.5) / n
    h6 = hues * 6.0
    c = np.ones(n, dtype=np.float32)
    x = 1.0 - np.abs((h6 % 2.0) - 1.0)
    rgb = np.zeros((n, 3), dtype=np.float32)
    seg = h6.astype(np.int32) % 6
    rgb[seg == 0] = np.stack([c, x, np.zeros(n, dtype=np.float32)], axis=1)[seg == 0]
    rgb[seg == 1] = np.stack([x, c, np.zeros(n, dtype=np.float32)], axis=1)[seg == 1]
    rgb[seg == 2] = np.stack([np.zeros(n, dtype=np.float32), c, x], axis=1)[seg == 2]
    rgb[seg == 3] = np.stack([np.zeros(n, dtype=np.float32), x, c], axis=1)[seg == 3]
    rgb[seg == 4] = np.stack([x, np.zeros(n, dtype=np.float32), c], axis=1)[seg == 4]
    rgb[seg == 5] = np.stack([c, np.zeros(n, dtype=np.float32), x], axis=1)[seg == 5]
    return rgb


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class PairVis:
    def __init__(self, dataset, num_corr: int, seed):
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        self.num_corr = int(num_corr)
        self.max_corr = 0
        self.flip_up = bool(getattr(dataset, "flip_up", False))

        # names for registered structures — kept fixed so we can re-register
        self.mesh_a_name = "shape_A"
        self.mesh_b_name = "shape_B"
        self.pts_a_name = "corr_A"
        self.pts_b_name = "corr_B"

        self.item_a = None
        self.item_b = None
        self.verts_a = None
        self.verts_b_offset = None

        self._resample_pair()

    # ------------------------------------------------------------------ shapes

    def _resample_pair(self):
        n = len(self.dataset)
        if n < 2:
            raise RuntimeError(f"Dataset has only {n} shapes; need at least 2.")
        idx_a, idx_b = self.rng.choice(n, size=2, replace=False)
        self.item_a = self.dataset[int(idx_a)]
        self.item_b = self.dataset[int(idx_b)]

        self.verts_a = _to_numpy(self.item_a["verts"]).astype(np.float32)
        faces_a = _to_numpy(self.item_a["faces"]).astype(np.int64)
        verts_b = _to_numpy(self.item_b["verts"]).astype(np.float32)
        faces_b = _to_numpy(self.item_b["faces"]).astype(np.int64)

        extent = float(self.verts_a[:, 0].max() - self.verts_a[:, 0].min())
        offset = np.array([extent * 1.3, 0.0, 0.0], dtype=np.float32)
        self.verts_b_offset = verts_b + offset

        # correspondences share a template index space
        corr_a = _to_numpy(self.item_a["corr"]).astype(np.int64)
        corr_b = _to_numpy(self.item_b["corr"]).astype(np.int64)
        self.corr_a = corr_a
        self.corr_b = corr_b
        self.max_corr = int(min(corr_a.shape[0], corr_b.shape[0]))
        if self.num_corr > self.max_corr:
            self.num_corr = self.max_corr

        # register meshes fresh
        ps.remove_all_structures()
        name_a = f"{self.mesh_a_name} [{self.item_a.get('name', idx_a)}]"
        name_b = f"{self.mesh_b_name} [{self.item_b.get('name', idx_b)}]"
        self.mesh_a = ps.register_surface_mesh(name_a, self.verts_a, faces_a, smooth_shade=True)
        self.mesh_b = ps.register_surface_mesh(name_b, self.verts_b_offset, faces_b, smooth_shade=True)

        self._add_feature_quantities()
        self._resample_correspondences()

    def _add_feature_quantities(self):
        feat_a = _to_numpy(self.item_a.get("feat")) if "feat" in self.item_a else None
        feat_b = _to_numpy(self.item_b.get("feat")) if "feat" in self.item_b else None
        if feat_a is None or feat_b is None:
            return
        n_channels = min(feat_a.shape[1], feat_b.shape[1], MAX_FEAT_CHANNELS)
        for c in range(n_channels):
            name = f"feat_{c:03d}"
            vmax = float(max(np.abs(feat_a[:, c]).max(), np.abs(feat_b[:, c]).max()))
            vrange = (-vmax, vmax) if vmax > 0 else (-1.0, 1.0)
            enabled = (c == 0)
            self.mesh_a.add_scalar_quantity(
                name, feat_a[:, c], cmap="coolwarm", vminmax=vrange, enabled=enabled
            )
            self.mesh_b.add_scalar_quantity(
                name, feat_b[:, c], cmap="coolwarm", vminmax=vrange, enabled=enabled
            )

    # ------------------------------------------------------- correspondences

    def _resample_correspondences(self):
        n = int(np.clip(self.num_corr, 0, self.max_corr))
        template_idx = self.rng.choice(self.max_corr, size=n, replace=False)
        va = self.verts_a[self.corr_a[template_idx]]
        vb = self.verts_b_offset[self.corr_b[template_idx]]
        colors = _hsv_palette(n)

        # radius scaled to shape extent for visibility
        extent = float(self.verts_a.max(axis=0).max() - self.verts_a.min(axis=0).min())
        radius = 0.008 if extent <= 0 else 0.008

        pc_a = ps.register_point_cloud(self.pts_a_name, va, radius=radius)
        pc_b = ps.register_point_cloud(self.pts_b_name, vb, radius=radius)
        if n > 0:
            pc_a.add_color_quantity("corr_color", colors, enabled=True)
            pc_b.add_color_quantity("corr_color", colors, enabled=True)

    # ------------------------------------------------------------------- UI

    def ui_callback(self):
        psim.TextUnformatted("Random pair correspondence viewer")
        psim.Separator()

        if self.max_corr > 0:
            changed, new_val = psim.SliderInt(
                "num correspondences", self.num_corr, 0, self.max_corr
            )
            if changed:
                self.num_corr = int(new_val)
                self._resample_correspondences()

        if psim.Button("resample correspondences"):
            self._resample_correspondences()
        psim.SameLine()
        if psim.Button("resample shape pair"):
            self._resample_pair()


def _parse_kv(items):
    """Parse `--dataset-arg key=value` pairs into a dict, converting basic types."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise argparse.ArgumentTypeError(f"expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        vl = v.lower()
        if vl in ("true", "false"):
            out[k] = (vl == "true")
        else:
            try:
                out[k] = int(v)
            except ValueError:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--type", required=True,
                        help="Dataset class name registered in DATASET_REGISTRY, e.g. SingleFaustDataset")
    parser.add_argument("--name", required=True,
                        help="Dataset name key for the default data root, e.g. Faust_r")
    parser.add_argument("--phase", default="test", help="train/test/full (default: test)")
    parser.add_argument("--num-corr", type=int, default=20,
                        help="Initial number of correspondence points to show")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument("--dataset-arg", action="append", default=[],
                        help="Extra key=value forwarded to the dataset constructor (repeatable)")
    args = parser.parse_args(argv)

    dataset_opt = {
        "type": args.type,
        "name": args.name,
        "phase": args.phase,
        "ret_faces": True,
        "ret_feats": True,
        "ret_corr": True,
        "ret_dist": False,
        "ret_evecs": False,
    }
    dataset_opt.update(_parse_kv(args.dataset_arg))
    dataset = build_dataset(dataset_opt)

    ps.init()
    ps.set_up_dir("neg_y_up" if getattr(dataset, "flip_up", False) else "y_up")

    viewer = PairVis(dataset, num_corr=args.num_corr, seed=args.seed)
    ps.set_user_callback(viewer.ui_callback)
    ps.show()


if __name__ == "__main__":
    main(sys.argv[1:])
