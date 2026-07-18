"""Visualise a trained sparse matcher's correspondence as matched GCN patches.

For a joint model (MatrixDiffusionModel with a GCNFeatureExtractor under
networks['extractor']), each sparse FPS point is described by a *patch*: the
`patch_size` full-mesh vertices nearest the point by geodesic distance — exactly the
receptive field the extractor pools over (networks/gcn_feature_extractor.py). This
viewer draws a random test pair side by side (A = first/X, B = second/Y), runs the
model to get the sparse point-to-point map, and paints each matched pair of patches
in a shared colour. Every patch is rendered as the actual `patch_size` vertices the
extractor gathered, so the coloured blob on each shape is the true patch extent.

A slider controls how many matched patches are shown (too many at once is unreadable);
patches are added in FPS order, so growing the slider spreads landmarks out rather than
clustering them. A button resamples the pair.

Example:
    python -m vis.gcn_patch_correspondence_vis \
        -c configs/joint_gcn_diffusion/faust_matrix_diffusion_gcn.yaml
"""
import argparse
import os
import sys

import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import torch

from datasets import build_dataset
from models import build_model
from train import autofill_feat_dim
from utils.options import load_yaml, resolve_experiment_paths
from vis.fps_neighbourhood_vis import _hsv_palette, _to_numpy

BASE_COLOR = (0.85, 0.85, 0.85)     # unclaimed mesh surface
CENTER_COLOR = np.array([0.05, 0.05, 0.05], dtype=np.float32)   # FPS centres


def build_opt(args):
    """Load the config for a net-only eval pass and point it at the checkpoint."""
    opt = load_yaml(args.config)
    if args.name is not None:
        opt['name'] = args.name
    if args.device is not None:
        opt['device'] = args.device
    opt['is_train'] = False
    resolve_experiment_paths(opt)
    ckpt = args.checkpoint or os.path.join(opt['path']['models'], 'final.pth')
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f'checkpoint not found: {ckpt}\nTrain first, or pass --checkpoint <path>.')
    opt['path']['resume_state'] = ckpt
    opt['path']['resume'] = False       # net-only load (extractor + denoiser)
    return opt, ckpt


def patch_indices(dist: torch.Tensor, idx: torch.Tensor, patch_size: int) -> np.ndarray:
    """The full-mesh vertex indices in each FPS point's patch, matching
    GCNFeatureExtractor.build_patches: the `patch_size` geodesically nearest vertices
    (column 0 is the centre itself). dist (N,N), idx (n,) -> (n, p) numpy."""
    idx = idx.long()
    p = min(patch_size, dist.shape[0])
    _, pidx = torch.topk(dist[idx], p, dim=-1, largest=False)    # (n, p)
    return pidx.cpu().numpy()


class PatchCorrespondenceVis:
    def __init__(self, model, dataset, num_patches, seed):
        self.model = model
        self.dataset = dataset
        self.rng = np.random.default_rng(seed)
        self.patch_size = int(model.networks['extractor'].patch_size)
        self.num_patches = int(num_patches)
        self.pair = None            # per-shape draw data (verts/faces/patch idx/centres)
        self.p2p = None             # sparse Y -> X map
        self.order = None           # FPS-order display order over the matched pairs
        self._resample_pair()

    # ----------------------------------------------------------------- sampling

    def _resample_pair(self):
        idx = int(self.rng.integers(len(self.dataset)))
        item = self.dataset[idx]

        # model's sparse correspondence: for Y point j, matched X point p2p[j]
        self.p2p = _to_numpy(self.model.validate_single(item)).astype(np.int64)

        ps.remove_all_structures()
        shapes, offset = [], np.zeros(3, dtype=np.float32)
        for tag, key in (("A", "first"), ("B", "second")):
            shape = item[key]
            verts = _to_numpy(shape['verts']).astype(np.float32) + offset
            faces = _to_numpy(shape['faces']).astype(np.int64)
            pidx = patch_indices(shape['dist'], shape['sparse']['idx'], self.patch_size)
            centres = _to_numpy(shape['sparse']['verts']).astype(np.float32) + offset

            extent = float(verts[:, 0].max() - verts[:, 0].min())
            offset = offset + np.array([extent * 1.4, 0.0, 0.0], dtype=np.float32)
            ps.register_surface_mesh(f"shape_{tag} [{shape.get('name', idx)}]",
                                     verts, faces, smooth_shade=True, color=BASE_COLOR)
            shapes.append({"tag": tag, "verts": verts, "patch_idx": pidx,
                           "centres": centres})
        self.pair = shapes

        # Display in FPS order: the sparse tokens are stored in FPS order (dataset K), so
        # taking a prefix adds well-spread landmarks as the slider grows (1 -> 2 jumps far).
        self.order = np.arange(self.p2p.shape[0])
        self._repaint()

    # -------------------------------------------------------------------- draw

    def max_patches(self) -> int:
        return int(self.p2p.shape[0])

    def _repaint(self):
        a, b = self.pair
        n = min(self.num_patches, self.max_patches())
        chosen_j = self.order[:n]                 # Y point indices on display (FPS order)
        palette = _hsv_palette(n)

        # aggregate each shape's displayed patches into one coloured point cloud, so a
        # matched pair (X patch p2p[j], Y patch j) shares palette[k]. Patches overlap;
        # a shared vertex simply gets drawn once per patch it belongs to.
        for shape, pick in ((a, lambda j: self.p2p[j]), (b, lambda j: j)):
            pts, cols = [], []
            for k, j in enumerate(chosen_j):
                pidx = shape["patch_idx"][pick(j)]         # (p,) vertices in this patch
                pts.append(shape["verts"][pidx])
                cols.append(np.tile(palette[k], (pidx.shape[0], 1)))
            pts = np.concatenate(pts, 0) if pts else np.zeros((0, 3), np.float32)
            cols = np.concatenate(cols, 0) if cols else np.zeros((0, 3), np.float32)
            pc = ps.register_point_cloud(f"patches_{shape['tag']}", pts, radius=0.004)
            pc.add_color_quantity("match", cols.astype(np.float32), enabled=True)

            # FPS centres of the displayed patches, for orientation
            cen = ps.register_point_cloud(f"centres_{shape['tag']}",
                                          shape["centres"][[pick(j) for j in chosen_j]],
                                          radius=0.006)
            cen.add_color_quantity("centre", np.tile(CENTER_COLOR, (n, 1)), enabled=True)

    # --------------------------------------------------------------------- UI

    def ui_callback(self):
        psim.TextUnformatted("Model correspondence — matched GCN patches share a colour")
        psim.TextUnformatted(f"patch size (extractor): {self.patch_size} verts")
        psim.Separator()

        max_n = self.max_patches()
        changed, new_n = psim.SliderInt("num patches", self.num_patches, 1, max_n)
        if changed:
            self.num_patches = int(new_n)
            self._repaint()
        psim.TextUnformatted(f"showing {min(self.num_patches, max_n)} of {max_n} "
                             f"matched patches (FPS order)")

        if psim.Button("resample pair"):
            self._resample_pair()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--config', required=True,
                        help='YAML config of the trained joint-GCN model')
    parser.add_argument('-n', '--name', default=None, help='override experiment name')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint (default: experiments/<name>/models/final.pth)')
    parser.add_argument('--device', default=None, help="'cuda'/'cpu' (auto when omitted)")
    parser.add_argument('--num-patches', type=int, default=12,
                        help='initial number of matched patch pairs to display')
    parser.add_argument('--seed', type=int, default=None, help='RNG seed for pair/patch choice')
    args = parser.parse_args(argv)

    opt, ckpt = build_opt(args)

    test_set = build_dataset(opt['datasets']['test'])
    test_set.independent_fps = False        # need the bijective sparse tokens + centres
    autofill_feat_dim(opt, int(test_set[0]['first']['feat'].shape[-1]))

    model = build_model(opt)
    if 'extractor' not in model.networks:
        raise ValueError("this config has no networks['extractor']; "
                         "it is not a joint-GCN model.")
    model.eval()
    print(f'Loaded "{opt["name"]}" (checkpoint: {ckpt}, device: {model.device}).')

    ps.init()
    ps.set_up_dir("neg_y_up" if getattr(test_set, "flip_up", False) else "y_up")
    viewer = PatchCorrespondenceVis(model, test_set, num_patches=args.num_patches,
                                    seed=args.seed)
    ps.set_user_callback(viewer.ui_callback)
    ps.show()


if __name__ == "__main__":
    main(sys.argv[1:])
