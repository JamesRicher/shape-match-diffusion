"""Interactively inspect texture transfer for a completed experiment in polyscope.

Loads a trained checkpoint (like evaluate.py), samples a few random test pairs,
runs the model to get the p2p map, and shows each pair as a row of three meshes:

    source X  |  target Y (hard p2p)  |  target Y (smoothed spectral)

Two qualitative views are attached to every mesh (toggle via the checkbox or
the quantities list): the repo's texture.png applied as a real texture
(per-pixel lookup through each mesh's UVs) and ULRSSM-style position-coded
correspondence colors — matching regions share the same color. Both use the
same hard/smoothed transfers as the evaluation figures (utils/texture_util.py).
A "resample pairs" button draws a new random set.

Example:
    python -m vis.texture_transfer_vis -c configs/smal_shape_matching.yaml
"""
import argparse

import numpy as np
import polyscope as ps
import polyscope.imgui as psim

from datasets import build_dataset
from models import build_model
from models.base_model import to_numpy
from utils.data_utils import sqrt_surface_area
from utils.options import load_yaml, resolve_experiment_paths
from utils.texture_util import (DEFAULT_TEXTURE_FILE, create_colormap,
                                generate_tex_coords, hard_uv_transfer,
                                load_texture, smoothed_uv_transfer)


def parse_args():
    parser = argparse.ArgumentParser(
        description='View texture-transfer results of a trained model in polyscope.')
    parser.add_argument('-c', '--config', required=True,
                        help='path to the YAML config used for training')
    parser.add_argument('-n', '--name', default=None,
                        help='override experiment name (subdir of experiments/)')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint to load (default: experiments/<name>/models/final.pth)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help='dataset split to sample pairs from (default: test)')
    parser.add_argument('--num-pairs', type=int, default=3,
                        help='pairs shown at once, one row each (default: 3)')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed for pair sampling (default: random)')
    parser.add_argument('--device', default=None,
                        help="'cuda' / 'cpu'; auto-detected when omitted")
    return parser.parse_args()


def build_eval_model(args):
    """Build the dataset + trained model exactly as evaluate.py does."""
    import os
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
    opt['path']['resume'] = False

    dataset = build_dataset(opt['datasets'][args.split])
    opt['networks']['encoder']['in_dim'] = int(dataset[0]['first']['feat'].shape[-1])
    model = build_model(opt)  # constructor loads ckpt (net-only, is_train=False)
    model.eval()
    return model, dataset, opt


class TextureTransferVis:
    """Polyscope UI: rows of (source, hard, smoothed) textured meshes."""

    def __init__(self, model, dataset, num_pairs, seed, texture):
        self.model = model
        self.dataset = dataset
        self.num_pairs = int(num_pairs)
        self.rng = np.random.default_rng(seed)
        self.texture = texture
        self.flip_up = bool(getattr(dataset, 'flip_up', False))
        self.show_texture = True     # False = position-coded correspondence colors
        self.pair_info = []          # [(name_x, name_y, mean geo err or None)]
        self.meshes = []             # [(ps mesh, uv, colors)] for view toggling
        self._resample()

    # ------------------------------------------------------------- transfers

    def _compute_pair(self, idx):
        """Run the model on pair `idx`; return meshes, transferred UVs/colors
        and the pair's mean geodesic error."""
        data = self.dataset[int(idx)]
        dx, dy = data['first'], data['second']
        p2p = to_numpy(self.model.validate_single(data))     # Y -> X, [Vy]

        verts_x, faces_x = to_numpy(dx['verts']), to_numpy(dx['faces'])
        verts_y, faces_y = to_numpy(dy['verts']), to_numpy(dy['faces'])

        # both per-vertex signals share the same hard/smoothed transfer maps
        uv_x = generate_tex_coords(verts_x)
        col_x = create_colormap(verts_x)
        uv_hard, col_hard = hard_uv_transfer(uv_x, p2p), hard_uv_transfer(col_x, p2p)
        uv_smooth = col_smooth = None
        if all(k in dx for k in ('evecs', 'evecs_trans')) and \
           all(k in dy for k in ('evecs', 'evecs_trans')):
            spectral = (to_numpy(dx['evecs']), to_numpy(dy['evecs']),
                        to_numpy(dx['evecs_trans']), to_numpy(dy['evecs_trans']))
            uv_smooth = smoothed_uv_transfer(uv_x, p2p, *spectral)
            col_smooth = smoothed_uv_transfer(col_x, p2p, *spectral)

        # per-pair mean geodesic error (same normalization as validation)
        err = None
        if 'geo_error' in self.model.metrics and 'dist' in dx:
            geo_err = self.model.metrics['geo_error'](
                to_numpy(dx['dist']), to_numpy(dx['corr']), to_numpy(dy['corr']),
                p2p, return_mean=False)
            if 'mass' in dx:
                geo_err = geo_err / to_numpy(sqrt_surface_area(dx['mass']))
            err = float(geo_err.mean())

        name_x = dx.get('name', f'{idx}_x')
        name_y = dy.get('name', f'{idx}_y')
        return (name_x, name_y, err,
                (verts_x, faces_x, uv_x, col_x),
                (verts_y, faces_y, uv_hard, col_hard),
                None if uv_smooth is None else (verts_y, faces_y, uv_smooth, col_smooth))

    # ----------------------------------------------------------- registration

    def _register_mesh(self, name, verts, faces, uv, colors, offset):
        """One mesh with both qualitative views: the texture map and the
        position-coded correspondence colors (toggle in the quantities list)."""
        mesh = ps.register_surface_mesh(name, verts + offset, faces, smooth_shade=True)
        mesh.add_parameterization_quantity('uv', uv, defined_on='vertices',
                                           coords_type='unit')
        mesh.add_color_quantity('texture', self.texture, defined_on='texture',
                                param_name='uv', image_origin='upper_left',
                                enabled=self.show_texture)
        mesh.add_color_quantity('corr colors', np.clip(colors, 0.0, 1.0),
                                defined_on='vertices',
                                enabled=not self.show_texture)
        self.meshes.append((mesh, uv, colors))
        return mesh

    def _resample(self):
        n = min(self.num_pairs, len(self.dataset))
        indices = self.rng.choice(len(self.dataset), size=n, replace=False)

        ps.remove_all_structures()
        self.pair_info = []
        self.meshes = []
        row_offset = np.zeros(3, dtype=np.float32)
        for r, idx in enumerate(indices):
            name_x, name_y, err, src, hard, smooth = self._compute_pair(idx)
            self.pair_info.append((name_x, name_y, err))

            # lay panels out along x, rows along z; spacing from this row's extent
            extent = float(src[0][:, 0].max() - src[0][:, 0].min())
            col = np.array([extent * 1.4, 0.0, 0.0], dtype=np.float32)

            self._register_mesh(f'r{r} src [{name_x}]', *src, row_offset)
            self._register_mesh(f'r{r} hard [{name_x}->{name_y}]', *hard,
                                row_offset + col)
            if smooth is not None:
                self._register_mesh(f'r{r} smooth [{name_x}->{name_y}]', *smooth,
                                    row_offset + 2 * col)
            row_offset = row_offset + np.array([0.0, 0.0, extent * 1.4],
                                               dtype=np.float32)

    def _apply_view_mode(self):
        """Switch every mesh between the texture map and correspondence colors.

        Re-adding a same-named quantity updates it in place, so this just flips
        which of the two is enabled.
        """
        for mesh, uv, colors in self.meshes:
            mesh.add_color_quantity('texture', self.texture, defined_on='texture',
                                    param_name='uv', image_origin='upper_left',
                                    enabled=self.show_texture)
            mesh.add_color_quantity('corr colors', np.clip(colors, 0.0, 1.0),
                                    defined_on='vertices',
                                    enabled=not self.show_texture)

    # -------------------------------------------------------------------- UI

    def callback(self):
        psim.TextUnformatted('columns: source X | target Y hard p2p | target Y smoothed')
        for r, (name_x, name_y, err) in enumerate(self.pair_info):
            line = f'row {r}: {name_x} -> {name_y}'
            if err is not None:
                line += f'   (mean geo err: {err:.4f})'
            psim.TextUnformatted(line)
        psim.Separator()

        changed, show_texture = psim.Checkbox('texture view (off = corr colors)',
                                              self.show_texture)
        if changed:
            self.show_texture = show_texture
            self._apply_view_mode()

        changed, self.num_pairs = psim.InputInt('num pairs', self.num_pairs)
        self.num_pairs = max(1, min(self.num_pairs, len(self.dataset)))
        if psim.Button('resample pairs'):
            self._resample()


def main():
    args = parse_args()
    model, dataset, opt = build_eval_model(args)
    texture = load_texture(DEFAULT_TEXTURE_FILE)

    ps.init()
    ps.set_up_dir('neg_y_up' if getattr(dataset, 'flip_up', False) else 'y_up')
    ps.set_ground_plane_mode('none')

    vis = TextureTransferVis(model, dataset, args.num_pairs, args.seed, texture)
    ps.set_user_callback(vis.callback)
    ps.show()


if __name__ == '__main__':
    main()
