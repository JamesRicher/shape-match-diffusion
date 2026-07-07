"""Visualise a shape-matching model in TensorBoard.

Two things are written to a TensorBoard log dir:

  1. The model *graph* (schematic) via ``SummaryWriter.add_graph`` — the encoder
     traced on a real pair of feature tensors pulled from the dataset.
  2. A few *meshes* via ``SummaryWriter.add_mesh`` — the 3D geometry (verts +
     faces) of the first shapes in the split, so you can spin them around in the
     TensorBoard "Mesh" tab.

Run with, e.g.:

    python visualise_tb.py -c configs/smal_shape_matching.yaml
    tensorboard --logdir experiments/<name>/tb_vis

Then open the "Graphs" and "Mesh" tabs in the browser.
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from datasets import build_dataset
from networks import build_network
from utils.options import load_yaml, resolve_experiment_paths


class _EncoderGraphWrapper(nn.Module):
    """Thin wrapper so ``add_graph`` traces a clean two-tensor signature.

    The encoder's ``forward`` takes several optional ``*_bias=None`` arguments;
    tracing is happiest with only the positional feature tensors, so we hide the
    rest here.
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, feat_x: torch.Tensor, feat_y: torch.Tensor):
        return self.encoder(feat_x, feat_y)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualise a shape-matching model (graph + meshes) in TensorBoard.")
    parser.add_argument('-c', '--config', required=True,
                        help='path to the YAML config (as used for training)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help='which dataset split to pull sample shapes from (default: test)')
    parser.add_argument('--logdir', default=None,
                        help='TensorBoard log dir (default: experiments/<name>/tb_vis)')
    parser.add_argument('--num-meshes', type=int, default=4,
                        help='how many shapes to log with add_mesh (default: 4)')
    parser.add_argument('--device', default=None,
                        help="'cuda' / 'cpu'; auto-detected when omitted")
    return parser.parse_args()


def _pick_device(arg_device):
    if arg_device:
        return torch.device(arg_device)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_model_graph(writer, opt, sample_pair, device):
    """Trace the encoder on one real feature pair and log its graph."""
    feat_x = sample_pair['first']['feat']
    feat_y = sample_pair['second']['feat']

    # match the encoder input dim to the actual per-vertex feature dim (as in train.py)
    net_opt = dict(opt['networks']['encoder'])
    net_opt['in_dim'] = int(feat_x.shape[-1])

    encoder = build_network(net_opt).to(device).eval()
    model = _EncoderGraphWrapper(encoder).to(device).eval()

    # add batch dim: forward expects (B, N, in_dim)
    fx = feat_x.unsqueeze(0).to(device)
    fy = feat_y.unsqueeze(0).to(device)

    with torch.no_grad():
        writer.add_graph(model, (fx, fy))
    print(f'Logged model graph (in_dim={net_opt["in_dim"]}, '
          f'Nx={fx.shape[1]}, Ny={fy.shape[1]}).')


def add_meshes(writer, dataset, num_meshes):
    """Log the geometry of the first ``num_meshes`` shapes with add_mesh.

    The pair dataset wraps a single-shape dataset in ``dataset.dataset``; we read
    individual shapes from there so we log distinct meshes rather than pairs.
    """
    single = getattr(dataset, 'dataset', dataset)
    n = min(num_meshes, len(single))
    for i in range(n):
        item = single[i]
        if 'faces' not in item:
            raise KeyError(
                "dataset items have no 'faces' (need ret_faces=True) — cannot add_mesh")
        verts = item['verts'].float().unsqueeze(0)      # (1, N, 3)
        faces = item['faces'].int().unsqueeze(0)        # (1, F, 3)

        # colour vertices by height (y) so the shading reads well in the viewer
        y = verts[..., 1]
        y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
        colors = torch.zeros_like(verts)
        colors[..., 0] = (y_norm * 255)                 # R ramps with height
        colors[..., 2] = ((1 - y_norm) * 255)           # B ramps inversely
        colors = colors.byte()

        tag = f"mesh/{item.get('name', f'shape_{i}')}"
        writer.add_mesh(tag, vertices=verts, colors=colors, faces=faces)
        print(f'Logged mesh "{tag}"  (V={verts.shape[1]}, F={faces.shape[1]}).')


def main():
    args = parse_args()
    device = _pick_device(args.device)

    opt = load_yaml(args.config)
    resolve_experiment_paths(opt)

    logdir = args.logdir or os.path.join(opt['path']['experiment_root'], 'tb_vis')
    os.makedirs(logdir, exist_ok=True)

    dataset = build_dataset(opt['datasets'][args.split])
    writer = SummaryWriter(log_dir=logdir)
    try:
        add_model_graph(writer, opt, dataset[0], device)
        add_meshes(writer, dataset, args.num_meshes)
    finally:
        writer.close()

    print(f'\nDone. View with:\n    tensorboard --logdir {logdir}')


if __name__ == '__main__':
    main()
