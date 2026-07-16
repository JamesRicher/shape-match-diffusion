"""Evaluate a pretrained GCN feature extractor on FAUST: sparse feature-matching accuracy
plus a Polyscope view colouring the sparse points by their features.

    python evaluate_extractor.py                          # accuracy + visualise pair 0
    python evaluate_extractor.py --no_vis                 # accuracy only (headless)
    python evaluate_extractor.py --vis_index 5 --ckpt extractor_faust.pth

Matching is nearest-neighbour in the (L2-normalised) feature space. The sparse GT is the
identity permutation over FPS points (point i of X <-> point i of Y), so a Y point j is
correct iff argmax_i <f_y[j], f_x[i]> == j. We report exact accuracy and the mean sparse
geodesic error (distance on X between the matched and the true vertex).
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from utils.options import load_yaml
from datasets import build_dataset
from networks import build_network


def load_extractor(config, ckpt, device, node_in=None):
    opt = load_yaml(config)
    cfg = dict(opt['networks']['extractor'])
    if node_in is not None:
        cfg['node_in'] = node_in
    ext = build_network(cfg).to(device).eval()
    state = torch.load(ckpt, map_location='cpu')
    sd = state['networks']['extractor'] if 'networks' in state else state
    ext.load_state_dict(sd)
    return ext, opt


@torch.no_grad()
def features(ext, shape):
    """L2-normalised sparse features (n, d) for one shape."""
    f = ext.extract(shape['verts'], shape['dist'], shape['sparse']['idx'])[0]
    return F.normalize(f, dim=-1)


@torch.no_grad()
def evaluate(ext, dataset):
    """Mean exact accuracy and mean sparse geodesic error over all pairs."""
    accs, errs = [], []
    for i in range(len(dataset)):
        s0, s1 = dataset[i]['first'], dataset[i]['second']
        fx, fy = features(ext, s0), features(ext, s1)
        pred = (fy @ fx.t()).argmax(1)                     # y -> x match, (n,)
        tgt = torch.arange(pred.shape[0])
        accs.append((pred.cpu() == tgt).float().mean().item())
        dx = s0['sparse']['dist']                          # (n, n) geodesic on X's FPS points
        errs.append(dx[pred.cpu(), tgt].mean().item())     # dist(matched, true) per Y point
    return float(np.mean(accs)), float(np.mean(errs))


def _pca_rgb(fx, fy):
    """Joint PCA of both shapes' features -> per-point RGB in [0,1] (shared basis so matched
    points get matched colours). Returns (cx, cy) numpy arrays."""
    Z = torch.cat([fx, fy], 0)
    Zc = Z - Z.mean(0, keepdim=True)
    _, _, V = torch.linalg.svd(Zc, full_matrices=False)
    proj = Zc @ V[:3].t()                                  # (m, 3) top-3 principal directions
    lo, hi = proj.amin(0, keepdim=True), proj.amax(0, keepdim=True)
    rgb = ((proj - lo) / (hi - lo).clamp_min(1e-8)).cpu().numpy()
    n = fx.shape[0]
    return rgb[:n], rgb[n:]


def visualise(ext, dataset, index, gap=1.2):
    import polyscope as ps
    s0, s1 = dataset[index]['first'], dataset[index]['second']
    cx, cy = _pca_rgb(features(ext, s0), features(ext, s1))

    vx, vy = s0['verts'].cpu().numpy(), s1['verts'].cpu().numpy()
    sx, sy = s0['sparse']['verts'].cpu().numpy(), s1['sparse']['verts'].cpu().numpy()
    off = np.array([(vx[:, 0].max() - vx[:, 0].min()) + gap, 0.0, 0.0])  # side-by-side

    ps.init()
    if 'faces' in s0 and 'faces' in s1:
        fa, fb = s0['faces'].cpu().numpy(), s1['faces'].cpu().numpy()
        ps.register_surface_mesh('X mesh', vx, fa, color=(0.85, 0.85, 0.85), transparency=0.35)
        ps.register_surface_mesh('Y mesh', vy + off, fb, color=(0.85, 0.85, 0.85), transparency=0.35)
    px = ps.register_point_cloud('X sparse', sx)
    py = ps.register_point_cloud('Y sparse', sy + off)
    px.add_color_quantity('feature PCA', cx, enabled=True)
    py.add_color_quantity('feature PCA', cy, enabled=True)
    print(f'visualising pair {index} ({dataset[index]["first"]["name"]} <-> '
          f'{dataset[index]["second"]["name"]}); matched points should share a colour.')
    ps.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default='configs/faust_matrix_diffusion_gcn.yaml')
    p.add_argument('--ckpt', default='extractor_faust.pth')
    p.add_argument('--node_in', default=None, help='override extractor node_in (must match training)')
    p.add_argument('--device', default=None)
    p.add_argument('--no_vis', action='store_true', help='accuracy only (no Polyscope window)')
    p.add_argument('--vis_index', type=int, default=0, help='which test pair to visualise')
    args = p.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ext, opt = load_extractor(args.config, args.ckpt, device, args.node_in)

    # honest test pairs (phase test, no self-pairs). The sparse FAUST dataset always returns
    # faces (used for the surface-mesh context in the viz).
    dataset = build_dataset(opt['datasets']['val'])

    acc, err = evaluate(ext, dataset)
    print(f'FAUST sparse feature matching over {len(dataset)} pairs: '
          f'accuracy {acc:.4f} | mean sparse geo error {err:.4f}')

    if not args.no_vis:
        visualise(ext, dataset, args.vis_index)


if __name__ == '__main__':
    main()
