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
from tqdm import tqdm

from utils.options import load_yaml
from datasets import build_dataset
from networks import build_network
from densifiers.nearest_anchor import NearestAnchorDensifier
from densifiers.base_densifier import DensifyContext
from metrics.geo_metric import calculate_geodesic_error


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
def evaluate(ext, dataset, limit=None):
    """Mean exact accuracy and mean sparse geodesic error over all pairs."""
    n_pairs = len(dataset) if limit is None else min(limit, len(dataset))
    accs, errs = [], []
    pbar = tqdm(range(n_pairs), desc='eval')
    for i in pbar:
        s0, s1 = dataset[i]['first'], dataset[i]['second']
        fx, fy = features(ext, s0), features(ext, s1)
        pred = (fy @ fx.t()).argmax(1)                     # y -> x match, (n,)
        tgt = torch.arange(pred.shape[0])
        accs.append((pred.cpu() == tgt).float().mean().item())
        dx = s0['sparse']['dist']                          # (n, n) geodesic on X's FPS points
        errs.append(dx[pred.cpu(), tgt].mean().item())     # dist(matched, true) per Y point
        pbar.set_postfix(acc=f'{np.mean(accs):.3f}', err=f'{np.mean(errs):.3f}')
    return float(np.mean(accs)), float(np.mean(errs))


@torch.no_grad()
def evaluate_independent(ext, dataset, thresh, limit=None):
    """Honest eval: FPS each shape independently (no bijective GT). The sparse Y->X map is
    lifted to the full mesh by nearest geodesic anchor, then scored against the template
    correspondence. Reports (PCK@thresh, mean geodesic error) over all template points."""
    n_pairs = len(dataset) if limit is None else min(limit, len(dataset))
    dens = NearestAnchorDensifier()
    errs = []
    pbar = tqdm(range(n_pairs), desc='eval (independent)')
    for i in pbar:
        s0, s1 = dataset[i]['first'], dataset[i]['second']
        fx, fy = features(ext, s0), features(ext, s1)
        pred = (fy @ fx.t()).argmax(1).cpu()               # sparse Y -> X (FPS-index space)
        ctx = DensifyContext(idx_x=s0['sparse']['idx'], idx_y=s1['sparse']['idx'],
                             n_x=s0['verts'].shape[0], n_y=s1['verts'].shape[0],
                             dist_x=s0['dist'], dist_y=s1['dist'])
        dense_p2p = dens.densify(pred, ctx).cpu().numpy()  # (N_y,) Y vertex -> X vertex
        e = calculate_geodesic_error(s0['dist'].cpu().numpy(), s0['corr'].cpu().numpy(),
                                     s1['corr'].cpu().numpy(), dense_p2p, return_mean=False)
        errs.append(e)
        cat = np.concatenate(errs)
        pbar.set_postfix(err=f'{cat.mean():.4f}', pck=f'{(cat <= thresh).mean():.3f}')
    cat = np.concatenate(errs)
    return float((cat <= thresh).mean()), float(cat.mean())


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
    p.add_argument('--limit', type=int, default=None, help='evaluate only the first N pairs')
    p.add_argument('--independent', action='store_true',
                   help='FPS each shape independently (no bijective GT); score by geodesic error')
    p.add_argument('--thresh', type=float, default=0.1, help='PCK geodesic-error threshold (--independent)')
    args = p.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ext, opt = load_extractor(args.config, args.ckpt, device, args.node_in)

    # honest test pairs (phase test, no self-pairs). The sparse FAUST dataset always returns
    # faces (used for the surface-mesh context in the viz).
    dataset = build_dataset(opt['datasets']['val'])
    n_eval = len(dataset) if args.limit is None else min(args.limit, len(dataset))

    if args.independent:
        dataset.independent_fps = True                 # unpaired FPS sets, geodesic-error score
        acc, err = evaluate_independent(ext, dataset, args.thresh, args.limit)
        print(f'FAUST independent-FPS matching over {n_eval} pairs: '
              f'PCK@{args.thresh} {acc:.4f} | mean geo error {err:.4f}')
    else:                                              # bijective FPS: exact-match accuracy
        acc, err = evaluate(ext, dataset, args.limit)
        print(f'FAUST sparse feature matching over {n_eval} pairs: '
              f'accuracy {acc:.4f} | mean sparse geo error {err:.4f}')

    if not args.no_vis:
        visualise(ext, dataset, args.vis_index)


if __name__ == '__main__':
    main()
