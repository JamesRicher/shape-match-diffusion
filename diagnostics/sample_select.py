"""ISOLATED, DELETE-SAFE DIAGNOSTIC -- does 'sample K, pick the smoothest' fix the flips?

Lives entirely under diagnostics/; all output goes to diagnostics/results/. Modifies no
tracked file. Delete the diagnostics/ folder to remove it and its data.

Tests the hypothesis: draw K sparse maps per pair (the sampler is stochastic through its
initial prior), score each map's map-energy, keep the lowest-energy one, and see whether its
symmetry-flip rate beats the single-sample baseline. Runs in the honest INDEPENDENT-FPS
regime, where the flips actually occur.

For each pair it reports four flip rates (fraction of surface with geodesic error > thresh,
measured by nearest-anchor / Voronoi propagation of the sparse map against exact template GT):

  baseline -- the FIRST draw (what you get today, one sample)
  selected -- the draw with minimum map-energy (the proposed fix)
  oracle   -- the MINIMUM flip rate over all K draws (needs GT; the CEILING any selection
              could reach -- how much diversity the sampler even has)
  mean     -- average flip rate over the K draws

Two energies are computed per draw (both cheap, sparse, no densification); --criterion picks
which one selection uses:
  dirichlet  -- graph-Dirichlet of the sparse map: sum_{i~j on Y} w_ij * d_X(m_i, m_j)^2.
                A flipped region spikes it at the seam (neighbouring Y points map far apart on X).
  distortion -- geodesic_distortion (metrics/map_metric.py): |d_Y(i,j) - d_X(m_i,m_j)|, the
                global-isometry violation a partial flip creates. (registered metric)

INTERPRETATION
  oracle << baseline               -> the sampler HAS diversity that can undo flips; the idea
                                      is viable in principle.
  selected ~ oracle (< baseline)   -> the energy criterion picks the good draws => IT WORKS.
  oracle ~ baseline                -> flips recur in every draw (deterministic); no selection
                                      can help => needs the training/coupling fix, not sampling.

Raise --temperature (>1) to widen the initial prior and force more diversity (off-distribution;
temperature 1.0 uses the model's own sampler unchanged).

USAGE
  python -m diagnostics.sample_select \
      -c configs/joint_gcn_diffusion/faust_matrix_diffusion_gcn_512_FMD.yaml \
         configs/joint_gcn_diffusion/faust_matrix_diffusion_gcn_512_redo.yaml \
         configs/joint_gcn_diffusion/runA_patch64_geo.yaml \
         configs/joint_gcn_diffusion/runB_patch48_euc.yaml \
      --k 20 --num-pairs 8

COMPUTE: K full DDIM samples per pair. Start small (--k 8 --num-pairs 4) to gauge timing.
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from datasets import build_dataset
from metrics.geo_metric import calculate_geodesic_error
from metrics.map_metric import geodesic_distortion
from models import build_model
from models.base_model import to_numpy
from train import autofill_feat_dim
from utils.options import load_yaml, resolve_experiment_paths
from utils.sinkhorn import cosine_alpha_bar, log_sinkhorn

_OUT_ROOT = os.path.join(os.path.dirname(__file__), 'results')
_ONE = torch.tensor(1.0)


def _build(config_path, checkpoint, device, split):
    """Load checkpoint + dataset as evaluate.py does, forced to independent-FPS sampling."""
    opt = load_yaml(config_path)
    if device is not None:
        opt['device'] = device
    opt['is_train'] = False
    resolve_experiment_paths(opt)
    ckpt = checkpoint or os.path.join(opt['path']['models'], 'final.pth')
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f'checkpoint not found: {ckpt}\nTrain first, or pass --checkpoint.')
    opt['path']['resume_state'] = ckpt
    opt['path']['resume'] = False
    dataset = build_dataset(opt['datasets'][split])
    dataset.independent_fps = True
    autofill_feat_dim(opt, int(dataset[0]['first']['feat'].shape[-1]))
    model = build_model(opt)
    model.eval()
    return model, dataset, opt, ckpt


@torch.no_grad()
def _sample_P0(model, F_x, F_y, D_x, D_y, temperature, eta):
    """One sampler draw -> P0 (n,n) DS matrix. temperature==1 & eta==0 uses the model's own
    (pure-DDIM) sampler; otherwise mirrors matrix_diffusion_model.sample() with a temperature-
    scaled initial prior and per-step DDIM<->DDPM stochastic noise (eta=0 = deterministic DDIM,
    eta=1 = full DDPM). Injecting noise throughout the trajectory (not just the prior) is the
    principled way to widen sample diversity when the reverse dynamics are contractive."""
    if temperature == 1.0 and eta == 0.0:
        return model.sample(F_x, F_y, D_x, D_y)[0]
    net = model.networks['denoiser']
    B, n = F_x.shape[0], F_x.shape[1]
    u = temperature * torch.randn(B, n, n, device=model.device)
    ts = torch.linspace(1.0, 0.0, model.sample_steps + 1, device=model.device)
    for i in range(model.sample_steps):
        t_i, t_prev = ts[i], ts[i + 1]
        P_t = log_sinkhorn(u, n_iters=model.proj_iters).exp()
        u0 = net(P_t, F_x, F_y, D_x, D_y, t_i.reshape(1).expand(B))
        ab_t = cosine_alpha_bar(t_i, model.schedule_s)
        ab_p = cosine_alpha_bar(t_prev, model.schedule_s)
        eps = (u - ab_t.sqrt() * u0) / (1.0 - ab_t).clamp_min(1e-8).sqrt()
        # DDIM<->DDPM interpolation (Song et al.): eta scales the injected noise sigma; the
        # deterministic drift coeff shrinks to keep the marginal variance consistent.
        sigma = eta * ((1.0 - ab_p) / (1.0 - ab_t).clamp_min(1e-8)).sqrt() \
                    * (1.0 - ab_t / ab_p.clamp_min(1e-8)).clamp_min(0.0).sqrt()
        coeff = (1.0 - ab_p - sigma ** 2).clamp_min(0.0).sqrt()
        u = ab_p.sqrt() * u0 + coeff * eps + sigma * torch.randn_like(u)
    return log_sinkhorn(u, n_iters=model.final_iters).exp()[0]


def _p2p(P0):
    """Hungarian snap of a DS matrix -> (n,) sparse Y-index -> sparse X-index (numpy)."""
    row, col = linear_sum_assignment(-to_numpy(P0))
    p = np.empty(P0.shape[0], dtype=np.int64)
    p[row] = col
    return p


def _dirichlet(p2p, dist_y_s, dist_x_s, knn):
    """Graph-Dirichlet of the sparse map: kNN graph on Y, edge weight * squared X-distance
    of the mapped endpoints. Seams (neighbouring Y -> far-apart X) dominate."""
    n = dist_y_s.shape[0]
    k = min(knn, n - 1)
    nn_d, nn_idx = torch.topk(dist_y_s, k + 1, largest=False)          # incl. self col 0
    nn_d, nn_idx = nn_d[:, 1:], nn_idx[:, 1:]
    sigma = nn_d.median().clamp_min(1e-8)
    w = torch.exp(-(nn_d ** 2) / (sigma ** 2))                          # (n,k)
    p = torch.as_tensor(p2p, device=dist_x_s.device)
    dX = dist_x_s[p][:, p]                                              # (n,n) d_X(m_i, m_j)
    return float((w * dX.gather(1, nn_idx) ** 2).sum())


def run_one(cfg, checkpoint, device, split, indices_arg, num_pairs, seed, K, temperature, eta,
            criterion, knn, thresh):
    model, dataset, opt, ckpt = _build(cfg, checkpoint, device, split)
    name = opt['name']
    if indices_arg:
        idxs = [i for i in indices_arg if 0 <= i < len(dataset)]
    else:
        # seeded random subset -- identical across configs (same seed + same dataset length),
        # so multi -c runs stay comparable while sampling the pair grid representatively.
        n = min(num_pairs, len(dataset))
        idxs = sorted(np.random.default_rng(seed).choice(len(dataset), size=n, replace=False).tolist())

    rows = []
    for i in tqdm(idxs, desc=f'{name} sample x{K}'):
        data = dataset[int(i)]
        x, y = data['first'], data['second']
        F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)

        idx_x = to_numpy(x['sparse']['idx']).astype(np.int64)
        idx_y = to_numpy(y['sparse']['idx']).astype(np.int64)
        dist_x_full, dist_y_full = to_numpy(x['dist']), to_numpy(y['dist'])
        corr_x = to_numpy(x['corr']).astype(np.int64)
        corr_y = to_numpy(y['corr']).astype(np.int64)
        nearest_anchor = dist_y_full[:, idx_y].argmin(axis=1)          # (N_y,) -> anchor, fixed per pair
        dist_x_s = x['sparse']['dist'].float()
        dist_y_s = y['sparse']['dist'].float()

        flips, e_dir, e_dis = [], [], []
        for _ in range(K):
            p2p = _p2p(_sample_P0(model, F_x, F_y, D_x, D_y, temperature, eta))
            vor = idx_x[p2p[nearest_anchor]]                           # Voronoi-propagated dense map
            err = calculate_geodesic_error(dist_x_full, corr_x, corr_y, vor, return_mean=False)
            flips.append(float(np.mean(err > thresh)))
            pt = torch.as_tensor(p2p)
            e_dir.append(_dirichlet(p2p, dist_y_s, dist_x_s, knn))
            e_dis.append(float(geodesic_distortion(pt, dist_y_s, dist_x_s, _ONE, _ONE)[1]))

        flips = np.array(flips)
        energy = np.array(e_dir if criterion == 'dirichlet' else e_dis)
        rows.append({'pair': int(i),
                     'baseline': float(flips[0]),
                     'selected': float(flips[int(energy.argmin())]),
                     'oracle': float(flips.min()),
                     'mean': float(flips.mean())})

    agg = {k: float(np.mean([r[k] for r in rows])) for k in ('baseline', 'selected', 'oracle', 'mean')}
    summary = {'name': name, 'checkpoint': ckpt, 'n_pairs': len(idxs), 'K': K,
               'temperature': temperature, 'eta': eta, 'criterion': criterion,
               'flip_thresh': thresh, 'aggregate': agg, 'per_pair': rows}
    out_dir = os.path.join(_OUT_ROOT, f'{name}_sampleselect')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'sample_select.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def _print_table(summaries, criterion, K):
    s0 = summaries[0]
    print(f"\nflip rate (fraction > thresh) | select by min {criterion} over K={K} draws "
          f"(temperature={s0['temperature']}, eta={s0['eta']})\n")
    head = f"{'experiment':30s}{'baseline':>10s}{'selected':>10s}{'oracle':>10s}{'mean':>10s}"
    print(head); print('-' * len(head))
    for s in summaries:
        a = s['aggregate']
        print(f"{s['name']:30s}{a['baseline']*100:9.1f}%{a['selected']*100:9.1f}%"
              f"{a['oracle']*100:9.1f}%{a['mean']*100:9.1f}%")
    print("\noracle<<baseline => sampler has flip-fixing diversity;  selected~oracle => criterion works;")
    print("oracle~baseline  => flips deterministic across draws => sampling can't help (needs a retrain/coupling fix).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-c', '--config', nargs='+', required=True, help='one or more training configs')
    p.add_argument('--checkpoint', default=None, help='checkpoint override (only with a single -c)')
    p.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--k', type=int, default=20, help='samples drawn per pair')
    p.add_argument('--num-pairs', type=int, default=8, help='pairs to probe (seeded random subset)')
    p.add_argument('--seed', type=int, default=0,
                   help='seed for the random pair subset; SAME across all -c configs so they stay comparable')
    p.add_argument('--pair-indices', type=int, nargs='+', default=None, help='explicit pair indices (overrides subset)')
    p.add_argument('--temperature', type=float, default=1.0, help='initial-prior scale (>1 = more diverse)')
    p.add_argument('--eta', type=float, default=0.0,
                   help='per-step DDIM<->DDPM stochasticity (0 = deterministic DDIM, 1 = full DDPM)')
    p.add_argument('--criterion', default='dirichlet', choices=['dirichlet', 'distortion'],
                   help='map-energy used to select the sample')
    p.add_argument('--knn', type=int, default=8, help='neighbours for the Dirichlet graph')
    p.add_argument('--flip-thresh', type=float, default=0.20, help='geodesic-error flip threshold')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    args = p.parse_args()
    if args.checkpoint and len(args.config) > 1:
        p.error('--checkpoint can only be used with a single -c')

    summaries = []
    for cfg in args.config:
        summaries.append(run_one(cfg, args.checkpoint, args.device, args.split,
                                 args.pair_indices, args.num_pairs, args.seed, args.k, args.temperature,
                                 args.eta, args.criterion, args.knn, args.flip_thresh))
    _print_table(summaries, args.criterion, args.k)
    print(f"\nper-experiment JSON under: {_OUT_ROOT}/<name>_sampleselect/")


if __name__ == '__main__':
    main()
