"""Fast, training-free-ish sanity checks for a feature extractor BEFORE a full pretrain.

Config-driven; works for any registered extractor (GCN or DiffusionNet -- the extractor's
`needs_operators` flag selects the call signature). Runs two cheap checks and prints PASS/FAIL
(non-zero exit on any failure), so a broken wiring/gradient path is caught in seconds rather
than after an 8-hour run.

  1. GRADIENT check: one forward+backward on a single pair. Every parameter must receive a
     finite, non-None gradient, and the features must be finite and not collapsed (per-point
     variance > 0). Catches a disconnected spectral path (e.g. diffusion_time not in the graph).
  2. OVERFIT check: a few FIXED pairs (cached once, so the sparse sampling is frozen) trained
     for a few hundred steps. Train accuracy must climb toward ~1. If the extractor cannot
     overfit a handful of pairs, the implementation is broken -- no point pretraining.

    python debug/feature_extractor/sanity_extractor.py -c debug/feature_extractor/configs/faust_diffusionnet.yaml
    ... --n_pairs 4 --steps 300 --pass_acc 0.9 --device cuda

Lives under debug/ and imports only shared helpers; delete-safe.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from datasets import build_dataset          # noqa: E402
from metrics.geo_metric import calculate_geodesic_error  # noqa: E402
from networks import build_network          # noqa: E402
from pretrain_extractor import contrastive_loss  # noqa: E402
from utils.options import load_yaml         # noqa: E402


def _extract(ext, shape):
    """Sparse features (n, d) for one shape, dispatching on the extractor's call convention."""
    idx = shape['sparse']['idx']
    if getattr(ext, 'needs_operators', False):
        return ext.extract(shape, idx)[0]                       # DiffusionNet: reads cached ops
    return ext.extract(shape['verts'], shape['dist'], idx)[0]   # GCN: (verts, dist, idx)


def _dense_feats(ext, shape, chunk=2048):
    """L2-normalised per-vertex features (V, d) over the whole mesh."""
    if getattr(ext, 'needs_operators', False):
        f = ext.extract_dense(shape)                            # DiffusionNet: one global forward
    else:                                                       # GCN: one patch per vertex, chunked
        V = shape['verts'].shape[0]
        f = torch.cat([ext.extract(shape['verts'], shape['dist'],
                                   torch.arange(lo, min(lo + chunk, V)))[0]
                       for lo in range(0, V, chunk)], dim=0)
    return F.normalize(f, dim=-1)


@torch.no_grad()
def dense_accuracy(ext, pairs, thresh, chunk=4096):
    """HONEST readout: per-vertex feature NN over the FULL mesh (not the 256 GT-paired sparse
    points), scored against template GT. Unlike the bijective sparse acc, the full candidate
    pool exposes symmetry flips. Returns (PCK@thresh, mean geodesic error) over the pairs."""
    ext.eval()
    errs = []
    for pair in pairs:
        s0, s1 = pair['first'], pair['second']
        fx, fy = _dense_feats(ext, s0), _dense_feats(ext, s1)
        p2p = torch.cat([(fy[lo:lo + chunk] @ fx.t()).argmax(1)
                         for lo in range(0, fy.shape[0], chunk)]).cpu().numpy()   # (Vy,) Y->X vertex
        errs.append(calculate_geodesic_error(
            s0['dist'].cpu().numpy(), s0['corr'].cpu().numpy(),
            s1['corr'].cpu().numpy(), p2p, return_mean=False))
    cat = np.concatenate(errs)
    return float((cat <= thresh).mean()), float(cat.mean())


def gradient_check(ext, pair, tau):
    """One forward+backward; verify every param gets a finite grad and features aren't collapsed."""
    ext.train()
    ext.zero_grad(set_to_none=True)
    fx, fy = _extract(ext, pair['first']), _extract(ext, pair['second'])
    finite = bool(torch.isfinite(fx).all() and torch.isfinite(fy).all())
    var = float(fx.detach().var(dim=0).mean())                  # per-point feature spread
    loss, acc = contrastive_loss(fx, fy, tau)
    loss.backward()

    total = none_grad = zero_grad = nonfinite = 0
    worst = []
    for name, p in ext.named_parameters():
        if not p.requires_grad:
            continue
        total += 1
        g = p.grad
        if g is None:
            none_grad += 1; worst.append((name, 'None'))
        elif not torch.isfinite(g).all():
            nonfinite += 1; worst.append((name, 'non-finite'))
        elif g.abs().sum() == 0:
            zero_grad += 1; worst.append((name, 'zero'))

    ok = (none_grad == 0 and nonfinite == 0 and finite and var > 1e-8)
    print(f'[gradient]  loss={loss.item():.4f} acc={acc.item():.3f} | params={total} '
          f'none={none_grad} zero={zero_grad} non-finite={nonfinite} | '
          f'feat finite={finite} var={var:.2e}')
    if worst:
        print('   problem params:', ', '.join(f'{n} ({w})' for n, w in worst[:8]))
    print(f'   -> {"PASS" if ok else "FAIL"}')
    return ok


def overfit_check(ext, pairs, tau, steps, lr, pass_acc):
    """Train on a few FIXED cached pairs; train acc must climb toward ~1."""
    ext.train()
    optim = torch.optim.AdamW(ext.parameters(), lr=lr)
    accs = []
    for step in range(steps):
        pair = pairs[step % len(pairs)]
        fx, fy = _extract(ext, pair['first']), _extract(ext, pair['second'])
        loss, acc = contrastive_loss(fx, fy, tau)
        optim.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(ext.parameters(), 1.0)
        optim.step()
        accs.append(acc.item())
        if step in (0, steps // 4, steps // 2, 3 * steps // 4, steps - 1):
            print(f'[overfit]   step {step:4d}  loss={loss.item():.4f}  acc={acc.item():.3f}')
    final = sum(accs[-len(pairs):]) / len(pairs)                # avg over a last full cycle
    ok = final >= pass_acc
    print(f'   final train acc (last cycle) = {final:.3f}  (pass >= {pass_acc}) '
          f'-> {"PASS" if ok else "FAIL"}')
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', required=True, help='extractor pretrain config (yaml)')
    p.add_argument('--n_pairs', type=int, default=4, help='fixed pairs for the overfit check')
    p.add_argument('--steps', type=int, default=300, help='overfit optimisation steps')
    p.add_argument('--lr', type=float, default=1e-3, help='overfit learning rate')
    p.add_argument('--pass_acc', type=float, default=0.9, help='overfit pass threshold')
    p.add_argument('--tau', type=float, default=0.07, help='InfoNCE temperature')
    p.add_argument('--thresh', type=float, default=0.1, help='PCK geodesic-error threshold (dense readout)')
    p.add_argument('--input_type', default=None,
                   help="override network.input_type (e.g. 'hks' for a non-trivial overfit signal)")
    p.add_argument('--device', default=None, help="'cuda'/'cpu'; auto when omitted")
    args = p.parse_args()

    opt = load_yaml(args.config)
    if args.input_type is not None:
        opt['network']['input_type'] = args.input_type
    device = torch.device(args.device or opt.get('device')
                          or ('cuda' if torch.cuda.is_available() else 'cpu'))
    dataset = build_dataset(opt['datasets']['train'])

    # cache pairs ONCE so the sparse sampling is frozen (a real overfit target), then move to device
    def to_dev(x):
        if torch.is_tensor(x):
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return x
    pairs = [to_dev(dataset[i]) for i in range(min(args.n_pairs, len(dataset)))]

    ext = build_network(dict(opt['network'])).to(device)
    n_params = sum(q.numel() for q in ext.parameters())
    print(f"sanity: {ext.__class__.__name__} ({n_params:,} params) on {device} | "
          f"config {os.path.basename(args.config)} | {len(pairs)} overfit pairs\n")

    g_ok = gradient_check(ext, pairs[0], args.tau)

    # honest dense readout before/after overfit: bijective sparse acc saturates at ~1.0 and hides
    # symmetry flips, so also match per-vertex over the full mesh vs template GT (PCK / geo err).
    pck0, err0 = dense_accuracy(ext, pairs, args.thresh)
    print(f'\n[dense]     pre-overfit : PCK@{args.thresh}={pck0:.3f}  mean geo err={err0:.4f}')
    print()
    o_ok = overfit_check(ext, pairs, args.tau, args.steps, args.lr, args.pass_acc)
    pck1, err1 = dense_accuracy(ext, pairs, args.thresh)
    print(f'[dense]     post-overfit: PCK@{args.thresh}={pck1:.3f}  mean geo err={err1:.4f}  '
          f'(honest; sparse acc hides symmetry flips)')

    print(f'\n==> {"ALL PASS" if (g_ok and o_ok) else "FAILURES PRESENT"}')
    sys.exit(0 if (g_ok and o_ok) else 1)


if __name__ == '__main__':
    main()
