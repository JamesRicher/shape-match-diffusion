"""Standalone contrastive pretraining of the GCN feature extractor (no diffusion).

Trains GCNFeatureExtractor on FAUST_r sparse pairs so that matched points get aligned
features: the sparse GT is the identity permutation over FPS points (sparse point i of X
<-> sparse point i of Y), so we take a symmetric InfoNCE over the (n, n) feature similarity
matrix with target arange(n). Saves the extractor weights for use as a warm start in the
matrix-diffusion config.

    python pretrain_extractor.py                         # defaults below
    python pretrain_extractor.py --epochs 50 --node_in anchor --out extractor_faust.pth
"""
import argparse

import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.options import load_yaml
from datasets import build_dataset
from networks import build_network


def contrastive_loss(fx, fy, tau):
    """Symmetric InfoNCE. fx, fy: (n, d) per-point features; row i of X matches row i of Y."""
    fx, fy = F.normalize(fx, dim=-1), F.normalize(fy, dim=-1)
    logits = (fy @ fx.t()) / tau                 # (n, n): row j (Y) over columns (X)
    target = torch.arange(fx.shape[0], device=fx.device)
    loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
    acc = (logits.argmax(1) == target).float().mean()
    return loss, acc


def _features(ext, shape):
    """Run the patch extractor on one shape: full verts/dist + FPS idx -> (n, out_dim)."""
    return ext(shape['verts'], shape['dist'], shape['sparse']['idx'])[0]


@torch.no_grad()
def evaluate(ext, dataset, device, tau, limit=None):
    ext.eval()
    accs = []
    for i in range(len(dataset) if limit is None else min(limit, len(dataset))):
        pair = dataset[i]
        fx = _features(ext, pair['first'])
        fy = _features(ext, pair['second'])
        _, acc = contrastive_loss(fx, fy, tau)
        accs.append(acc.item())
    ext.train()
    return sum(accs) / len(accs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default='configs/faust_matrix_diffusion_gcn.yaml',
                   help='config to borrow dataset roots + extractor kwargs from')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--tau', type=float, default=0.07, help='InfoNCE temperature')
    p.add_argument('--node_in', default=None, help="override extractor node_in (xyz|anchor)")
    p.add_argument('--device', default=None)
    p.add_argument('--eval_limit', type=int, default=40, help='val pairs per eval')
    p.add_argument('--max_steps', type=int, default=None, help='cap train steps/epoch (smoke)')
    p.add_argument('--out', default='extractor_faust.pth')
    args = p.parse_args()

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    opt = load_yaml(args.config)

    train_set = build_dataset(opt['datasets']['train'])
    val_set = build_dataset(opt['datasets']['val'])

    ext_cfg = dict(opt['networks']['extractor'])
    if args.node_in is not None:
        ext_cfg['node_in'] = args.node_in
    ext = build_network(ext_cfg).to(device)
    optim = torch.optim.AdamW(ext.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=args.lr * 0.1)

    n_params = sum(q.numel() for q in ext.parameters())
    print(f'extractor {ext.__class__.__name__} ({n_params:,} params) on {device}; '
          f'{len(train_set)} train / {len(val_set)} val pairs')

    best_val = -1.0
    for epoch in range(args.epochs):
        order = torch.randperm(len(train_set)).tolist()
        if args.max_steps is not None:
            order = order[:args.max_steps]
        run_loss = run_acc = 0.0
        t0 = time.time()
        # live progress bar: running loss/acc + it/s + ETA for the epoch (tqdm), like the
        # rest of the repo's loops. postfix updates every step; leave=False keeps the log tidy.
        pbar = tqdm(order, desc=f'epoch {epoch:3d}/{args.epochs - 1}', leave=False)
        for step, idx in enumerate(pbar):
            pair = train_set[idx]
            fx = _features(ext, pair['first'])
            fy = _features(ext, pair['second'])
            loss, acc = contrastive_loss(fx, fy, args.tau)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ext.parameters(), 1.0)
            optim.step()
            run_loss += loss.item(); run_acc += acc.item()
            pbar.set_postfix(loss=f'{run_loss/(step+1):.3f}', acc=f'{run_acc/(step+1):.3f}',
                             lr=f'{sched.get_last_lr()[0]:.1e}')
        sched.step()

        n = len(order)
        val_acc = evaluate(ext, val_set, device, args.tau, args.eval_limit)
        best = val_acc > best_val
        best_val = max(best_val, val_acc)
        print(f'epoch {epoch:3d} | train loss {run_loss/n:.4f} acc {run_acc/n:.3f} '
              f'| val acc {val_acc:.3f} (best {best_val:.3f}) | lr {sched.get_last_lr()[0]:.2e} '
              f'| {time.time() - t0:.1f}s' + ('  <- new best, saved' if best else ''))
        if best:                                    # keep the best-val checkpoint, not just last
            torch.save({'networks': {'extractor': ext.state_dict()}}, args.out)

    print(f'done. best val acc {best_val:.3f}. saved -> {args.out}  '
          f'(load into a run via path.resume_state; the key is "extractor")')


if __name__ == '__main__':
    main()
