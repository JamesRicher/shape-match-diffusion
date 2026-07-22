"""Standalone contrastive pretraining of the GCN feature extractor (no diffusion).

Config-driven and self-contained under debug/feature_extractor/ -- kept separate from the
real configs/ and experiments/. A run reads a debug config (network + datasets + train opts)
and writes its checkpoints/logs to extractor_experiments/<kind>/<name>/.

Trains GCNFeatureExtractor on FAUST_r sparse pairs so matched points get aligned features:
the sparse GT is the identity permutation over FPS points (point i of X <-> point i of Y), so
we take a symmetric InfoNCE over the (n, n) feature similarity matrix with target arange(n).

    python pretrain_extractor.py                                            # default debug config
    python pretrain_extractor.py -c debug/feature_extractor/configs/faust_gcn.yaml
    python pretrain_extractor.py --name faust_gcn_anchor --epochs 80        # CLI overrides
"""
import os
import time
import argparse
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.options import load_yaml
from utils.metric_logger import MetricLogger
from datasets import build_dataset
from networks import build_network
from networks.gcn_feature_extractor import build_patches

# Configs still live under debug/feature_extractor/configs/; run OUTPUTS (checkpoints + logs)
# now live under extractor_experiments/<kind>/<name>/, grouped by extractor family.
# Anchored to this file's directory so paths hold regardless of the caller's cwd.
ROOT = os.path.dirname(os.path.abspath(__file__))
DEBUG_FE_ROOT = os.path.join(ROOT, 'debug', 'feature_extractor')
DEFAULT_CONFIG = os.path.join(DEBUG_FE_ROOT, 'configs', 'faust_gcn_4layer.yaml')
EXTRACTOR_EXPERIMENTS_ROOT = os.path.join(ROOT, 'extractor_experiments')

# extractor network type -> run-folder family (subfolder under extractor_experiments/)
EXTRACTOR_KIND = {'GCNFeatureExtractor': 'gcn', 'DiffusionNetExtractor': 'diffusion_net'}


def extractor_kind(opt):
    """Run-folder family for a config's extractor. Reads `network` (pretrain config) or
    `networks.extractor` (joint config)."""
    cfg = opt.get('network') or opt['networks']['extractor']
    t = cfg['type']
    if t not in EXTRACTOR_KIND:
        raise KeyError(f"unknown extractor type {t!r}; add it to EXTRACTOR_KIND")
    return EXTRACTOR_KIND[t]


def run_paths(name, kind):
    """extractor_experiments/<kind>/<name>/ with models/ (checkpoints) and results/ (logs).
    `kind` groups runs by extractor family ('gcn' | 'diffusion_net')."""
    run_dir = os.path.join(EXTRACTOR_EXPERIMENTS_ROOT, kind, name)
    models_dir = os.path.join(run_dir, 'models')
    results_dir = os.path.join(run_dir, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    return run_dir, models_dir, results_dir


def _patch_collate(batch, patch_size):
    """batch_size=1 collate that patchifies BOTH shapes in the worker and drops the full
    (N,N) geodesic. This moves the topk/gather off the main thread and shrinks worker->main
    IPC to the small per-patch tensors. Returns {'first': patches, 'second': patches}."""
    pair = batch[0]
    return {key: build_patches(pair[key]['verts'], pair[key]['dist'],
                               pair[key]['sparse']['idx'], patch_size)
            for key in ('first', 'second')}


def contrastive_loss(fx, fy, tau):
    """Symmetric InfoNCE. fx, fy: (n, d) per-point features; row i of X matches row i of Y."""
    fx, fy = F.normalize(fx, dim=-1), F.normalize(fy, dim=-1)
    logits = (fy @ fx.t()) / tau                 # (n, n): row j (Y) over columns (X)
    target = torch.arange(fx.shape[0], device=fx.device)
    loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
    acc = (logits.argmax(1) == target).float().mean()
    return loss, acc


# --- optional triplet loss (BendingGraphs-style; train.loss: triplet) ------------------
# Self-contained: to remove, delete this function and the `triplet` branch in main().
def triplet_loss(fx, fy, margin):
    """Symmetric hardest-negative triplet on L2-normalised features (BendingGraphs uses
    TripletMarginLoss p=2). Positive of Y point i is X point i; the negative is the *hardest*
    (nearest) non-matching point, mined in-set -- what makes triplet competitive with InfoNCE.
    Same (loss, acc) contract as contrastive_loss."""
    fx, fy = F.normalize(fx, dim=-1), F.normalize(fy, dim=-1)
    D = torch.cdist(fy, fx)                       # (n, n) Euclidean, row j (Y) vs col (X)
    n = fx.shape[0]
    target = torch.arange(n, device=fx.device)
    pos = D.diagonal()                            # matched-pair distance
    off = D.masked_fill(torch.eye(n, dtype=torch.bool, device=fx.device), float('inf'))
    neg_y = off.min(1).values                     # Y anchor -> hardest X negative
    neg_x = off.min(0).values                     # X anchor -> hardest Y negative
    loss = 0.5 * (F.relu(pos - neg_y + margin).mean() + F.relu(pos - neg_x + margin).mean())
    acc = (D.argmin(1) == target).float().mean()  # nearest-neighbour == matched (== cosine argmax)
    return loss, acc
# --- end optional triplet loss ---------------------------------------------------------


@torch.no_grad()
def evaluate(ext, dataset, loss_fn, limit=None):
    ext.eval()
    accs = []
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(n):
        s0, s1 = dataset[i]['first'], dataset[i]['second']
        fx = ext.extract(s0['verts'], s0['dist'], s0['sparse']['idx'])[0]
        fy = ext.extract(s1['verts'], s1['dist'], s1['sparse']['idx'])[0]
        _, acc = loss_fn(fx, fy)
        accs.append(acc.item())
    ext.train()
    return sum(accs) / len(accs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-c', '--config', default=DEFAULT_CONFIG, help='debug FE config (yaml)')
    p.add_argument('--name', default=None, help='override run name (run dir under runs/)')
    p.add_argument('--epochs', type=int, default=None, help='override train.epochs')
    p.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    p.add_argument('--num_workers', type=int, default=None, help='override train.num_workers')
    p.add_argument('--max_steps', type=int, default=None, help='cap train steps/epoch (smoke)')
    args = p.parse_args()

    opt = load_yaml(args.config)
    tcfg = opt.get('train', {})
    name = args.name or opt['name']
    epochs = args.epochs or tcfg.get('epochs', 50)
    lr = tcfg.get('lr', 1e-3)
    tau = tcfg.get('tau', 0.07)
    wd = tcfg.get('weight_decay', 1e-4)
    # loss selection (default InfoNCE). `triplet` branch is optional -> see triplet_loss.
    if tcfg.get('loss', 'contrastive') == 'triplet':
        margin = tcfg.get('margin', 1.0)
        loss_fn = lambda a, b: triplet_loss(a, b, margin)
    else:
        loss_fn = lambda a, b: contrastive_loss(a, b, tau)
    eval_limit = tcfg.get('eval_limit', 40)
    num_workers = args.num_workers if args.num_workers is not None else tcfg.get('num_workers', 8)
    device = torch.device(args.device or opt.get('device')
                          or ('cuda' if torch.cuda.is_available() else 'cpu'))

    run_dir, models_dir, results_dir = run_paths(name, extractor_kind(opt))
    best_ckpt = os.path.join(models_dir, 'best.pth')
    final_ckpt = os.path.join(models_dir, 'final.pth')
    tb_dir = os.path.join(run_dir, 'tb')

    train_set = build_dataset(opt['datasets']['train'])
    val_set = build_dataset(opt['datasets']['val'])

    ext = build_network(dict(opt['network'])).to(device)
    optim = torch.optim.AdamW(ext.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs, eta_min=lr * 0.1)

    # Prefetching loader: the per-shape geodesic matrices are large and the dataset's LRU
    # cache is small, so loading + the patch gather dominate. The collate patchifies both
    # shapes IN THE WORKER (build_patches) and drops the full (N,N) dist, so the main thread
    # only does the trivial GPU forward. persistent_workers keeps the OS page cache warm.
    collate = partial(_patch_collate, patch_size=ext.patch_size)
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, collate_fn=collate,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        prefetch_factor=(4 if num_workers > 0 else None), pin_memory=(device.type == 'cuda'))

    n_params = sum(q.numel() for q in ext.parameters())
    print(f"run '{name}' -> {run_dir}\nextractor {ext.__class__.__name__} ({n_params:,} params) "
          f"on {device}; {len(train_set)} train / {len(val_set)} val pairs")

    # CSV (results/metrics.csv) + TensorBoard (tb/) via the repo's MetricLogger.
    #   tensorboard --logdir extractor_experiments
    mlogger = MetricLogger(results_dir, tb_dir=tb_dir)

    total = args.max_steps or len(train_loader)
    gstep = 0                                        # global step for per-iteration TB curves
    best_val = -1.0
    for epoch in range(epochs):
        run_loss = run_acc = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, total=total, desc=f'epoch {epoch:3d}/{epochs - 1}', leave=False)
        step = 0
        for pair in pbar:
            if args.max_steps is not None and step >= args.max_steps:
                break
            fx = ext(pair['first'])[0]              # pair['first'] = worker-built patches
            fy = ext(pair['second'])[0]
            loss, acc = loss_fn(fx, fy)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ext.parameters(), 1.0)
            optim.step()
            run_loss += loss.item(); run_acc += acc.item()
            step += 1; gstep += 1
            pbar.set_postfix(loss=f'{run_loss/step:.3f}', acc=f'{run_acc/step:.3f}',
                             lr=f'{sched.get_last_lr()[0]:.1e}')
            mlogger.log_many({'Loss/train': loss.item(), 'Acc/train': acc.item()}, gstep, epoch)
        sched.step()

        lr_now = sched.get_last_lr()[0]
        val_acc = evaluate(ext, val_set, loss_fn, eval_limit)
        best = val_acc > best_val
        best_val = max(best_val, val_acc)
        secs = time.time() - t0
        print(f'epoch {epoch:3d} | train loss {run_loss/step:.4f} acc {run_acc/step:.3f} '
              f'| val acc {val_acc:.3f} (best {best_val:.3f}) | lr {lr_now:.2e} | {secs:.1f}s'
              + ('  <- new best, saved' if best else ''))
        mlogger.log_many({'Val/acc': val_acc, 'Loss/train_epoch': run_loss / step,
                          'Acc/train_epoch': run_acc / step, 'LR': lr_now}, gstep, epoch)
        if best:                                    # keep the best-val checkpoint, not just last
            torch.save({'networks': {'extractor': ext.state_dict()}}, best_ckpt)

    torch.save({'networks': {'extractor': ext.state_dict()}}, final_ckpt)
    mlogger.close()
    print(f'done. best val acc {best_val:.3f}.\n  best  -> {best_ckpt}\n  final -> {final_ckpt}'
          f'\n  logs  -> {results_dir}/metrics.csv  +  tensorboard --logdir {run_dir}/tb')


if __name__ == '__main__':
    main()
