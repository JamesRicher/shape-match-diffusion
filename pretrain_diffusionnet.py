"""Standalone contrastive pretraining of the DiffusionNet feature extractor (no diffusion).

The DiffusionNet analogue of pretrain_extractor.py: same objective, logging and checkpoint
layout, but the extractor runs on the WHOLE mesh from cached spectral operators (evecs / evals /
mass / gradX / gradY) instead of local patches, so the collate ships those operators + the FPS
indices rather than patchifying. The sparse GT is the identity permutation over FPS points
(point i of X <-> point i of Y), so matched points are pulled together by a symmetric InfoNCE
(or triplet) over the (n, n) feature-similarity matrix -- reused verbatim from pretrain_extractor.

Config needs the datasets to set ret_evecs: true (num_evecs >= network.k_eig).

    python pretrain_diffusionnet.py -c debug/feature_extractor/configs/faust_diffusionnet.yaml
    python pretrain_diffusionnet.py --name faust_dfn_hks --epochs 80        # CLI overrides

Outputs land under extractor_experiments/<kind>/<name>/ (models/best.pth, final.pth; logs).
Warm-start the joint model from best.pth via path.resume_state in the joint config.
"""
import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.options import load_yaml
from utils.metric_logger import MetricLogger
from datasets import build_dataset
from networks import build_network
# reuse the identical loss/eval-metric contract and run-dir layout from the GCN pretrainer
from pretrain_extractor import contrastive_loss, triplet_loss, run_paths, extractor_kind

# Dense operator fields DiffusionNetExtractor.extract() reads off a shape dict; the big (N,N)
# geodesic is dropped in the collate to keep worker->main IPC small.
_DENSE_KEYS = ('verts', 'evecs', 'evals', 'mass')
_SPARSE_KEYS = ('gradX', 'gradY')

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(ROOT, 'debug', 'feature_extractor', 'configs', 'faust_diffusionnet.yaml')


def _op_collate(batch):
    """batch_size=1 collate: keep each shape's spectral operators + FPS idx, drop the rest.

    Torch sparse tensors don't survive DataLoader worker->main IPC on many builds (the reason
    num_workers>0 errored), so gradX/gradY are shipped as plain (indices, values, size) tuples
    and rebuilt with _restore_sparse in the main thread. Returns {'first': {..}, 'second': {..}}."""
    pair = batch[0]
    out = {}
    for key in ('first', 'second'):
        s = pair[key]
        d = {k: s[k] for k in _DENSE_KEYS if k in s}
        for k in _SPARSE_KEYS:                                    # IPC-safe: dense (idx, val, size)
            g = s[k].coalesce()
            d[k] = (g.indices(), g.values(), tuple(g.size()))
        d['idx'] = s['sparse']['idx']
        out[key] = d
    return out


def _restore_sparse(shape):
    """Rebuild gradX/gradY sparse tensors from the collate's IPC-safe tuples, in place."""
    for k in _SPARSE_KEYS:
        idx, val, size = shape[k]
        shape[k] = torch.sparse_coo_tensor(idx, val, torch.Size(size)).coalesce()
    return shape


@torch.no_grad()
def evaluate(ext, dataset, loss_fn, limit=None):
    ext.eval()
    accs = []
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(n):
        s0, s1 = dataset[i]['first'], dataset[i]['second']
        fx = ext.extract(s0, s0['sparse']['idx'])[0]
        fy = ext.extract(s1, s1['sparse']['idx'])[0]
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
    # loss selection (default InfoNCE). `triplet` branch mirrors the GCN pretrainer.
    if tcfg.get('loss', 'contrastive') == 'triplet':
        margin = tcfg.get('margin', 1.0)
        loss_fn = lambda a, b: triplet_loss(a, b, margin)
    else:
        loss_fn = lambda a, b: contrastive_loss(a, b, tau)
    eval_limit = tcfg.get('eval_limit', 40)
    num_workers = args.num_workers if args.num_workers is not None else tcfg.get('num_workers', 4)
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

    # gradX/gradY ship as IPC-safe tuples (see _op_collate) so num_workers>0 works and prefetch
    # hides the per-shape geodesic load; pin_memory off (rebuilt sparse tensors aren't pinnable).
    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, collate_fn=_op_collate,
        num_workers=num_workers, persistent_workers=num_workers > 0,
        prefetch_factor=(4 if num_workers > 0 else None), pin_memory=False)

    n_params = sum(q.numel() for q in ext.parameters())
    print(f"run '{name}' -> {run_dir}\nextractor {ext.__class__.__name__} ({n_params:,} params) "
          f"on {device}; {len(train_set)} train / {len(val_set)} val pairs")

    # CSV (results/metrics.csv) + TensorBoard (tb/):  tensorboard --logdir extractor_experiments
    mlogger = MetricLogger(results_dir, tb_dir=tb_dir)

    total = args.max_steps or len(train_loader)
    gstep = 0
    best_val = -1.0
    for epoch in range(epochs):
        run_loss = run_acc = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, total=total, desc=f'epoch {epoch:3d}/{epochs - 1}', leave=False)
        step = 0
        for pair in pbar:
            if args.max_steps is not None and step >= args.max_steps:
                break
            _restore_sparse(pair['first']); _restore_sparse(pair['second'])   # rebuild gradX/gradY
            fx = ext.extract(pair['first'], pair['first']['idx'])[0]   # full-mesh DiffusionNet -> FPS pts
            fy = ext.extract(pair['second'], pair['second']['idx'])[0]
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
        if best:
            torch.save({'networks': {'extractor': ext.state_dict()}}, best_ckpt)

    torch.save({'networks': {'extractor': ext.state_dict()}}, final_ckpt)
    mlogger.close()
    print(f'done. best val acc {best_val:.3f}.\n  best  -> {best_ckpt}\n  final -> {final_ckpt}'
          f'\n  logs  -> {results_dir}/metrics.csv  +  tensorboard --logdir {run_dir}/tb')


if __name__ == '__main__':
    main()
