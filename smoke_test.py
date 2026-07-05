"""Fast end-to-end smoke test of the training stack.

Exercises every moving part on a tiny slice of real data so you can catch wiring /
path / data problems in ~a minute before launching the full run:

    python smoke_test.py                       # uses configs/faust_shape_matching.yaml
    python smoke_test.py -c <cfg> --device cuda --iters 3 --pairs 3

Exits non-zero if any stage fails.
"""
import argparse
import itertools
import os
import os.path as osp
import shutil
import tempfile
import time
import types

import matplotlib
matplotlib.use('Agg')  # headless-safe (remote/SSH)

import torch

import train  # reuse the real build_opt / build_dataloaders / move_to_device
from datasets import build_dataset
from models import build_model
from utils.registry import (DATASET_REGISTRY, NETWORK_REGISTRY, LOSS_REGISTRY,
                            METRIC_REGISTRY, MODEL_REGISTRY)


# --------------------------------------------------------------------------- #
# tiny check harness
# --------------------------------------------------------------------------- #
_RESULTS = []


def check(name, fn):
    """Run fn(); record PASS/FAIL. fn may return a detail string."""
    t0 = time.time()
    try:
        detail = fn() or ''
        ok = True
    except Exception as e:  # noqa: BLE001 - smoke test wants to keep going
        detail = f'{type(e).__name__}: {e}'
        ok = False
    dt = time.time() - t0
    _RESULTS.append(ok)
    print(f'[{"PASS" if ok else "FAIL"}] {name:38s} ({dt:5.1f}s)' + (f'  {detail}' if detail else ''))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-c', '--config', default='configs/faust_shape_matching.yaml')
    p.add_argument('--device', default=None)
    p.add_argument('--iters', type=int, default=3, help='training iterations to run')
    p.add_argument('--pairs', type=int, default=3, help='validation pairs to run')
    return p.parse_args()


def main():
    args = parse_args()
    tmp = tempfile.mkdtemp(prefix='smoke_')
    print(f'scratch dir: {tmp}\n')

    # shared state populated across stages
    state = types.SimpleNamespace(opt=None, train_loader=None, val_loader=None, model=None)

    # --- 1. registries populated ------------------------------------------- #
    def _registries():
        for reg, expect in [(DATASET_REGISTRY, 'PairFaustDataset'),
                            (NETWORK_REGISTRY, 'ShapeMatchingEncoder'),
                            (LOSS_REGISTRY, 'SupervisedContrastiveLoss'),
                            (METRIC_REGISTRY, 'calculate_geodesic_error'),
                            (MODEL_REGISTRY, 'ShapeMatchingModel')]:
            assert expect in reg, f'{expect} missing from {reg._name} registry'
        return 'datasets/networks/losses/metrics/models all registered'
    check('registries', _registries)

    # --- 2. config load + T_max coupling ----------------------------------- #
    def _config():
        cli = types.SimpleNamespace(config=args.config, name=None, epochs=None,
                                    device=args.device, resume=None)
        opt = train.build_opt(cli)
        # route outputs to scratch
        opt['path']['models'] = osp.join(tmp, 'models')
        opt['path']['results'] = osp.join(tmp, 'results')
        sched = opt['train']['schedulers']['encoder']
        assert sched.get('T_max') == opt['train']['total_epochs'], 'T_max != total_epochs'
        state.opt = opt
        return f"model={opt['model_type']} epochs={opt['train']['total_epochs']} T_max={sched['T_max']}"
    check('config + T_max coupling', _config)

    # --- 3. paths / data present ------------------------------------------- #
    def _paths():
        import paths
        assert osp.isdir(paths.DATA_ROOT), f'DATA_ROOT missing: {paths.DATA_ROOT}'
        root = paths.DEFAULT_DATA_ROOTS['Faust_r']
        assert osp.isdir(root), f'FAUST_r missing: {root}'
        assert osp.isdir(osp.join(root, 'diffusion')), 'diffusion cache missing (ret_evecs needs it)'
        assert osp.isdir(osp.join(root, 'feats')), 'feats missing (model needs data[feat])'
        return f'DATA_ROOT={paths.DATA_ROOT}'
    check('paths + data present', _paths)

    # --- 4. datasets build + item schema ----------------------------------- #
    def _datasets():
        train_set, train_loader, val_loader = train.build_dataloaders(state.opt, num_workers=0)
        assert len(train_set) > 0
        item = train_set[0]['first']
        for k in ('feat', 'corr', 'dist', 'verts', 'L'):
            assert k in item, f'missing key {k} in dataset item'
        state.train_loader, state.val_loader = train_loader, val_loader
        state.opt['networks']['encoder']['in_dim'] = int(item['feat'].shape[-1])
        return f'{len(train_set)} train pairs, in_dim={state.opt["networks"]["encoder"]["in_dim"]}'
    check('datasets + item schema', _datasets)

    # --- 5. model build (train mode): nets/optims/scheds/losses ------------ #
    def _build():
        m = build_model(state.opt)
        assert 'encoder' in m.networks and 'encoder' in m.optimizers and 'encoder' in m.schedulers
        assert len(m.losses) >= 1
        state.model = m
        n = sum(p.numel() for p in m.networks['encoder'].parameters())
        return f'optims={list(m.optimizers)} scheds={[type(s).__name__ for s in m.schedulers.values()]} losses={list(m.losses)} params={n:,}'
    check('model build + wiring', _build)

    # --- 6. training iterations (feed_data -> optimize_parameters) --------- #
    def _train_step():
        m = state.model
        losses = []
        for i, data in enumerate(state.train_loader):
            m.curr_iter += 1
            data = train.move_to_device(data, m.device)
            m.feed_data(data)
            m.optimize_parameters()
            m.update_model_per_iteration()
            losses.append(m.loss_metrics['l_total'].item())
            if i + 1 >= args.iters:
                break
        assert all(torch.isfinite(torch.tensor(l)) for l in losses), 'non-finite loss'
        # grads actually flowed
        g = [p.grad for p in m.networks['encoder'].parameters() if p.grad is not None]
        assert g, 'no gradients on encoder'
        return f'l_total: {[round(l, 3) for l in losses]}  (grads on {len(g)} tensors)'
    check('training step + grad flow', _train_step)

    # --- 7. scheduler advances the lr -------------------------------------- #
    def _scheduler():
        m = state.model
        lr0 = m.get_current_learning_rate()[0]
        m.update_model_per_epoch()
        lr1 = m.get_current_learning_rate()[0]
        assert lr1 != lr0, 'cosine lr did not change on epoch step'
        return f'lr {lr0:.2e} -> {lr1:.2e}'
    check('scheduler steps lr', _scheduler)

    # --- 8. validation + PCK artifacts ------------------------------------- #
    def _validation():
        m = state.model
        val_slice = list(itertools.islice(state.val_loader, args.pairs))
        res = m.validation(val_slice, update=True)
        assert 'avg_error' in res and 'auc' in res
        results = state.opt['path']['results']
        assert osp.isfile(osp.join(results, 'pck.png')) and osp.isfile(osp.join(results, 'pck.npy'))
        return f"avg_error={res['avg_error']:.4f} auc={res['auc']:.4f}, wrote pck.png/pck.npy"
    check('validation + PCK output', _validation)

    # --- 9. save + resume round-trip --------------------------------------- #
    def _save_resume():
        m = state.model
        m.save_model()                              # <iter>.pth (full)
        m.save_model(net_only=True, best=True)      # final.pth (best weights)
        models_dir = state.opt['path']['models']
        assert osp.isfile(osp.join(models_dir, 'final.pth'))

        # rebuild in inference mode from final.pth and confirm weights match
        opt2 = train.build_opt(types.SimpleNamespace(
            config=args.config, name=None, epochs=None, device=args.device,
            resume=osp.join(models_dir, 'final.pth')))
        opt2['is_train'] = False
        opt2['networks']['encoder']['in_dim'] = state.opt['networks']['encoder']['in_dim']
        opt2['path']['models'] = models_dir
        m2 = build_model(opt2)
        k = next(iter(m.networks['encoder'].state_dict()))
        w1 = m.networks['encoder'].state_dict()[k].cpu()
        w2 = m2.networks['encoder'].state_dict()[k].cpu()
        assert torch.equal(w1, w2), 'resumed weights differ from saved'
        return f'saved final.pth + full ckpt; inference reload matches ({k})'
    check('save + resume (inference)', _save_resume)

    # --- summary ----------------------------------------------------------- #
    shutil.rmtree(tmp, ignore_errors=True)
    passed, total = sum(_RESULTS), len(_RESULTS)
    print(f'\n{"="*60}\n{passed}/{total} stages passed')
    if passed != total:
        print('SMOKE TEST FAILED — fix the above before the full run.')
        raise SystemExit(1)
    print('SMOKE TEST PASSED — safe to launch the full pipeline.')


if __name__ == '__main__':
    main()
