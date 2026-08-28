"""Localise the nobp loss blow-up: per-stage magnitudes over the first N train batches.

  python -m diagnostics.diag_blowup -c configs/penultimate/dt4d_mpnn_512_ds_geo_nobp.yaml -n 30

Prints, per step: the gaussian/DS target, the clean logit u0, the read-in P_t, the denoiser
output u0_hat, and the loss. The first column whose magnitude is absurd is the culprit.
"""
import argparse

import torch

from datasets import build_dataset
from models import build_model
from train import autofill_feat_dim, _single_collate
from utils.options import load_yaml, resolve_experiment_paths


def rng(t):
    if t is None:
        return "        none        "
    t = t.detach().float()
    bad = "!" if not torch.isfinite(t).all() else " "
    return f"[{t.min():>9.2e},{t.max():>9.2e}]{bad}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("-n", "--steps", type=int, default=30)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    opt = load_yaml(a.config)
    opt["is_train"] = True
    if a.device:
        opt["device"] = a.device
    resolve_experiment_paths(opt)
    opt["path"]["resume"] = False

    train_cfg = opt["datasets"]["train"]
    train_cfg = train_cfg if isinstance(train_cfg, list) else [train_cfg]
    sets = [build_dataset(c) for c in train_cfg]
    ds = torch.utils.data.ConcatDataset(sets) if len(sets) > 1 else sets[0]
    autofill_feat_dim(opt, sets[0])
    model = build_model(opt)

    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True,
                                         collate_fn=_single_collate, num_workers=0)

    print(f"{'it':>4} {'target':>24} {'u0':>24} {'P_t':>24} {'u0_hat':>24} {'loss':>12}")
    it = 0
    for data in loader:
        if it >= a.steps:
            break
        F_x, F_y, D_x, D_y, P0 = model._sparse_inputs(data)
        D_cross = data.get("gt_cross_dist")
        if D_cross is not None:
            D_cross = (D_cross.unsqueeze(0) if D_cross.dim() == 2 else D_cross)
            D_cross = D_cross.to(model.device).float()
        P_ce, u0, H = model._target(P0, D_x, D_cross)
        t = model._sample_t(u0.shape[0], False)
        u_t = None
        from utils.sinkhorn import q_sample
        u_t = q_sample(u0, t)
        P_t = model._read_in(u_t)
        with torch.no_grad():
            u0_hat = model.networks["denoiser"](P_t, F_x, F_y, D_x, D_y, t)
        model.feed_data(data)
        loss = model.loss_metrics["l_ce"]
        print(f"{it:>4} {rng(P_ce)} {rng(u0)} {rng(P_t)} {rng(u0_hat)} {loss.item():>12.3e}")
        it += 1


if __name__ == "__main__":
    main()
