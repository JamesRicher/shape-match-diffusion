"""Precompute per-vertex WKS descriptors and save them as frozen feature .npy files.

Writes one (N, n_e) array per shape into <data_root>/<out_dir>/<name>.npy, matching the
filename ordering the datasets use (sort_list over off/). These become a fixed, non-learned
feature source for a no-extractor MatrixDiffusionModel: point the dataset at them with
`feats_dir: feats_wks` and the model loads them exactly like any other frozen features.

The descriptor is computed from the cached LBO spectrum (evals, evecs) the dataset already
loads under ret_evecs, so no re-meshing or eigen-solve happens here.

Example:
    python -m scripts.compute_spectral_feats --dataset Faust_r --phase full --n_e 100
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from datasets.shape_datasets import (SingleFaustDataset, SingleScapeDataset,
                                     SingleSmalDataset)
from paths import FAUST_DIR, SCAPE_DIR, SMAL_DIR
from utils.spectral_features import wks

_SINGLE = {
    'Faust_r': (SingleFaustDataset, FAUST_DIR),
    'Scape_r': (SingleScapeDataset, SCAPE_DIR),
    'Smal_r':  (SingleSmalDataset,  SMAL_DIR),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default='Faust_r', choices=list(_SINGLE))
    ap.add_argument('--phase', default='full', help="'full' writes every shape (recommended)")
    ap.add_argument('--n_e', type=int, default=100, help='number of WKS energy bands (feature dim)')
    ap.add_argument('--num_evecs', type=int, default=200, help='eigenpairs used for the descriptor')
    ap.add_argument('--out_dir', default='feats_wks', help='subdir of the data root to write into')
    args = ap.parse_args()

    cls, root = _SINGLE[args.dataset]
    # ret_feats False: we are *producing* features, so don't require an existing feats dir.
    ds = cls(root, phase=args.phase, ret_faces=False, ret_feats=False, ret_corr=False,
             ret_dist=False, ret_evecs=True, num_evecs=args.num_evecs)

    out_path = os.path.join(root, args.out_dir)
    os.makedirs(out_path, exist_ok=True)
    print(f'Writing {len(ds)} WKS descriptors (n_e={args.n_e}) -> {out_path}')

    for i in tqdm(range(len(ds)), desc='WKS'):
        item = ds[i]
        desc = wks(item['evals'], item['evecs'], n_e=args.n_e)      # (N, n_e)
        np.save(os.path.join(out_path, f"{item['name']}.npy"),
                desc.to(torch.float32).cpu().numpy())


if __name__ == '__main__':
    main()
