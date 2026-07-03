import os
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Sequence, Tuple, Optional
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import warnings

from utils.data_utils import *

FAUST_TRAIN_NAMES = [f"tr_reg_{i:03d}" for i in range(0, 80)]
FAUST_VAL_NAMES = [f"tr_reg_{i:03d}" for i in range(80, 100)]

@dataclass
class FaustDatasetConfig:
    n_points: int = 64
    epoch_size: int = 1024
    train_shape_names: Sequence[str] = field(default_factory=lambda: FAUST_TRAIN_NAMES)
    test_shape_names: Sequence[str] = field(default_factory=lambda: FAUST_VAL_NAMES)
    overfit_pair: Optional[Tuple[str, str]] = None
    overfit_lock: bool = False
    dataset_dir: str = DEFAULT_DATASET_DIR
    seed: Optional[int] = None
    feature_noise_std: float = 0.0


class _ShapeCache:
    """Lazy loader for shape data, a copy of this class is owned by a Dataset object"""
    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self._feats: dict = {}
        self._corres: dict = {}
        self._verts: dict = {}
        self._faces: dict = {}
        self._dists: dict = {}

    def feats(self, name: str) -> np.ndarray:
        if name not in self._feats:
            self._feats[name] = load_feats(name, self.dataset_dir).astype(np.float32)
        return self._feats[name]

    def geom(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        if name not in self._verts:
            v, f = load_off(name, self.dataset_dir)
            self._verts[name] = v.astype(np.float32)
            self._faces[name] = f.astype(np.int64)
        return self._verts[name], self._faces[name]

    def corres(self, name: str) -> np.ndarray:
        if name not in self._corres:
            self._corres[name] = load_corres(name, self.dataset_dir)
        return self._corres[name]


class FaustPairDataset(Dataset):
    def __init__(self, config: FaustDatasetConfig):
        self.config = config
        self.cache = _ShapeCache(config.dataset_dir)

        if config.overfit_pair is not None:
            a, b = config.overfit_pair
            assert a != b
            self.pool = [(a,b)]
        else:
            assert len(config.test_shape_names) >= 2
            self.pool = None

        if config.overfit_lock:
            assert config.overfit_pair is not None
            rng = np.random.default_rng(0 if config.seed is None else config.seed)
            self._locked = self._draw_with(rng, *self.pool[0])
        else:
            self._locked = None
    
    def __len__(self):
        # use over the total pair count as we dont want to have to do all pairs per run
        return self.config.epoch_size 

    def __getitem__(self, idx: int):
        """returns C, Pi and gt for a randomly drawn pair of shapes"""
        if self._locked is not None:
            C, Pi, gt = self._locked
            return (
                torch.from_numpy(C.copy()),
                torch.from_numpy(Pi.copy()),
                torch.from_numpy(gt.copy()),
            )
        
        rng = self._rng_for(idx)
        if self.pool is not None:
            a, b = self.pool[0]
        else:
            a, b = self._random_pair(rng)
        C, Pi, gt = self._draw_with(rng, a, b)
        return (
            torch.from_numpy(C.copy()),
            torch.from_numpy(Pi.copy()),
            torch.from_numpy(gt.copy()),
        )


    # HELPERS
    def _rng_for(self, idx: int) -> np.random.Generator:
        base = 0 if self.config.seed is None else int(self.config.seed)
        wi = torch.utils.data.get_worker_info()
        worker = 0 if wi is None else wi.id
        return np.random.default_rng((base * 1_000_003) ^ (worker * 9176) ^ idx)
    

    def _random_pair(self, rng: np.random.Generator) -> Tuple[str, str]:
        names = list(self.config.train_shape_names)
        i, j = rng.choice(len(names), size=2, replace=False)
        return names[i], names[j]

    
    def _draw_with(self, rng: np.random.Generator, shape_a: str, shape_b: str
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = self.config.n_points

        # load up all the full mesh data
        feats_a = self.cache.feats(shape_a)
        feats_b = self.cache.feats(shape_b)
        corres_a = self.cache.corres(shape_a)
        corres_b = self.cache.corres(shape_b)
        verts_a, _ = self.cache.geom(shape_a)

        # perform FPS
        pool = verts_a[corres_a] # the vertices we are choosing from
        n_eff = min(n, pool.shape[0])
        start = int(rng.integers(0, pool.shape[0]))
        template = fps(pool, n_eff, start=start)

        a_idx = corres_a[template]
        b_idx = corres_b[template]
        a_idx = np.clip(a_idx, 0, feats_a.shape[0] - 1)
        b_idx = np.clip(b_idx, 0, feats_b.shape[0] - 1)

        perm = rng.permutation(n_eff)
        b_idx_perm = b_idx[perm]

        feat1 = feats_a[a_idx]
        feat2 = feats_b[b_idx_perm]


        # make the features more noisy if we need to
        if self.config.feature_noise_std > 0:
            feat1 = feat1 + rng.normal(
                0.0, self.config.feature_noise_std, size=feat1.shape
            ).astype(np.float32)
            feat2 = feat2 + rng.normal(
                0.0, self.config.feature_noise_std, size=feat2.shape
            ).astype(np.float32)

        feat1 = l2_normalize_rows(feat1)
        feat2 = l2_normalize_rows(feat2)

        C = (feat1 @ feat2.T).astype(np.float32)
        gt = np.argsort(perm).astype(np.int64)
        Pi = permutation_matrix(gt)

        return C, Pi, gt


    # Vis helpers
    def get_single_shape(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.cache.geom(name)

    
    def get_single_shape(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        assert idx >= 0 and idx < 100
        name = f"tr_reg_{idx:03d}"
        return self.cache.geom(name)


# Debugging
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    config = FaustDatasetConfig()
    ds = FaustPairDataset(config)
    ds.__getitem__(0)

    loader = DataLoader(ds, batch_size=1, shuffle=True)
    result = next(iter(loader))
    print(result)
    tensor = torch.rand(3, 256, 256)
    transform = T.ToPILImage()
    img = transform(result[1])
    plt.imshow(img, cmap="gray")
    plt.show()