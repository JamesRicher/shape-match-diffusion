import os
import numpy as np
from typing import Tuple
import scipy.io 

# DATA PATHS
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = "FAUST_r"
DEFAULT_DATASET_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "data", DEFAULT_DATASET)
)

def _dataset_paths(dataset_dir: str = DEFAULT_DATASET_DIR):
    return {
        "off": os.path.join(dataset_dir, "off"),
        "corres": os.path.join(dataset_dir, "corres"),
        "feats": os.path.join(dataset_dir, "feats"),
        "dist": os.path.join(dataset_dir, "dist")
    }

# LOADER HELPERS
def load_feats(feat_file: str) -> np.ndarray:
    assert os.path.isfile(feat_file), f"Invalid .npy feature file: {feat_file}"
    return np.load(feat_file)


def load_corres(corr_file: str) -> np.ndarray:
    assert os.path.isfile(corr_file), f"Invalid .vts file: {corr_file}"
    with open(corr_file) as f:
        return np.array([int(l.strip())-1 for l in f if l.strip()], dtype=np.int64)


def load_off(off_file: str) -> Tuple[np.ndarray, np.ndarray]:
    assert os.path.isfile(off_file), f"Invalid .off file: {off_file}"
    with open(off_file) as f:
        header = f.readline().strip()
        assert header == "OFF"
        n_verts, n_faces, _ = map(int, f.readline().split())
        verts = [list(map(float, f.readline().split())) for _ in range(n_verts)]
        faces = [list(map(int, f.readline().split()))[1:] for _ in range(n_faces)]
    return np.array(verts), np.array(faces)


def load_dist(dist_file: str) -> np.ndarray:
    assert os.path.isfile(dist_file), f"Invalid .mat file: {dist_file}"
    return scipy.io.loadmat(dist_file)["dist"]


def fps(pool, n: int, start: int):
    N = pool.shape[0]
    indices = np.empty(n, dtype=np.int64)
    indices[0] = start
    distances = np.full(N, np.inf)
    for i in range(1, n):
        d = np.sum((pool - pool[indices[i-1]])**2, axis=1)
        distances = np.minimum(d, distances)
        indices[i] = np.argmax(distances)
    return indices


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def permutation_matrix(perm: np.ndarray) -> np.ndarray:
    N = len(perm)
    P = np.zeros((N,N), dtype=np.float32)
    P[np.arange(N), perm] = 1.0
    return P