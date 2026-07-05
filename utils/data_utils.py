import os
import hashlib
import numpy as np
from typing import Tuple
import scipy.io
import scipy.sparse

from paths import FAUST_DIR as DEFAULT_DATASET_DIR

DEFAULT_DATASET = "FAUST_r"

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


# GEOMETRY HELPERS
def surface_area(mass):
    """Total surface area of a mesh, read off the lumped-mass (per-vertex area) vector.

    The DiffusionNet-style spectral operators expose ``mass`` as the diagonal of the
    lumped mass matrix, whose entries sum to the mesh area. Accepts a torch tensor or
    a numpy array and returns the same type.
    """
    return mass.sum()


def sqrt_surface_area(mass):
    """sqrt of the total surface area; the scale used to area-normalize geodesic
    distances. See :func:`surface_area`."""
    return surface_area(mass) ** 0.5


def _hash_arrays(*arrs) -> str:
    h = hashlib.sha1()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def load_diffusion_operators(verts_np: np.ndarray, faces_np: np.ndarray, cache_dir: str):
    """Loads DiffusionNet-style cached spectral operators for a shape, matching by
    SHA1 hash over (verts, faces) with linear probing on collisions.

    Hashing convention matches ULRSSM's `utils.geometry_util.get_operators`: SHA1
    over float32 verts bytes concatenated with int64 faces bytes.

    Returns dict with keys L, mass, evals, evecs, gradX, gradY, k_eig where L,
    gradX, gradY are scipy.sparse.csc_matrix and the rest are np.ndarray."""
    assert os.path.isdir(cache_dir), f"Missing diffusion cache dir: {cache_dir}"
    verts_np = verts_np.astype(np.float32, copy=False)
    faces_np = faces_np.astype(np.int64, copy=False)
    key = _hash_arrays(verts_np, faces_np)

    def _read_sp(npz, prefix):
        return scipy.sparse.csc_matrix(
            (npz[f"{prefix}_data"], npz[f"{prefix}_indices"], npz[f"{prefix}_indptr"]),
            shape=tuple(npz[f"{prefix}_shape"]),
        )

    i = 0
    while True:
        path = os.path.join(cache_dir, f"{key}_{i}.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No cached operators for hash {key} in {cache_dir}")
        npz = np.load(path, allow_pickle=True)
        if np.array_equal(verts_np, npz["verts"]) and np.array_equal(faces_np, npz["faces"]):
            return {
                "L": _read_sp(npz, "L"),
                "mass": npz["mass"],
                "evals": npz["evals"],
                "evecs": npz["evecs"],
                "gradX": _read_sp(npz, "gradX"),
                "gradY": _read_sp(npz, "gradY"),
                "k_eig": int(npz["k_eig"].item()),
            }
        i += 1


def sparse_np_to_torch(A: scipy.sparse.spmatrix) -> "torch.Tensor":
    """scipy sparse -> torch sparse_coo tensor (float32). Matches ULRSSM helper."""
    import torch
    Acoo = A.tocoo()
    indices = np.vstack((Acoo.row, Acoo.col))
    # Explicitly enable sparse invariant checks: validates the (trusted, cached)
    # operator and silences torch's "implicitly disabled" warning. Cheap here.
    with torch.sparse.check_sparse_tensor_invariants(enable=True):
        return torch.sparse_coo_tensor(
            torch.from_numpy(indices).long(),
            torch.from_numpy(Acoo.data).float(),
            torch.Size(Acoo.shape),
        ).coalesce()


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