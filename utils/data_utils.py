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


def load_map(map_file: str) -> np.ndarray:
    """Load a SHREC19-style dense pointwise map (.map file).

    Line k (1-indexed) holds the 1-indexed target-shape vertex matching source vertex k.
    Returns a 0-indexed (V_src,) array of target vertex indices. Unlike a .vts template
    map this is a per-pair correspondence, so it is attached by the pair dataset, not the
    single-shape dataset.
    """
    assert os.path.isfile(map_file), f"Invalid .map file: {map_file}"
    return np.loadtxt(map_file, dtype=np.int64) - 1


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
    """Geodesic distance matrix, as float32.

    The .mat stores float64, which is 200 MB per FAUST shape (520 MB on DT4D) — doubling both
    the per-worker LRU footprint and every item's copy, for precision no downstream consumer
    keeps: the dataset converts to float32 tensors anyway. Casting here makes that conversion a
    no-op view instead of a full copy.
    """
    assert os.path.isfile(dist_file), f"Invalid .mat file: {dist_file}"
    return scipy.io.loadmat(dist_file)["dist"].astype(np.float32, copy=False)


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


def fps(pool, n: int, start: int, dist=None):
    """Greedy farthest-point sampling of n points from pool, starting at index `start`.

    dist is None (default): extrinsic FPS on squared Euclidean distance in pool's coords.
    dist given: intrinsic FPS on the geodesic matrix `dist` (N, N), aligned row/col to pool,
    so points are spread evenly along the surface irrespective of pose. Squared-vs-raw is
    immaterial -- FPS depends only on the ordering of distances, which both preserve.
    """
    N = pool.shape[0]
    indices = np.empty(n, dtype=np.int64)
    indices[0] = start
    distances = np.full(N, np.inf)
    for i in range(1, n):
        d = dist[indices[i-1]] if dist is not None else np.sum((pool - pool[indices[i-1]])**2, axis=1)
        distances = np.minimum(d, distances)
        indices[i] = np.argmax(distances)
    return indices


def _greedy_bijective(order: np.ndarray, corr_x: np.ndarray, corr_y: np.ndarray, n=None) -> np.ndarray:
    """Walk template positions in `order`, keeping one only if its X- and Y-vertices are
    both unused. Guarantees sparse_x and sparse_y are each injective. Stops at n picks
    (or exhausts `order` when n is None)."""
    sel, seen_x, seen_y = [], set(), set()
    for k in order:
        xv, yv = int(corr_x[k]), int(corr_y[k])
        if xv in seen_x or yv in seen_y:
            continue
        seen_x.add(xv); seen_y.add(yv); sel.append(k)
        if n is not None and len(sel) == n:
            break
    return np.asarray(sel, dtype=np.int64)


def consistent_bijective_fps(verts_x: np.ndarray, corr_x: np.ndarray, corr_y: np.ndarray,
                             n: int, start: int, dist_x: np.ndarray = None) -> np.ndarray:
    """Pick n template positions giving a bijective sparse correspondence on both shapes.

    The template point -> vertex map is many-to-one per shape (FAUST's .vts: 5000 template
    points -> ~3.5k distinct vertices; SHREC19's per-pair .map likewise, with X's own vertex
    set as the template), so corr_x[K] and corr_y[K] can collide. FPS on
    X's covered points gives a well-spread ordering; :func:`_greedy_bijective` keeps only
    positions distinct on both shapes, so the GT is an exact permutation. FPS depth grows
    until n points are collected (cheap: caps well below the full template for small n).

    Args:
        verts_x: (Vx, 3) vertices of the first shape.
        corr_x, corr_y: (T,) template -> vertex maps for the two shapes.
        n: number of sparse points.
        start: FPS start position (index into the T covered points).
        dist_x: (Vx, Vx) geodesic matrix on X. When given, FPS is geodesic over the covered
            points (dist_x[corr_x][:, corr_x]); None -> Euclidean on verts_x[corr_x].

    Returns K (n,) template positions, in FPS order.
    """
    T = corr_x.shape[0]
    # corr_x is the identity when the template IS X's own vertex set (a per-pair dense map,
    # e.g. SHREC19). Skip the gather then: fancy-indexing dist_x would copy the whole (V, V)
    # geodesic matrix per item for nothing.
    if T == verts_x.shape[0] and bool((corr_x == np.arange(T)).all()):
        pool, pool_dist = verts_x, dist_x
    else:
        pool = verts_x[corr_x]
        pool_dist = dist_x[corr_x][:, corr_x] if dist_x is not None else None
    # Greedy farthest-point sampling is NESTED: the first d entries of the ordering are the same
    # whatever depth it was computed to. So depth only decides how many candidates the bijective
    # filter gets to reject before the loop below doubles and retries — the returned points are
    # identical either way. 2n is enough at every dataset's usual rejection rate; 4n roughly
    # doubled the cost of what is the single most expensive step in building a training item.
    depth = min(T, max(2 * n, 64))
    while True:
        sel = _greedy_bijective(fps(pool, depth, start, dist=pool_dist), corr_x, corr_y, n)
        if len(sel) == n:
            return sel
        if depth == T:
            raise ValueError(f"only {len(sel)} bijective points available, need {n}")
        depth = min(T, depth * 2)


def bijective_fps_order(verts_x: np.ndarray, corr_x: np.ndarray, corr_y: np.ndarray,
                        start: int) -> np.ndarray:
    """All bijective template positions in FPS order (exhaustive over the template).

    Its length is the maximum n_sparse for which a bijective sparse GT exists for this
    pair and start — i.e. the point beyond which consistent bijective FPS is impossible.
    O(T^2); for a one-off/vis use, not the training path (use consistent_bijective_fps).
    """
    T = corr_x.shape[0]
    return _greedy_bijective(fps(verts_x[corr_x], T, start), corr_x, corr_y)


def knn_from_dist(dist_sub, k: int):
    """k nearest neighbours per row of an (n, n) distance submatrix, excluding self.

    Self is the zero-distance diagonal, so it sorts first and is dropped. Returns a
    (n, k) long tensor of neighbour indices (k clamped to n-1). Torch in, torch out.
    """
    import torch
    n = dist_sub.shape[0]
    k = min(k, n - 1)
    return torch.argsort(dist_sub, dim=-1)[:, 1:k + 1].long()


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def permutation_matrix(perm: np.ndarray) -> np.ndarray:
    N = len(perm)
    P = np.zeros((N,N), dtype=np.float32)
    P[np.arange(N), perm] = 1.0
    return P