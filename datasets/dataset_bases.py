from utils.data_utils import *
from torch.utils.data import Dataset
from glob import glob
import torch
import re
import os
from collections import OrderedDict
from itertools import product
from typing import Optional

def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', x).group())))
    except AttributeError:
        return sorted(l)
    

class ShapeCache:
    """Lazy path-keyed loader for shape data, owned by a Dataset instance.

    The heavy per-shape entries (the full N×N geodesic distance matrix and the spectral
    operators) are held in bounded LRU stores, so worker RAM stays flat regardless of how
    many distinct shapes an epoch touches — important with multiple DataLoader workers on a
    shared machine, where an unbounded cache would grow to (#shapes × ~100MB) per worker.
    The light entries (feats/verts/faces/corres) stay unbounded; they're small.

    Args:
        dist_maxsize: max distinct geodesic matrices kept (LRU). None => unbounded.
        ops_maxsize: max distinct spectral-operator sets kept (LRU). None => unbounded.
    """
    def __init__(self, dist_maxsize: Optional[int] = 16, ops_maxsize: Optional[int] = 16):
        self._feats: dict = {}
        self._corres: dict = {}
        self._verts: dict = {}
        self._faces: dict = {}
        self._dists: OrderedDict = OrderedDict()
        self._ops: OrderedDict = OrderedDict()
        self._dist_maxsize = dist_maxsize
        self._ops_maxsize = ops_maxsize

    @staticmethod
    def _lru_get(store: OrderedDict, key, loader, maxsize: Optional[int]):
        """Return store[key], loading via loader() on a miss and evicting the least-recently
        used entry once the store exceeds maxsize (maxsize=None => never evict)."""
        if key in store:
            store.move_to_end(key)                 # mark most-recently used
            return store[key]
        value = loader()
        store[key] = value
        if maxsize is not None:
            while len(store) > maxsize:
                store.popitem(last=False)          # drop least-recently used
        return value

    def feats(self, feat_file: str) -> np.ndarray:
        if feat_file not in self._feats:
            self._feats[feat_file] = load_feats(feat_file).astype(np.float32)
        return self._feats[feat_file]

    def geom(self, off_file: str) -> Tuple[np.ndarray, np.ndarray]:
        if off_file not in self._verts:
            v, f = load_off(off_file)
            self._verts[off_file] = v.astype(np.float32)
            self._faces[off_file] = f.astype(np.int64)
        return self._verts[off_file], self._faces[off_file]

    def corres(self, corr_file: str) -> np.ndarray:
        if corr_file not in self._corres:
            self._corres[corr_file] = load_corres(corr_file)
        return self._corres[corr_file]

    def dist(self, dist_file: str) -> np.ndarray:
        return self._lru_get(self._dists, dist_file,
                             lambda: load_dist(dist_file), self._dist_maxsize)

    def ops(self, off_file: str, diffusion_dir: str) -> dict:
        key = (off_file, diffusion_dir)
        return self._lru_get(self._ops, key,
                             lambda: load_diffusion_operators(*self.geom(off_file), diffusion_dir),
                             self._ops_maxsize)


class SingleShapeDataset(Dataset):
    """
    Represents a shape dataset in which we take elements one at a time - used mostly for debugging

    Args:
        data_root (str): the path to the base dir of the dataset
        ret_faces (bool): whether to return faces
        ret_feats (bool): whether to return feats
        ret_corr (bool): whether to return GT correspondence
        ret_dist (bool): whether to return distances
        ret_evecs (bool): whether to return LBO spectral operators (evecs, evals, mass, L, gradX, gradY)
        num_evecs (int): number of eigenvectors to load when ret_evecs is True
    """
    def __init__(self, data_root: str,
                 ret_faces: bool=True,
                 ret_feats: bool=False,
                 ret_corr: bool=True,
                 ret_dist: bool=True,
                 ret_evecs: bool=False,
                 num_evecs: int=200,
                 feats_dir: str='feats'):
        super().__init__()
        assert os.path.isdir(data_root), f"Invalid data root for SingleShapeDataset: {data_root}"

        # initialisation
        self.data_root = data_root
        self.ret_faces = ret_faces
        self.ret_feats = ret_feats
        self.feats_dir = feats_dir     # subdir of data_root holding the frozen feature .npy files
        self.ret_corr = ret_corr
        self.ret_dist = ret_dist
        self.ret_evecs = ret_evecs
        self.num_evecs = num_evecs
        self.diffusion_dir = os.path.join(data_root, 'diffusion') if ret_evecs else None

        if self.ret_evecs:
            assert os.path.isdir(self.diffusion_dir), \
                f"Invalid path {self.diffusion_dir} not containing cached diffusion operators"

        self.off_files = []
        self.corr_files = [] if self.ret_corr else None
        self.dist_files = [] if self.ret_dist else None
        self.feat_files = [] if self.ret_feats else None

        self.cache = ShapeCache()

        self._init_data()

        # check what we have loaded
        self._size = len(self.off_files)
        assert self._size > 0

        if self.ret_corr:
            assert self._size == len(self.corr_files)
        if self.ret_dist:
            assert self._size == len(self.dist_files)
        if self.ret_feats:
            assert self._size == len(self.feat_files)

        # polyscope settings
        self.flip_up = False

    
    def _init_data(self):
        off_path = os.path.join(self.data_root, 'off')
        assert os.path.isdir(off_path), f"Invalid path {off_path} not containing .off files"
        self.off_files = sort_list(glob(f'{off_path}/*.off'))

        if self.ret_dist:
            dist_path = os.path.join(self.data_root, 'dist')
            assert os.path.isdir(dist_path), f"Invalid path {dist_path} not containing .mat files"
            self.dist_files = sort_list(glob(f'{dist_path}/*.mat'))

        if self.ret_corr:
            corr_path = os.path.join(self.data_root, 'corres')
            assert os.path.isdir(corr_path), f"Invalid path {corr_path} not containing .vts files"
            self.corr_files = sort_list(glob(f'{corr_path}/*.vts'))

        if self.ret_feats:
            feat_path = os.path.join(self.data_root, self.feats_dir)
            assert os.path.isdir(feat_path), f"Inavlid path {feat_path} not containing .npy files"
            self.feat_files = sort_list(glob(f'{feat_path}/*.npy'))
    

    def __getitem__(self, index):
        item = dict()

        # get shape name
        off_file = self.off_files[index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        item['name'] = basename

        # get vertices and faces
        verts, faces = self.cache.geom(off_file)
        item['verts'] = torch.from_numpy(verts)
        if self.ret_faces:
            item['faces'] = torch.from_numpy(faces)

        # get geodesic distance matrix
        if self.ret_dist:
            dist = self.cache.dist(self.dist_files[index])
            item['dist'] = torch.from_numpy(dist).float()

        # get correspondences
        if self.ret_corr:
            corr = self.cache.corres(self.corr_files[index])
            item['corr'] = torch.from_numpy(corr).long()

        # get frozen featrues
        if self.ret_feats:
            feat = self.cache.feats(self.feat_files[index])
            item['feat'] = torch.from_numpy(feat).float()

        # get spectral operators (cached by ULRSSM-style preprocess)
        if self.ret_evecs:
            self._load_ops(item, off_file)

        return item

    def _load_ops(self, item, off_file: str):
        ops = self.cache.ops(off_file, self.diffusion_dir)
        assert ops['k_eig'] >= self.num_evecs, \
            f"Cache has {ops['k_eig']} evecs, requested {self.num_evecs}"

        k = self.num_evecs
        evecs = ops['evecs'][:, :k]
        evals = ops['evals'][:k]
        mass = ops['mass']

        item['evecs'] = torch.from_numpy(evecs).float()
        item['evecs_trans'] = torch.from_numpy(evecs.T * mass[None]).float()
        item['evals'] = torch.from_numpy(evals).float()
        item['mass'] = torch.from_numpy(mass).float()
        item['L'] = sparse_np_to_torch(ops['L'])
        item['gradX'] = sparse_np_to_torch(ops['gradX'])
        item['gradY'] = sparse_np_to_torch(ops['gradY'])


    def __len__(self):
        return self._size


class PairShapeDataset(Dataset):
    """Pairs of shapes drawn from a SingleShapeDataset.

    Two hooks let a benchmark customise what a pair IS without reimplementing the class:
    :meth:`_build_combinations` decides which (i, j) pairs exist, and :meth:`_pair_gt`
    rewrites a fetched pair's ground truth. Benchmarks whose GT is per-pair rather than a
    shared template (SHREC19's .map files) or whose pair set is dictated by split rules
    (DT4D) need only those two, and inherit sparsification and everything else for free.
    """
    def __init__(self, dataset, exclude_self=False):
        """
        Args:
            dataset (SingleShapeDataset): single shape dataset
            exclude_self (bool): drop the diagonal (i, i) self-pairs. Self-pairs are
                trivial (a shape matched to itself, ~zero geodesic error) and bias
                evaluation metrics; ``run_baselines.py`` skips them, so set this True
                for val/test to keep results comparable. Default False.
        """
        assert isinstance(dataset, SingleShapeDataset), f'Invalid input data type of dataset: {type(dataset)}'
        self.dataset = dataset
        self.combinations = self._build_combinations(exclude_self)
        self.flip_up = self.dataset.flip_up

    def _build_combinations(self, exclude_self):
        """The ordered (i, j) shape-index pairs this dataset enumerates. Default: the full
        cartesian product, minus the diagonal when exclude_self. Override when the benchmark
        decides the pair set (DT4D's split rules, SHREC19's shipped .map files)."""
        return [(i, j) for i, j in product(range(len(self.dataset)), repeat=2)
                if not (exclude_self and i == j)]

    def _pair_gt(self, item, index):
        """Hook to rewrite a fetched pair's GT, called before sparsification. Default: none --
        both shapes already carry a 'corr' into the shared template. Override when the GT is
        pair-level (SHREC19) or must be composed across categories (DT4D inter-class)."""
        return item

    def __getitem__(self, index):
        first_index, second_index = self.combinations[index]
        item = {'first': self.dataset[first_index], 'second': self.dataset[second_index]}
        return self._pair_gt(item, index)

    def __len__(self):
        return len(self.combinations)


class SparsePairShapeDataset(PairShapeDataset):
    """Pair dataset augmented with FPS-subsampled sparse tokens for the matrix
    diffusion matcher.

    Wraps a SingleShapeDataset exactly like PairShapeDataset and keeps the full
    'first'/'second' dicts (needed for eval-time densification). Per shape it adds a
    'sparse' sub-dict (same key names as the full dict: feat, dist, verts, plus idx),
    holding n_sparse points chosen by FPS on 'first' (X) and pushed through the shared
    template to 'second' (Y), so the sparse GT is bijective by construction (main
    note 5). "Template" means whatever indexes both shapes' corr: a shared .vts for
    FAUST/SCAPE/SMAL/DT4D, or X's own vertex set when the GT is a per-pair dense map
    (SHREC19, corr_x = arange). Pair-level fields ('gt_perm', 'fps_idx') sit at the
    top level. Keeping
    the per-shape/first-second layout mirrors the denoiser's pair-swap symmetry.

    Args:
        dataset: a SingleShapeDataset with ret_corr and ret_dist True. ret_feats is optional:
            a learnable extractor recomputes descriptors from the sparse idx + operators, so
            the sparse 'feat' block is only attached when the dataset carries frozen features.
        n_sparse: number of sparse points per shape.
        phase: "train" randomises the FPS start each item (sweeps the surface over
            epochs, doubles as augmentation); anything else uses a fixed start
            (index-derived) for comparable val/test numbers.
        exclude_self: forwarded to PairShapeDataset (drop the (i, i) self-pairs).

    independent_fps (attribute, default False): eval-only honest-sampling mode. When True,
    each shape is FPS'd on its OWN geometry (Y is NOT the GT image of X's points), so no
    ground truth enters instance construction and there is no bijective sparse target -- the
    realistic test setup. Emits sparse tokens but no gt_perm/fps_idx; only dense MGE (GT via
    .vts, inside the metric) is a valid score. Left False for training and the fast dev
    metric; flipped on solely by evaluate.py so it can never leak into training.
    """
    # False: both shapes carry a 'corr' into a shared template, loaded by the single-shape
    # dataset (ret_corr). True: there is no shared template and _pair_gt attaches a per-pair
    # corr instead (SHREC19), so ret_corr is legitimately off on the single-shape dataset.
    pair_level_corr = False

    def __init__(self, dataset, n_sparse: int = 128, phase: str = "train", exclude_self: bool = False,
                 fps_metric: str = "geodesic"):
        super().__init__(dataset, exclude_self=exclude_self)
        assert dataset.ret_corr or self.pair_level_corr, \
            "SparsePairShapeDataset needs corr: set ret_corr, or attach it per pair " \
            "(_pair_gt) and set pair_level_corr"
        assert dataset.ret_dist, "SparsePairShapeDataset needs dist"
        assert fps_metric in ("geodesic", "euclidean"), \
            f"fps_metric must be 'geodesic' or 'euclidean', got {fps_metric!r}"
        self.n_sparse = n_sparse
        self.train = (phase == "train")
        self.independent_fps = False
        # (A) covered-vertex honest-FPS training fraction. When >0, each train step is, with
        # this probability, sampled with source and target FPS'd INDEPENDENTLY over their own
        # GT-covered vertices (the test regime) instead of the bijective correspondence. The
        # gaussian soft target still gets full signal because every query is covered. Set from
        # config after construction (like independent_fps), so it never touches eval/val.
        self.independent_train_prob = 0.0
        self.fps_metric = fps_metric        # 'geodesic' (intrinsic, pose-robust) or 'euclidean'

    @staticmethod
    def _sparse_view(shape, idx):
        """Per-shape sparse token dict for the n picked vertices. 'feat' is attached only when
        the shape carries frozen features (ret_feats); a learnable extractor recomputes
        descriptors from idx + the cached operators, so feats are optional (see class doc)."""
        view = {
            'idx': idx,
            'verts': shape['verts'][idx],           # (n, 3), viz/debug (not a denoiser input)
            'dist': shape['dist'][idx][:, idx],     # (n, n) area-normalised geodesic submatrix
        }
        if 'feat' in shape:
            view['feat'] = shape['feat'][idx]       # (n, d_f) frozen features, when present
        return view

    def _independent_train_item(self, item, x, y, index):
        """(A) Honest independent-FPS training instance on the GT-covered vertices.

        Source (X, keys) and target (Y, queries) are FPS'd independently -- separate random
        starts, each spread on its OWN geometry -- so the two sparse sets do NOT correspond,
        matching the test-time regime that bijective training never sees. Reusing
        consistent_bijective_fps per shape only guarantees the n picks are distinct vertices;
        the independence comes from the separate starts and separate geometries.

        Unlike eval-only _independent_item this keeps full supervision: every query is a covered
        vertex, so its GT image on X, corr_x[K_y], is known. The soft target is a geodesic
        kernel centred on that image and evaluated at the (arbitrary) source anchors, so it needs
        no query<->anchor coincidence -- carried downstream as gt_cross_dist rather than gt_perm.
        """
        corr_x = x['corr'].numpy()
        corr_y = y['corr'].numpy()
        T = corr_x.shape[0]
        n = self.n_sparse
        geo = (self.fps_metric == "geodesic")
        dist_x = x['dist'].numpy() if geo else None
        dist_y = y['dist'].numpy() if geo else None

        # Independent starts -> the source and target anchor sets do not correspond.
        start_x = int(np.random.randint(T)) if self.train else index % T
        start_y = int(np.random.randint(T)) if self.train else (index * 7 + 1) % T
        K_x = consistent_bijective_fps(x['verts'].numpy(), corr_x, corr_y, n, start_x, dist_x=dist_x)
        K_y = consistent_bijective_fps(y['verts'].numpy(), corr_y, corr_x, n, start_y, dist_x=dist_y)

        idx_x = torch.from_numpy(corr_x[K_x]).long()   # (n,) source anchors (keys) on X
        idx_y = torch.from_numpy(corr_y[K_y]).long()   # (n,) target queries on Y
        img_x = torch.from_numpy(corr_x[K_y]).long()   # (n,) GT image of each Y query on X

        for shape, idx in ((x, idx_x), (y, idx_y)):
            shape['sparse'] = self._sparse_view(shape, idx)
        # (n_y, n_x) geodesic distance from each query's GT image to each source anchor, in X's
        # metric -- the soft-target support. No gt_perm: the sets are non-corresponding.
        item['gt_cross_dist'] = x['dist'][img_x][:, idx_x].float()
        item['fps_idx'] = torch.from_numpy(K_y).long()  # queries' template positions (cascade reuse)
        return item

    def _independent_item(self, item, x, y):
        """FPS each shape on its own geometry (fixed start, deterministic). No gt_perm: the
        two independently sampled sets have no bijective correspondence. See independent_fps."""
        geo = (self.fps_metric == "geodesic")
        idx_x = torch.from_numpy(fps(x['verts'].numpy(), self.n_sparse, 0,
                                     dist=x['dist'].numpy() if geo else None)).long()
        idx_y = torch.from_numpy(fps(y['verts'].numpy(), self.n_sparse, 0,
                                     dist=y['dist'].numpy() if geo else None)).long()
        for shape, idx in ((x, idx_x), (y, idx_y)):
            shape['sparse'] = self._sparse_view(shape, idx)
        return item

    def __getitem__(self, index):
        item = super().__getitem__(index)          # {'first', 'second'} full dicts
        return self._sparsify(item, index)

    def _sparsify(self, item, index):
        """Attach the sparse views to a fetched pair whose GT is already final (i.e. after
        PairShapeDataset._pair_gt)."""
        x, y = item['first'], item['second']

        if self.independent_fps:
            return self._independent_item(item, x, y)

        if self.train and self.independent_train_prob > 0.0 \
                and np.random.rand() < self.independent_train_prob:
            return self._independent_train_item(item, x, y, index)

        corr_x = x['corr'].numpy()                 # (T,) template point -> vertex on X
        corr_y = y['corr'].numpy()
        T = corr_x.shape[0]
        assert corr_y.shape[0] == T, "paired shapes must share the template length"
        assert self.n_sparse <= T, f"n_sparse={self.n_sparse} exceeds template coverage {T}"

        # FPS over X's covered vertices, kept bijective on both shapes (the .vts map is
        # many-to-one per shape, so corr_y[K] can collide). K indexes both shapes' corr,
        # so sparse_x[i] <-> sparse_y[i] is an exact permutation.
        start = int(np.random.randint(T)) if self.train else index % T
        dist_x = x['dist'].numpy() if self.fps_metric == "geodesic" else None
        K = consistent_bijective_fps(x['verts'].numpy(), corr_x, corr_y, self.n_sparse, start,
                                     dist_x=dist_x)

        idx_x = torch.from_numpy(corr_x[K]).long()  # (n,) vertex indices on X, FPS order
        idx_y = torch.from_numpy(corr_y[K]).long()  # (n,) matched vertex indices on Y

        # Per-shape sparse views, nested in each shape dict with the full dict's own key
        # names (verts/dist, plus feat when frozen features are loaded) and idx for eval densification.
        x['sparse'] = self._sparse_view(x, idx_x)
        y['sparse'] = self._sparse_view(y, idx_y)

        # Pair-level fields. Sparse GT is the identity permutation over K: gt_perm[j, i]
        # = 1 means second point j matches first point i. No target shuffle: the denoiser
        # is intrinsic-only (no index PE) so identity cannot be cheated, and keeping FPS
        # order means D[:, :a] yields well-spread anchors for the spatial encoding.
        n = self.n_sparse
        item['gt_perm'] = torch.eye(n, dtype=torch.float32)   # P0 (n_y, n_x)
        item['fps_idx'] = torch.from_numpy(K).long()          # shared template positions (cascade reuse)
        return item