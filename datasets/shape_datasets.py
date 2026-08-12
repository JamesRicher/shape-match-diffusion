from datasets.dataset_bases import SingleShapeDataset, PairShapeDataset, SparsePairShapeDataset, sort_list
import os
from glob import glob
from itertools import product
import numpy as np
import torch
from utils.data_utils import load_map
from utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SingleFaustDataset(SingleShapeDataset):
    """
    FAUST_r dataset one at a time

    Args:
        phase (str): one of test train or full
    """

    def __init__(self,
                 data_root,
                 phase: str = "train",
                 ret_faces=True,
                 ret_feats=True,
                 ret_corr=True,
                 ret_dist=True,
                 ret_evecs=False,
                 num_evecs=200,
                 feats_dir='feats'):
        super().__init__(data_root, ret_faces, ret_feats, ret_corr, ret_dist, ret_evecs, num_evecs, feats_dir)
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        assert len(self) == 100, f'FAUST dataset should contain 100 human body shapes, but get {len(self)}.'
        if phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:80]
            if self.corr_files:
                self.corr_files = self.corr_files[:80]
            if self.dist_files:
                self.dist_files = self.dist_files[:80]
            if self.feat_files:
                self.feat_files = self.feat_files[:80]
            self._size = 80
        elif phase == 'test':
            if self.off_files:
                self.off_files = self.off_files[80:]
            if self.corr_files:
                self.corr_files = self.corr_files[80:]
            if self.dist_files:
                self.dist_files = self.dist_files[80:]
            if self.feat_files:
                self.feat_files = self.feat_files[80:]
            self._size = 20


@DATASET_REGISTRY.register()
class SingleSmalDataset(SingleShapeDataset):
    """
    SMAL_r dataset one at a time

    Args:
        phase (str): one of test train or full
        category (bool):
    """
    def __init__(self,
                 data_root,
                 phase='train',
                 category=True,
                 ret_faces=True,
                 ret_feats=True,
                 ret_corr=True,
                 ret_dist=True,
                 ret_evecs=False,
                 num_evecs=200):
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        self.phase = phase
        self.category = category
        super(SingleSmalDataset, self).__init__(data_root, ret_faces, ret_feats, ret_corr, ret_dist, ret_evecs, num_evecs)

        self.flip_up = True


    def _init_data(self):
        if self.category:
            txt_file = os.path.join(self.data_root, f'{self.phase}_cat.txt')
        else:
            txt_file = os.path.join(self.data_root, f'{self.phase}.txt')
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                self.off_files += [os.path.join(self.data_root, 'off', f'{line}.off')]
                if self.ret_corr:
                    self.corr_files += [os.path.join(self.data_root, 'corres', f'{line}.vts')]
                if self.ret_dist:
                    self.dist_files += [os.path.join(self.data_root, 'dist', f'{line}.mat')]
                if self.ret_feats:
                    self.feat_files += [os.path.join(self.data_root, 'feats', f'{line}.npy')]


@DATASET_REGISTRY.register()
class SingleScapeDataset(SingleShapeDataset):
    def __init__(self,
                 data_root,
                 phase="train",
                 ret_faces=True,
                 ret_feats=True,
                 ret_corr=True,
                 ret_dist=True,
                 ret_evecs=False,
                 num_evecs=200,
                 feats_dir='feats'):
        super(SingleScapeDataset, self).__init__(data_root, ret_faces, ret_feats, ret_corr, ret_dist, ret_evecs, num_evecs, feats_dir)
        assert phase in ['train', 'test', 'full'], f'Invalid phase {phase}, only "train" or "test" or "full"'
        assert len(self) == 71, f'FAUST dataset should contain 71 human body shapes, but get {len(self)}.'
        if phase == 'train':
            if self.off_files:
                self.off_files = self.off_files[:51]
            if self.corr_files:
                self.corr_files = self.corr_files[:51]
            if self.dist_files:
                self.dist_files = self.dist_files[:51]
            if self.feat_files:
                self.feat_files = self.feat_files[:51]
            self._size = 51
        elif phase == 'test':
            if self.off_files:
                self.off_files = self.off_files[51:]
            if self.corr_files:
                self.corr_files = self.corr_files[51:]
            if self.dist_files:
                self.dist_files = self.dist_files[51:]
            if self.feat_files:
                self.feat_files = self.feat_files[51:]
            self._size = 20


@DATASET_REGISTRY.register()
class PairFaustDataset(PairShapeDataset):
    def __init__(self,
                 data_root,
                 phase="train",
                 ret_faces=True,
                 ret_feats=True,
                 ret_corr=True,
                 ret_dist=True,
                 ret_evecs=False,
                 num_evecs=200,
                 exclude_self=False):
        dataset = SingleFaustDataset(data_root, phase, ret_faces, ret_feats, ret_corr, ret_dist, ret_evecs, num_evecs)
        super().__init__(dataset, exclude_self=exclude_self)


@DATASET_REGISTRY.register()
class SparsePairFaustDataset(SparsePairShapeDataset):
    """FAUST_r pairs with FPS-sparse tokens + bijective sparse GT (diffusion matcher)."""
    def __init__(self,
                 data_root,
                 phase="train",
                 n_sparse=128,
                 ret_evecs=False,
                 num_evecs=200,
                 exclude_self=False,
                 fps_metric="geodesic",
                 feats_dir='feats'):
        dataset = SingleFaustDataset(data_root, phase, ret_faces=True, ret_feats=False,
                                     ret_corr=True, ret_dist=True, ret_evecs=ret_evecs,
                                     num_evecs=num_evecs, feats_dir=feats_dir)
        super().__init__(dataset, n_sparse=n_sparse, phase=phase, exclude_self=exclude_self,
                         fps_metric=fps_metric)


@DATASET_REGISTRY.register()
class PairSmalDataset(PairShapeDataset):
    def __init__(self,
                 data_root,
                 phase='train',
                 category=True,
                 ret_faces=True,
                 ret_feats=True,
                 ret_corr=True,
                 ret_dist=True,
                 ret_evecs=False,
                 num_evecs=200,
                 exclude_self=False):
        dataset = SingleSmalDataset(data_root, phase, category, ret_faces, ret_feats, ret_corr, ret_dist, ret_evecs, num_evecs)
        super().__init__(dataset=dataset, exclude_self=exclude_self)


@DATASET_REGISTRY.register()
class SparsePairSmalDataset(SparsePairShapeDataset):
    """SMAL_r pairs with FPS-sparse tokens + bijective sparse GT (diffusion matcher)."""
    def __init__(self,
                 data_root,
                 phase="train",
                 category=True,
                 n_sparse=128,
                 ret_evecs=False,
                 num_evecs=200,
                 exclude_self=False,
                 fps_metric="geodesic"):
        dataset = SingleSmalDataset(data_root, phase, category, ret_faces=True,
                                    ret_feats=False, ret_corr=True, ret_dist=True,
                                    ret_evecs=ret_evecs, num_evecs=num_evecs)
        super().__init__(dataset, n_sparse=n_sparse, phase=phase, exclude_self=exclude_self,
                         fps_metric=fps_metric)


@DATASET_REGISTRY.register()
class SingleShrec19Dataset(SingleShapeDataset):
    """SHREC19_r shapes one at a time (test benchmark; geometry only).

    SHREC19_r has no shared-template .vts and no dist/ or feats/ dir -- the ground truth
    is a fixed set of per-pair dense pointwise maps (corres/{i}_{j}.map) attached by
    PairShrec19Dataset. So corr, dist and feats default off here; enable ret_dist /
    ret_evecs once those caches are generated (see PairShrec19Dataset).
    """
    def __init__(self,
                 data_root,
                 ret_faces=True,
                 ret_feats=False,
                 ret_corr=False,
                 ret_dist=False,
                 ret_evecs=False,
                 num_evecs=200):
        super().__init__(data_root, ret_faces, ret_feats, ret_corr, ret_dist, ret_evecs, num_evecs)


def _shrec19_map_files(data_root):
    """The .map files defining SHREC19's pair set. Following ULRSSM, shape 40 (which has
    holes) is dropped as both source and target, so 430 -> 407 pairs."""
    corr_path = os.path.join(data_root, "corres")
    assert os.path.isdir(corr_path), f"Invalid path {corr_path} not containing .map files"
    files = [f for f in sort_list(glob(f"{corr_path}/*.map"))
             if "40" not in os.path.basename(f)]
    assert files, f"No .map files under {corr_path}"
    return files


class _Shrec19Pairs:
    """Pair enumeration + per-pair GT for SHREC19_r, shared by the dense and sparse datasets.

    SHREC19 ships a fixed set of directed dense pointwise maps (corres/{i}_{j}.map) rather than
    a shared template, so pairs come from the files on disk instead of a cartesian product, and
    corr is attached per pair rather than per shape. GT is index-aligned so first['corr'][k] <->
    second['corr'][k], matching calculate_geodesic_error(dist_x, corr_x, corr_y, p2p):
    first['corr'] = arange(V_first) and second['corr'][k] = target vertex of source vertex k.

    That per-pair corr is still a valid template for bijective sparse sampling -- the template
    is X's own vertex set (corr_x = arange), and consistent_bijective_fps resolves the map's
    many-to-one collisions exactly as it does for a .vts. The one thing a shared template would
    add is the reverse direction, which SHREC19 genuinely lacks (the maps are directed).

    Mixed in BEFORE PairShapeDataset so these hooks win; __init__ must set self.map_files
    before calling super().__init__ (which builds combinations from them).
    """
    pair_level_corr = True                  # corr comes from _pair_gt, not the single-shape .vts

    def _build_combinations(self, exclude_self):
        # filenames are 1-indexed shape ids; no self-pairs exist, so exclude_self is moot.
        return [tuple(int(v) - 1 for v in os.path.splitext(os.path.basename(f))[0].split("_"))
                for f in self.map_files]

    def _pair_gt(self, item, index):
        corr = load_map(self.map_files[index])                      # (V_first,) 0-indexed
        item["first"]["corr"] = torch.arange(corr.shape[0]).long()
        item["second"]["corr"] = torch.from_numpy(corr).long()
        return item


@DATASET_REGISTRY.register()
class PairShrec19Dataset(_Shrec19Pairs, PairShapeDataset):
    """SHREC19_r evaluation pairs (test-only benchmark). See _Shrec19Pairs for the GT layout.

    dist/feats/evecs default off (SHREC19_r ships neither dist/ nor feats/); turn on ret_dist
    (and ret_evecs for a DiffusionNet extractor) once those caches exist -- geodesic-error
    eval needs the first shape's dist matrix.
    """
    def __init__(self,
                 data_root,
                 phase="test",
                 ret_faces=True,
                 ret_feats=False,
                 ret_dist=False,
                 ret_evecs=False,
                 num_evecs=200):
        assert phase == "test", f"PairShrec19Dataset is test-only, got phase={phase!r}"
        self.map_files = _shrec19_map_files(data_root)
        super().__init__(SingleShrec19Dataset(data_root, ret_faces=ret_faces, ret_feats=ret_feats,
                                              ret_corr=False, ret_dist=ret_dist,
                                              ret_evecs=ret_evecs, num_evecs=num_evecs))


@DATASET_REGISTRY.register()
class SparsePairShrec19Dataset(_Shrec19Pairs, SparsePairShapeDataset):
    """SHREC19_r evaluation pairs with FPS-sparse tokens (sparse diffusion matcher, test-only).

    Both eval regimes work here: dense MGE under honest independent FPS, and the bijective
    sparse_ avg_error/acc pass, whose template is X's own vertex set (see _Shrec19Pairs). feats
    are never loaded -- a DiffusionNet extractor recomputes descriptors from the cached spectral
    operators, so only faces/dist/evecs are needed (ret_evecs is required by both the extractor
    and the densifier).
    """
    def __init__(self,
                 data_root,
                 phase="test",
                 n_sparse=512,
                 ret_evecs=True,
                 num_evecs=200,
                 fps_metric="geodesic",
                 exclude_self=False,           # accepted for config-shape parity; no self-pairs exist
                 ret_faces=True):
        assert phase == "test", f"SparsePairShrec19Dataset is test-only, got phase={phase!r}"
        assert ret_evecs, "SparsePairShrec19Dataset needs ret_evecs (extractor + densifier read the ops)"
        self.map_files = _shrec19_map_files(data_root)
        # dist required for geodesic MGE (and geodesic FPS); no feats (extractor recomputes them).
        super().__init__(SingleShrec19Dataset(data_root, ret_faces=ret_faces, ret_feats=False,
                                              ret_corr=False, ret_dist=True,
                                              ret_evecs=ret_evecs, num_evecs=num_evecs),
                         n_sparse=n_sparse, phase=phase, exclude_self=exclude_self,
                         fps_metric=fps_metric)


@DATASET_REGISTRY.register()
class SingleDT4DDataset(SingleShapeDataset):
    """DT4D_r (DeformingThings4D remeshed) shapes one at a time; ULRSSM semantics.

    Shapes are listed by {phase}.txt (lines '<category>/<frame>'), and 'pumpkinhulk' is
    excluded entirely (ULRSSM ignore-list) -> 177 train / 85 test usable shapes over 9
    categories. Each category has one reference template (the frame whose .vts is the
    identity); a frame's .vts has T lines (T = that category's template size), line i =
    the frame vertex of reference point i (1-based on disk, load_corres 0-bases it).
    Maps are many-to-one, and T differs per category.

    The raw dataset ships only off/ and corres/, so feats/dist/evecs default off (like
    SHREC19); enable once the caches exist (dist/<cat>/<frame>.mat,
    feats/<cat>/<frame>.npy, diffusion/). Frame basenames repeat across categories, so
    item['name'] is '<category>_<frame>'.
    """
    def __init__(self,
                 data_root,
                 phase='train',
                 ret_faces=True,
                 ret_feats=False,
                 ret_corr=True,
                 ret_dist=False,
                 ret_evecs=False,
                 num_evecs=200):
        assert phase in ['train', 'test'], f'Invalid phase {phase}, only "train" or "test"'
        self.phase = phase
        self.ignored_categories = ['pumpkinhulk']
        super().__init__(data_root, ret_faces, ret_feats, ret_corr, ret_dist, ret_evecs, num_evecs)

    def _init_data(self):
        with open(os.path.join(self.data_root, f'{self.phase}.txt'), 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            if line.split('/')[0] in self.ignored_categories:
                continue
            self.off_files += [os.path.join(self.data_root, 'off', f'{line}.off')]
            if self.ret_corr:
                self.corr_files += [os.path.join(self.data_root, 'corres', f'{line}.vts')]
            if self.ret_dist:
                self.dist_files += [os.path.join(self.data_root, 'dist', f'{line}.mat')]
            if self.ret_feats:
                self.feat_files += [os.path.join(self.data_root, 'feats', f'{line}.npy')]

    def category(self, index):
        """Category of shape `index` (the parent directory of its .off)."""
        return os.path.basename(os.path.dirname(self.off_files[index]))

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['name'] = f'{self.category(index)}_{item["name"]}'   # basenames repeat across cats
        return item


def _dt4d_combinations(dataset, inter_class, exclude_self):
    """Ordered (i, j) pairs under the ULRSSM DT4D split rules.

    intra: cat(i) == cat(j) (optionally minus self-pairs). inter: cat(i) != cat(j) AND
    corres/cross_category_corres/<cat_i>_<cat_j>.vts exists -- cross-maps only link
    crypto<->{drake, mannequin, ninja, prisoner, skeletonzombie, zlorp} (both directions),
    so every inter pair has crypto at one end; mousey/ortiz are intra-only."""
    cats = [dataset.category(i) for i in range(len(dataset))]
    if inter_class:
        cross_dir = os.path.join(dataset.data_root, 'corres', 'cross_category_corres')
        avail = {tuple(os.path.splitext(f)[0].split('_')) for f in os.listdir(cross_dir)}
        return [(i, j) for i, j in product(range(len(cats)), repeat=2)
                if cats[i] != cats[j] and (cats[i], cats[j]) in avail]
    return [(i, j) for i, j in product(range(len(cats)), repeat=2)
            if cats[i] == cats[j] and not (exclude_self and i == j)]


def _dt4d_cross_corr(data_root, cat_x, cat_y, cache):
    """Reference-to-reference cross map cat_x -> cat_y as a (T_x,) LongTensor (0-based),
    memoised in `cache`. NOT frame vertices: it must be composed with the target frame's
    own .vts (corr_y = Y.vts[cross]) before use."""
    key = (cat_x, cat_y)
    if key not in cache:
        path = os.path.join(data_root, 'corres', 'cross_category_corres', f'{cat_x}_{cat_y}.vts')
        cache[key] = torch.from_numpy(np.loadtxt(path, dtype=np.int64) - 1)
    return cache[key]


def _dt4d_compose(item, dataset, i, j, cache):
    """Rewrite an inter-class pair's GT in place: corr stays indexed by X's reference
    points (length T_x), and the target column becomes Y.vts[cross_{catX_catY}]. Using
    the raw cross map instead would be wrong for any non-reference target frame."""
    cross = _dt4d_cross_corr(dataset.data_root, dataset.category(i), dataset.category(j), cache)
    item['second']['corr'] = item['second']['corr'][cross]
    return item


class _DT4DPairs:
    """Pair enumeration + cross-category GT composition for DT4D_r, shared by the dense and
    sparse datasets. Inter pairs are composed BEFORE sparsification, so the K-indexed corr pair
    the bijective FPS consumes already spans the two categories via the reference alignment.

    Mixed in BEFORE PairShapeDataset so these hooks win; __init__ must call _init_dt4d before
    super().__init__ (which builds combinations from inter_class).
    """
    def _init_dt4d(self, inter_class):
        self.inter_class = inter_class
        self._cross_cache = {}

    def _build_combinations(self, exclude_self):
        return _dt4d_combinations(self.dataset, self.inter_class, exclude_self)

    def _pair_gt(self, item, index):
        if self.inter_class and self.dataset.ret_corr:
            i, j = self.combinations[index]
            item = _dt4d_compose(item, self.dataset, i, j, self._cross_cache)
        return item


@DATASET_REGISTRY.register()
class PairDT4DDataset(_DT4DPairs, PairShapeDataset):
    """DT4D_r dense pairs with the ULRSSM split semantics.

    inter_class=False (default): intra-category pairs only (825 test / 3587 train ordered
    pairs incl. self-pairs; exclude_self drops the diagonal). inter_class=True:
    cross-category pairs where a cross-map exists (1200 test / 5292 train; crypto-hub star
    topology), with corr COMPOSED per _dt4d_compose so first['corr'][k] <->
    second['corr'][k] over X's reference points -- exactly what
    calculate_geodesic_error(dist_x, corr_x, corr_y, p2p) expects.

    dist/feats/evecs default off (raw DT4D_r ships neither); geodesic-error eval needs the
    first shape's dist cache once generated.
    """
    def __init__(self,
                 data_root,
                 phase='train',
                 inter_class=False,
                 ret_faces=True,
                 ret_feats=False,
                 ret_corr=True,
                 ret_dist=False,
                 ret_evecs=False,
                 num_evecs=200,
                 exclude_self=False):
        self._init_dt4d(inter_class)
        super().__init__(SingleDT4DDataset(data_root, phase, ret_faces, ret_feats, ret_corr,
                                           ret_dist, ret_evecs, num_evecs),
                         exclude_self=exclude_self)


@DATASET_REGISTRY.register()
class SparsePairDT4DDataset(_DT4DPairs, SparsePairShapeDataset):
    """DT4D_r pairs with FPS-sparse tokens + bijective sparse GT (diffusion matcher),
    under the ULRSSM split rules (pumpkinhulk exclusion, intra/inter options) of
    PairDT4DDataset. Inter pairs compose the cross map BEFORE sparsification, so the
    K-indexed corr pair the bijective FPS consumes already spans the two categories via
    the reference alignment; n_sparse is bounded by the SOURCE category's template size.
    Needs the dist/ and feats/ caches (not shipped with raw DT4D_r) -- construction
    succeeds but item loading fails until they are generated."""
    def __init__(self,
                 data_root,
                 phase="train",
                 inter_class=False,
                 n_sparse=128,
                 ret_evecs=False,
                 num_evecs=200,
                 exclude_self=False,
                 fps_metric="geodesic"):
        self._init_dt4d(inter_class)
        super().__init__(SingleDT4DDataset(data_root, phase, ret_faces=True, ret_feats=False,
                                           ret_corr=True, ret_dist=True, ret_evecs=ret_evecs,
                                           num_evecs=num_evecs),
                         n_sparse=n_sparse, phase=phase, exclude_self=exclude_self,
                         fps_metric=fps_metric)


@DATASET_REGISTRY.register()
class PairScapeDataset(PairShapeDataset):
    def __init__(self,
                 data_root,
                 phase='train',
                 ret_faces=True,
                 ret_feats=True,
                 ret_corr=True,
                 ret_dist=True,
                 ret_evecs=False,
                 num_evecs=200,
                 exclude_self=False):
        dataset = SingleScapeDataset(data_root, phase, ret_faces, ret_feats, ret_corr, ret_dist, ret_evecs, num_evecs)
        super().__init__(dataset, exclude_self=exclude_self)


@DATASET_REGISTRY.register()
class SparsePairScapeDataset(SparsePairShapeDataset):
    """SCAPE_r pairs with FPS-sparse tokens + bijective sparse GT (diffusion matcher)."""
    def __init__(self,
                 data_root,
                 phase="train",
                 n_sparse=128,
                 ret_evecs=False,
                 num_evecs=200,
                 exclude_self=False,
                 fps_metric="geodesic",
                 feats_dir='feats'):
        dataset = SingleScapeDataset(data_root, phase, ret_faces=True, ret_feats=False,
                                     ret_corr=True, ret_dist=True, ret_evecs=ret_evecs,
                                     num_evecs=num_evecs, feats_dir=feats_dir)
        super().__init__(dataset, n_sparse=n_sparse, phase=phase, exclude_self=exclude_self,
                         fps_metric=fps_metric)
