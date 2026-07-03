from utils.data_utils import *
from torch.utils.data import Dataset
from glob import glob
import torch
import re
import os
from itertools import product

def sort_list(l):
    try:
        return list(sorted(l, key=lambda x: int(re.search(r'\d+(?=\.)', x).group())))
    except AttributeError:
        return sorted(l)
    

class ShapeCache:
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
                 ret_feats: bool=True,
                 ret_corr: bool=True,
                 ret_dist: bool=True,
                 ret_evecs: bool=False,
                 num_evecs: int=200):
        super().__init__()
        assert os.path.isdir(data_root), f"Invalid data root for SingleShapeDataset: {data_root}"

        # initialisation
        self.data_root = data_root
        self.ret_faces = ret_faces
        self.ret_feats = ret_feats
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
            feat_path = os.path.join(self.data_root, 'feats')
            assert os.path.isdir(feat_path), f"Inavlid path {feat_path} not containing .npy files"
            self.feat_files = sort_list(glob(f'{feat_path}/*.npy'))
    

    def __getitem__(self, index):
        item = dict()

        # get shape name
        off_file = self.off_files[index]
        basename = os.path.splitext(os.path.basename(off_file))[0]
        item['name'] = basename

        # get vertices and faces
        verts, faces = load_off(off_file)
        verts = verts.astype(np.float32)
        faces = faces.astype(np.int64)
        item['verts'] = torch.from_numpy(verts)
        if self.ret_faces:
            item['faces'] = torch.from_numpy(faces)

        # get geodesic distance matrix
        if self.ret_dist:
            dist = load_dist(self.dist_files[index])
            item['dist'] = torch.from_numpy(dist).float()

        # get correspondences
        if self.ret_corr:
            corr = load_corres(self.corr_files[index])
            item['corr'] = torch.from_numpy(corr).long()

        # get frozen featrues
        if self.ret_feats:
            feat = load_feats(self.feat_files[index])
            item['feat'] = torch.from_numpy(feat).float()

        # get spectral operators (cached by ULRSSM-style preprocess)
        if self.ret_evecs:
            self._load_ops(item, verts, faces)

        return item

    def _load_ops(self, item, verts_np: np.ndarray, faces_np: np.ndarray):
        ops = load_diffusion_operators(verts_np, faces_np, self.diffusion_dir)
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
    def __init__(self, dataset):
        """
        Pair Shape Dataset

        Args:
            dataset (SingleShapeDataset): single shape dataset
        """
        assert isinstance(dataset, SingleShapeDataset), f'Invalid input data type of dataset: {type(dataset)}'
        self.dataset = dataset
        self.combinations = list(product(range(len(dataset)), repeat=2))

        self.flip_up = self.dataset.flip_up

    def __getitem__(self, index):
        # get index
        first_index, second_index = self.combinations[index]

        item = dict()
        item['first'] = self.dataset[first_index]
        item['second'] = self.dataset[second_index]

        return item

    def __len__(self):
        return len(self.combinations)