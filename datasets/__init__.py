from copy import deepcopy
import torch
from utils.registry import DATASET_REGISTRY
from paths import DEFAULT_DATA_ROOTS

# make the registry decorators run
import datasets.shape_datasets

__all__ = ["build_dataset"]

def build_dataset(dataset_opt):
    """
    Constructs a dataset object given a dataset and other options

    Args:
        dataset_opt is a dict that must contain at least
        name (str): the name of the dataset
        type (str): the class name e.g. SingleFaustDataset
    """
    dataset_opt = deepcopy(dataset_opt)
    type = dataset_opt.pop('type')
    name = dataset_opt.pop('name')

    if dataset_opt.get('data_root'):
        root = dataset_opt.pop('data_root')
    else:
        root = None

    root = root or DEFAULT_DATA_ROOTS.get(name)
    if root is None:
        raise ValueError(f"No data_root given and no default registered for {name}")

    dataset_cls = DATASET_REGISTRY.get(type)
    return dataset_cls(data_root=root, **dataset_opt)
