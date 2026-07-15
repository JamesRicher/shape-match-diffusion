from copy import deepcopy
from utils.registry import NETWORK_REGISTRY

# make the registry decorators run
import networks.shape_matching_transformer
import networks.matrix_denoiser
import networks.gcn_feature_extractor

__all__ = ["build_network"]


def build_network(network_opt):
    """
    Constructs a network (nn.Module) given its options.

    Args:
        network_opt (dict): must contain at least
            type (str): the registered network class name, e.g. ShapeMatchingEncoder
        Every other key is forwarded as a keyword argument to the constructor.
    """
    network_opt = deepcopy(network_opt)
    network_type = network_opt.pop('type')
    network_cls = NETWORK_REGISTRY.get(network_type)
    return network_cls(**network_opt)
