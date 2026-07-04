from copy import deepcopy
from utils.registry import MODEL_REGISTRY

# make the registry decorators run
import models.shape_matching_model

__all__ = ["build_model"]


def build_model(opt):
    """
    Constructs a model given the full option dict.

    Args:
        opt (dict): must contain
            model_type (str): the registered model class name, e.g. ShapeMatchingModel
        The whole (copied) opt is forwarded to the model constructor.
    """
    opt = deepcopy(opt)
    model_type = opt.pop('model_type')
    model_cls = MODEL_REGISTRY.get(model_type)
    return model_cls(opt)
