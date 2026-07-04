from copy import deepcopy
from utils.registry import LOSS_REGISTRY

# make the registry decorators run
import losses.matching_loss
import losses.dirichlet_loss

__all__ = ["build_loss"]


def build_loss(loss_opt):
    """
    Constructs a loss (nn.Module) given its options.

    Args:
        loss_opt (dict): must contain
            type (str): the registered loss class name, e.g. SupervisedContrastiveLoss
        Every other key is forwarded as a keyword argument to the constructor.
    """
    loss_opt = deepcopy(loss_opt)
    loss_type = loss_opt.pop('type')
    loss_cls = LOSS_REGISTRY.get(loss_type)
    return loss_cls(**loss_opt)
