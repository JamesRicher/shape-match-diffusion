from copy import deepcopy

from utils.registry import DENSIFIER_REGISTRY
from .base_densifier import BaseDensifier, DensifyContext

# make the registry decorators run (add concrete densifier modules here as they land)
import densifiers.functional_map
import densifiers.nearest_anchor
import densifiers.row_stochastic
import densifiers.spectral_refine   # ZoomOutDensifier + the shared zoomout_refine helper

__all__ = ["build_densifier", "BaseDensifier", "DensifyContext"]


def build_densifier(densifier_opt):
    """Constructs a densifier given its options.

    Args:
        densifier_opt (dict or None): when None/falsy or type is null, returns None
            (sparse-only, no densification). Otherwise must contain
                type (str): the registered densifier class name.
            The whole (copied) opt is forwarded to the constructor.
    """
    if not densifier_opt or densifier_opt.get('type') is None:
        return None
    densifier_opt = deepcopy(densifier_opt)
    densifier_type = densifier_opt.pop('type')
    densifier_cls = DENSIFIER_REGISTRY.get(densifier_type)
    return densifier_cls(densifier_opt)
