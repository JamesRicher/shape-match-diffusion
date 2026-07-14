"""Geodesic nearest-anchor densifier: the Step-3 ceiling baseline.

Each Y vertex copies the match of its geodesically-nearest sparse anchor (a Voronoi partition
of Y by the FPS anchors). Zero params, reads only the dense geodesic matrix on Y. Preserves
every anchor exactly but is piecewise-constant, so quality is capped by anchor density and it
propagates anchor errors verbatim -- the floor other schemes must beat.
"""
import torch

from utils.registry import DENSIFIER_REGISTRY
from .base_densifier import BaseDensifier, DensifyContext


@DENSIFIER_REGISTRY.register()
class NearestAnchorDensifier(BaseDensifier):
    """Assign each Y vertex the match of its nearest anchor in Y geodesic distance."""

    def densify(self, sparse_p2p: torch.Tensor, ctx: DensifyContext) -> torch.Tensor:
        assert ctx.dist_y is not None, "NearestAnchorDensifier needs ctx.dist_y"
        device = ctx.dist_y.device
        idx_x, idx_y = ctx.idx_x.to(device), ctx.idx_y.to(device)
        sparse_p2p = sparse_p2p.to(device)

        d_to_anchors = ctx.dist_y[:, idx_y]        # (N_y, n) geodesic to each anchor
        nearest = d_to_anchors.argmin(dim=1)       # (N_y,) sparse Y index of closest anchor
        return idx_x[sparse_p2p[nearest]]          # (N_y,) full-mesh X vertex
