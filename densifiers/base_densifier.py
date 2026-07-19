"""Base interface for map densifiers: lift a sparse p2p to a dense whole-shape map.

A densifier is a non-learned post-process (kept out of the training loss, steps.md Step 3):
the diffusion matcher emits a sparse p2p over FPS points, and a densifier completes it to a
dense vertex-to-vertex map for the full mesh. Every scheme shares one signature
(densify(sparse_p2p, ctx) -> dense p2p) so they are config-swappable; they differ only in
which fields of DensifyContext they read (geodesics, features, eigenbases).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DensifyContext:
    """Full-mesh context for one pair, gathered from a dataset item.

    Densifiers consume only the subset they need; unused fields may be left None. The sparse
    p2p is in FPS-index space, so idx_x/idx_y lift it to full-mesh vertices:
    Y vertex idx_y[j] matches X vertex idx_x[sparse_p2p[j]].

    idx_x, idx_y: (n,) sparse FPS index -> full-mesh vertex index, per shape.
    n_x, n_y:     full vertex counts of X (source) and Y (target).
    dist_x, dist_y: (N,N) dense geodesic matrices (nearest-anchor / feature-NN).
    feat_x, feat_y: (N,d) dense per-vertex features (feature-NN).
    evecs_x, evecs_y, evals_x, evals_y, mass_x, mass_y: spectral operators (ZoomOut / FM).
    """
    idx_x: torch.Tensor
    idx_y: torch.Tensor
    n_x: int
    n_y: int
    dist_x: Optional[torch.Tensor] = None
    dist_y: Optional[torch.Tensor] = None
    feat_x: Optional[torch.Tensor] = None
    feat_y: Optional[torch.Tensor] = None
    evecs_x: Optional[torch.Tensor] = None
    evecs_y: Optional[torch.Tensor] = None
    evals_x: Optional[torch.Tensor] = None
    evals_y: Optional[torch.Tensor] = None
    mass_x: Optional[torch.Tensor] = None
    mass_y: Optional[torch.Tensor] = None


class BaseDensifier(ABC):
    """Lift a sparse p2p to a dense whole-shape p2p. Subclasses register in DENSIFIER_REGISTRY."""

    def __init__(self, opt: Optional[dict] = None):
        self.opt = opt or {}
        # Feature source for any data term: True asks the caller to fill ctx.feat_x/feat_y with
        # DENSE GCN descriptors (one patch per full-mesh vertex) instead of the frozen .npy
        # field. Declared here but fulfilled by the model, which owns the extractor network; a
        # densifier with no data term simply ignores it.
        self.gcn_feats = self.opt.get('gcn_feats', False)

    @abstractmethod
    def densify(self, sparse_p2p: torch.Tensor, ctx: DensifyContext) -> torch.Tensor:
        """Args:
            sparse_p2p: (n,) LongTensor, sparse Y FPS index -> sparse X FPS index.
            ctx: full-mesh context for this pair.
        Returns:
            (n_y,) LongTensor, target Y vertex -> source X vertex (full-mesh indices).
        """
        raise NotImplementedError
