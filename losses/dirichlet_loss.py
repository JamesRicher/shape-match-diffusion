import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class DirichletLoss(nn.Module):
    """
    Dirichlet energy of the per-vertex embeddings on each mesh, E = f^T L f, using the
    cached cotangent stiffness matrix L. Penalises embeddings that vary sharply across
    neighbouring vertices, encouraging spatially smooth feature fields.

    Requires the cached operators, i.e. the dataset must be built with ``ret_evecs=True``
    so ``data['first']['L']`` / ``data['second']['L']`` are present. Assumes a batch
    size of 1 (sparse operators do not batch trivially).

    Args:
        normalize (bool): L2-normalise embeddings before computing the energy. Default True.
        loss_weight (float): scalar multiplier on the returned loss. Default 1.0.
    """

    def __init__(self, normalize: bool = True, loss_weight: float = 1.0):
        super().__init__()
        self.normalize = normalize
        self.loss_weight = loss_weight

    def _energy(self, emb: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        # emb (N, d), L sparse (N, N). Energy summed over feature dims, per-vertex mean.
        f = F.normalize(emb, dim=-1) if self.normalize else emb
        return (f * torch.sparse.mm(L, f)).sum() / emb.shape[0]

    def forward(self, emb_x: torch.Tensor, emb_y: torch.Tensor, data: dict) -> torch.Tensor:
        assert emb_x.shape[0] == 1, 'DirichletLoss currently supports batch size = 1'
        L_x, L_y = data['first']['L'], data['second']['L']
        energy = self._energy(emb_x[0], L_x) + self._energy(emb_y[0], L_y)
        return self.loss_weight * energy
