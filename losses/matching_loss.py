import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised InfoNCE over ground-truth correspondences.

    Given the refined per-vertex embeddings of a shape pair and the ground-truth
    matched vertices (``data['first']['corr'][k]`` <-> ``data['second']['corr'][k]``),
    pull the embeddings of matched vertices together and push non-matched apart,
    symmetrically over both matching directions.

    The temperature is a fixed hyperparameter (not learnable): a free learnable
    scale would simply collapse the objective.

    Args:
        temperature (float): softmax temperature. Default 0.07.
        loss_weight (float): scalar multiplier on the returned loss. Default 1.0.
    """

    def __init__(self, temperature: float = 0.07, loss_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.loss_weight = loss_weight

    def forward(self, emb_x: torch.Tensor, emb_y: torch.Tensor, data: dict) -> torch.Tensor:
        corr_x, corr_y = data['first']['corr'], data['second']['corr']
        B = emb_x.shape[0]

        total = emb_x.new_zeros(())
        for b in range(B):
            cx = corr_x[b] if corr_x.dim() == 2 else corr_x
            cy = corr_y[b] if corr_y.dim() == 2 else corr_y

            ex = F.normalize(emb_x[b][cx], dim=-1)     # (M, d) matched anchors on X
            ey = F.normalize(emb_y[b][cy], dim=-1)     # (M, d) matched anchors on Y

            logits = ex @ ey.t() / self.temperature    # (M, M), diagonal are positives
            labels = torch.arange(logits.shape[0], device=logits.device)
            total = total + 0.5 * (F.cross_entropy(logits, labels)
                                   + F.cross_entropy(logits.t(), labels))

        return self.loss_weight * total / B
