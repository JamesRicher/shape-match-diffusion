from collections import OrderedDict

import torch
import torch.nn.functional as F

from utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class ShapeMatchingModel(BaseModel):
    """
    Wraps a ``ShapeMatchingEncoder`` (or any network with the same
    ``forward(feat_x, feat_y, ...) -> (emb_x, emb_y)`` signature) registered under
    ``opt['networks']['encoder']``.

    Inference (``validate_single``) is fully wired: it refines the per-vertex
    features with the encoder and reads off a point-to-point map by nearest neighbour
    in the learned embedding space. The training objective is intentionally left as a
    hook (``_compute_losses``) since it is the research-specific part.
    """

    def _encode(self, data):
        """Run the encoder on a shape pair. Returns (emb_x, emb_y), each (B, N, dim)."""
        encoder = self.networks['encoder']

        def batched(shape):
            feat = shape['feat']
            if feat.dim() == 2:              # (N, C) -> (1, N, C) when unbatched
                feat = feat.unsqueeze(0)
            return feat.to(self.device)

        feat_x = batched(data['first'])
        feat_y = batched(data['second'])
        return encoder(feat_x, feat_y)

    def _compute_losses(self, emb_x, emb_y, data):
        """Populate a loss dict from the registered losses.

        Convention: each registered loss is called as ``loss(emb_x, emb_y, data)``.
        Override this (and add a ``losses`` package) to implement the actual training
        objective — with no losses configured, training is a no-op.
        """
        loss_metrics = OrderedDict()
        for name, loss_fn in self.losses.items():
            loss_metrics[name] = loss_fn(emb_x, emb_y, data)
        return loss_metrics

    def feed_data(self, data):
        emb_x, emb_y = self._encode(data)
        self.emb_x, self.emb_y = emb_x, emb_y
        self.loss_metrics = self._compute_losses(emb_x, emb_y, data)

    @torch.no_grad()
    def validate_single(self, data):
        emb_x, emb_y = self._encode(data)          # (1, Nx, d), (1, Ny, d)
        emb_x = F.normalize(emb_x[0], dim=-1)
        emb_y = F.normalize(emb_y[0], dim=-1)
        sim = emb_y @ emb_x.T                       # (Ny, Nx)
        return sim.argmax(dim=1)                    # p2p: shape y -> shape x, (Ny,)
