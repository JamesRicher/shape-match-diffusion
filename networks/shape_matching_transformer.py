import torch
import torch.nn as nn

from utils.registry import NETWORK_REGISTRY
from networks.blocks import IntraShapeBlock, InterShapeBlock


@NETWORK_REGISTRY.register()
class ShapeMatchingEncoder(nn.Module):
    """
    The overall shape matching block. This combines inter and intra shape attention blocks interleave

    Args:
        in_dim (int): input feature dimensins
        dim (int): the dimension of the latent transformer space
        heads (int): the nunber of attention heads (must divide dim)
        depth (int): the number of layer clones
        mlp_ratio (float): relative size of the hidden layer of the FFN components
        dropout (float): dropout probability
    """
    def __init__(self, in_dim: int, dim: int, heads: int, depth: int = 2,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Linear(in_dim, dim)
        # ONE intra block per layer, reused for both shapes => weight sharing.
        self.intra = nn.ModuleList(
            IntraShapeBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth))
        self.inter = nn.ModuleList(
            InterShapeBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth))
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, feat_x: torch.Tensor, feat_y: torch.Tensor,
                intra_bias_x: torch.Tensor | None = None,
                intra_bias_y: torch.Tensor | None = None,
                inter_bias: torch.Tensor | None = None):
        """
        feat_x: (B, Nx, in_dim), feat_y: (B, Ny, in_dim)
        intra_bias_*: (B, N, N) per-shape relation (kNN / geodesic) or None
        inter_bias:   (B, Nx, Ny) cross relation (noisy assignment) or None
        returns refined embeddings (emb_x, emb_y), each (B, N, dim)
        """
        x, y = self.embed(feat_x), self.embed(feat_y)
        for intra, inter in zip(self.intra, self.inter):
            x = intra(x, bias=intra_bias_x)
            y = intra(y, bias=intra_bias_y)   # same instance
            x, y = inter(x, y, bias=inter_bias)
        return self.out_norm(x), self.out_norm(y)
