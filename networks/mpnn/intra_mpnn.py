"""Intra-shape stage: an anisotropic geodesic MPNN layer (networks/mpnn design).

One layer = two AdaLN-Zero sublayers, mirroring the DiT block discipline of
networks/denoiser_blocks.py but with message passing where that trunk has
self-attention:

  1. message passing over the fixed geodesic-kNN graph
       message      m_ij = FilterNet(rbf(d_ij)) * MLP_msg([h_i, h_j - h_i])
       aggregation  a_i  = [mean_j m_ij ; max_j m_ij]
       update       h_i  = h_i + gate * MLP_upd([h_i, a_i])
  2. pointwise FFN

Lineage: EdgeConv message construction (Wang et al. 2019) with continuous
edge-attribute conditioning via a multiplicative per-channel distance filter
(SchNet cfconv / MoNet family), GraphSAGE-style mean‖max aggregation and concat
update. No softmax anywhere: this is a hard-cutoff spatial GNN, deliberately NOT
a biased transformer (that is what the parallel MatrixDenoiser already is).

Edges carry no state — the distance RBF is a static attribute precomputed once
per denoiser call and reused by every layer. One instance per depth is shared
across both shapes (call once per shape), preserving pair-swap symmetry.
Identity at init: AdaLN-Zero gates start at 0.
"""
import torch
import torch.nn as nn

from networks.denoiser_blocks import AdaLNModulation, modulate
from networks.layers import FeedForward
from networks.mpnn.geometry import gather_neighbors


class IntraMPNNLayer(nn.Module):
    """Anisotropic geodesic message-passing layer with AdaLN-Zero conditioning.

    Args:
        dim: token width.
        n_rbf: distance RBF basis size (input width of the filter net).
        mlp_ratio: FFN hidden expansion.
        dropout: dropout inside the FFN.
    """

    def __init__(self, dim: int, n_rbf: int = 16, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm_mp = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(dim, elementwise_affine=False)
        self.filter_net = nn.Sequential(nn.Linear(n_rbf, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.msg_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.upd_mlp = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.ffn = FeedForward(dim, mlp_ratio, dropout)
        self.mod = AdaLNModulation(dim, n_sublayers=2)

    def forward(self, h: torch.Tensor, nbr_idx: torch.Tensor, dist_rbf: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        """
        h:        (B, n, D) node embeddings for ONE shape.
        nbr_idx:  (B, n, k) geodesic-kNN indices (static per call).
        dist_rbf: (B, n, k, n_rbf) RBF-embedded normalised geodesic distances.
        c:        (B, D) conditioning vector (timestep spine).
        """
        sh_mp, sc_mp, g_mp, sh_f, sc_f, g_f = self.mod(c)

        # ---- sublayer 1: message passing ---- #
        hn = modulate(self.norm_mp(h), sh_mp, sc_mp)             # (B, n, D)
        h_nbr = gather_neighbors(hn, nbr_idx)                    # (B, n, k, D)
        h_ctr = hn.unsqueeze(2).expand_as(h_nbr)                 # receiver, broadcast over k
        m = self.filter_net(dist_rbf) * self.msg_mlp(torch.cat([h_ctr, h_nbr - h_ctr], dim=-1))
        a_mean = m.mean(dim=2)                                   # (B, n, D)
        a_max = m.max(dim=2).values                              # (B, n, D)
        h = h + g_mp * self.upd_mlp(torch.cat([hn, a_mean, a_max], dim=-1))

        # ---- sublayer 2: FFN ---- #
        h = h + g_f * self.ffn(modulate(self.norm_ffn(h), sh_f, sc_f))
        return h
