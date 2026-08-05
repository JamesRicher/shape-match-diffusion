"""Cross stage: modular inter-shape feature exchange (networks/mpnn design).

Both variants implement one interface so the denoiser can A/B them by config:

    forward(hx, hy, u, c, cand_x=None, cand_y=None) -> (hx, hy)

  hx: (B, n_x, D)  X tokens        hy: (B, n_y, D)  Y tokens
  u:  (B, n_y, n_x) current log-assignment state (soft-projected, rows = Y),
      the SAME orientation as the repo's P_t.
  cand_*: static feature-kNN candidate indices (MPNN variant only; None for attn).

Variant 'attn' — CrossAttnExchange: dense bidirectional cross-attention with a
per-head gamma * u additive bias. This is exactly the existing transformer inter
stage (AdaLNInterShapeBlock + LogAssignmentBias), composed here against the live
state u instead of the read-in P_t; blocks are imported, not copied.

Variant 'mpnn' — CrossMPNNExchange: hard-cutoff bipartite GNN. Candidate set per
receiver = static feature-kNN ∪ dynamic top-k of the current state (the union
protects recall while u is still noise; duplicates are allowed and harmless).
Messages mirror the intra layer with the state value as the edge attribute:
  m_ij = FilterNet(rbf(u_ij)) * MLP_msg([h_i, h_j - h_i]),  mean‖max, concat update.
Competition among candidates is NOT enforced here (no softmax) — it is delegated
to the state track's Sinkhorn.

Both variants share weights across the two directions (X<-Y and Y<-X use the same
modules with the transposed state), preserving pair-swap symmetry.
"""
import torch
import torch.nn as nn

from networks.denoiser_blocks import AdaLNModulation, modulate, AdaLNInterShapeBlock
from networks.layers import FeedForward
from networks.mpnn.geometry import RBFEmbed, gather_neighbors

# log-state values live in [log(eps), 0] after the soft projection; the RBF grid
# and clamps below use this range.
U_LO = -18.0


class CrossAttnExchange(nn.Module):
    """Variant A: dense cross-attention, state-biased (the transformer inter stage)."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.block = AdaLNInterShapeBlock(dim, heads, mlp_ratio, dropout)
        self.gamma = nn.Parameter(torch.zeros(heads))   # per-head state-bias scale, fade-in

    def forward(self, hx, hy, u, c, cand_x=None, cand_y=None):
        # u is already a log-assignment; clamp only against stray -inf. The block
        # wants the X<-Y bias (B, H, n_x, n_y) and transposes internally for Y<-X.
        base = u.clamp_min(U_LO).transpose(-1, -2).unsqueeze(1)      # (B, 1, n_x, n_y)
        bias = self.gamma.view(1, -1, 1, 1) * base
        return self.block(hx, hy, c, bias=bias)


class CrossMPNNExchange(nn.Module):
    """Variant B: sparse bipartite MPNN over the union candidate graph.

    Args:
        dim: token width.
        k_state: dynamic candidates per receiver from the current state's top-k.
        n_rbf: RBF basis size for the state-value edge attribute.
        mlp_ratio, dropout: FFN settings.
    """

    def __init__(self, dim: int, k_state: int = 10, n_rbf: int = 16,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.k_state = k_state
        self.u_rbf = RBFEmbed(n_rbf, U_LO, 0.0)
        self.norm_mp = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_ffn = nn.LayerNorm(dim, elementwise_affine=False)
        self.filter_net = nn.Sequential(nn.Linear(n_rbf, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.msg_mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.upd_mlp = nn.Sequential(nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.ffn = FeedForward(dim, mlp_ratio, dropout)
        self.mod = AdaLNModulation(dim, n_sublayers=2)

    def _candidates(self, u_dir: torch.Tensor, cand_static: torch.Tensor | None) -> torch.Tensor:
        """Union candidate indices for one direction.

        u_dir: (B, n_dst, n_src) log-state oriented receiver-major.
        cand_static: (B, n_dst, k_f) feature-kNN indices into src, or None.
        Returns (B, n_dst, k_f + k_state) indices (duplicates possible, harmless
        under mean/max up to a mild weighting; kept for simplicity).
        """
        k_s = min(self.k_state, u_dir.shape[-1])
        top = u_dir.topk(k_s, dim=-1).indices                        # (B, n_dst, k_s)
        return top if cand_static is None else torch.cat([cand_static, top], dim=-1)

    def _direction(self, h_dst, h_src, u_dir, cand_static):
        """One receiving direction with the shared weights.

        h_dst: (B, n_dst, D) receivers (already modulated+normed);
        h_src: (B, n_src, D) senders (already modulated+normed);
        u_dir: (B, n_dst, n_src) log-state, receiver-major.
        Returns the aggregated update input (B, n_dst, 3D) for upd_mlp.
        """
        idx = self._candidates(u_dir, cand_static)                   # (B, n_dst, K)
        h_nbr = gather_neighbors(h_src, idx)                         # (B, n_dst, K, D)
        h_ctr = h_dst.unsqueeze(2).expand_as(h_nbr)
        u_val = torch.gather(u_dir, 2, idx).clamp(U_LO, 0.0)         # (B, n_dst, K)
        m = self.filter_net(self.u_rbf(u_val)) \
            * self.msg_mlp(torch.cat([h_ctr, h_nbr - h_ctr], dim=-1))
        return torch.cat([h_dst, m.mean(dim=2), m.max(dim=2).values], dim=-1)

    def forward(self, hx, hy, u, c, cand_x=None, cand_y=None):
        sh_mp, sc_mp, g_mp, sh_f, sc_f, g_f = self.mod(c)

        # ---- sublayer 1: bipartite message passing, both directions ---- #
        # same norm+modulation for both shapes -> pair-swap symmetric
        hx_n = modulate(self.norm_mp(hx), sh_mp, sc_mp)
        hy_n = modulate(self.norm_mp(hy), sh_mp, sc_mp)
        uT = u.transpose(-1, -2)                                     # (B, n_x, n_y)
        upd_x = self._direction(hx_n, hy_n, uT, cand_x)              # X <- Y
        upd_y = self._direction(hy_n, hx_n, u, cand_y)               # Y <- X
        hx = hx + g_mp * self.upd_mlp(upd_x)
        hy = hy + g_mp * self.upd_mlp(upd_y)

        # ---- sublayer 2: FFN ---- #
        hx = hx + g_f * self.ffn(modulate(self.norm_ffn(hx), sh_f, sc_f))
        hy = hy + g_f * self.ffn(modulate(self.norm_ffn(hy), sh_f, sc_f))
        return hx, hy


CROSS_VARIANTS = {
    "attn": CrossAttnExchange,
    "mpnn": CrossMPNNExchange,
}


def build_cross_stage(cross_type: str, dim: int, heads: int, k_state: int,
                      n_rbf: int, mlp_ratio: float, dropout: float) -> nn.Module:
    """Config-string dispatch for the cross stage (the A/B switch)."""
    if cross_type == "attn":
        return CrossAttnExchange(dim, heads, mlp_ratio, dropout)
    if cross_type == "mpnn":
        return CrossMPNNExchange(dim, k_state, n_rbf, mlp_ratio, dropout)
    raise ValueError(f"unknown cross_type {cross_type!r}; options: {sorted(CROSS_VARIANTS)}")
