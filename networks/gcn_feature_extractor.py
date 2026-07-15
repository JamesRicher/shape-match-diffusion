"""Learnable per-point GCN feature extractor (BendingGraphs Graphite, reimplemented).

Replaces the frozen .npy features with a small graph net trained end-to-end with the
matcher. It runs on the SPARSE FPS points of one shape: node coordinates plus the sparse
geodesic submatrix, which also defines the graph (a geodesic kNN). Output is per-point
(1, n, out_dim) -- the same shape the denoiser consumes for F, so it drops into
_sparse_inputs in place of the loaded feat.

TAGConv is the topology-adaptive polynomial GCN of BendingGraphs (src/models/tag_conv.py):
x' = sum_{k=0..K} A_hat^k x Theta_k with a symmetric-normalised adjacency. Reimplemented
here in dense torch (one graph per forward, so batching is unnecessary) to avoid the
torch_geometric / torch_scatter dependency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import NETWORK_REGISTRY


class TAGConv(nn.Module):
    """Topology-adaptive graph conv: sum_{k=0..K} (A_hat^k x) Theta_k.

    A_hat is a precomputed dense normalised adjacency (n, n); a separate linear per hop
    plays the role of Theta_k, with a single bias on the k=0 term.
    """
    def __init__(self, in_ch: int, out_ch: int, K: int):
        super().__init__()
        self.K = K
        self.lins = nn.ModuleList(
            nn.Linear(in_ch, out_ch, bias=(k == 0)) for k in range(K + 1))

    def forward(self, x: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        out = self.lins[0](x)
        xk = x
        for k in range(1, self.K + 1):
            xk = A_hat @ xk                       # propagate one more hop
            out = out + self.lins[k](xk)
        return out


@NETWORK_REGISTRY.register()
class GCNFeatureExtractor(nn.Module):
    """Sparse per-point GCN descriptor. forward(points, dist) -> (1, n, out_dim).

    Args:
        out_dim: per-point feature dim (BendingGraphs FEATURE_SIZE, default 64).
        node_in: input node features. 'xyz' = mean-centred, scale-normalised coords
            (extrinsic, like Graphite minus normals); 'anchor' = geodesic distances to the
            first n_anchors points (dist[:, :a]), a fully intrinsic input matching the
            denoiser's no-raw-xyz design.
        hidden: hidden widths between node_in and out_dim (widths mirror Graphite 16->32).
        hops: per-layer TAGConv hop counts K; len(hops) == len(hidden) + 1.
        knn: neighbours per point when building the geodesic graph.
        n_anchors: anchor count for node_in='anchor'.
        sigma: edge-weight bandwidth exp(-d^2/sigma^2); None -> per-graph median neighbour
            distance (scale-adaptive).
    """
    def __init__(self, out_dim: int = 64, node_in: str = 'xyz',
                 hidden: tuple = (16, 32), hops: tuple = (1, 2, 3),
                 knn: int = 8, n_anchors: int = 16,
                 sigma: float | None = None, eps: float = 1e-8):
        super().__init__()
        if node_in not in ('xyz', 'anchor'):
            raise ValueError(f"node_in must be 'xyz' or 'anchor', got {node_in!r}")
        if len(hops) != len(hidden) + 1:
            raise ValueError('len(hops) must equal len(hidden) + 1')
        self.node_in = node_in
        self.knn = knn
        self.n_anchors = n_anchors
        self.sigma = sigma
        self.eps = eps

        in_ch = 3 if node_in == 'xyz' else n_anchors
        widths = [in_ch, *hidden, out_dim]
        self.convs = nn.ModuleList(
            TAGConv(widths[i], widths[i + 1], hops[i]) for i in range(len(hops)))
        self.bns = nn.ModuleList(nn.BatchNorm1d(w) for w in widths[1:])

    def _adjacency(self, dist: torch.Tensor) -> torch.Tensor:
        """Symmetric-normalised geodesic-kNN adjacency A_hat (n, n) from dist (n, n)."""
        n = dist.shape[0]
        k = min(self.knn, n - 1)
        # k smallest geodesics per row, dropping the self column (distance 0 on diagonal)
        nn_d, nn_idx = torch.topk(dist, k + 1, dim=-1, largest=False)
        nn_d, nn_idx = nn_d[:, 1:], nn_idx[:, 1:]
        sigma = self.sigma if self.sigma is not None else nn_d.median().clamp_min(self.eps)
        w = torch.exp(-(nn_d ** 2) / (sigma ** 2 + self.eps))       # (n, k)

        A = torch.zeros(n, n, device=dist.device, dtype=dist.dtype)
        rows = torch.arange(n, device=dist.device).unsqueeze(1).expand(-1, k)
        A[rows, nn_idx] = w
        A = torch.maximum(A, A.t())                                 # symmetrise
        A = A + torch.eye(n, device=dist.device, dtype=dist.dtype)  # self-loops
        d_inv_sqrt = A.sum(-1).clamp_min(self.eps).pow(-0.5)
        return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)

    def _node_features(self, points: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        if self.node_in == 'xyz':
            x = points - points.mean(0, keepdim=True)
            return x / x.norm(dim=-1).max().clamp_min(self.eps)
        a = min(self.n_anchors, dist.shape[0])
        return dist[:, :a]

    def forward(self, points: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        """points: (n, 3) sparse coords; dist: (n, n) sparse geodesic. Returns (1, n, out_dim)."""
        A_hat = self._adjacency(dist)
        x = self._node_features(points, dist)
        last = len(self.convs) - 1
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = bn(conv(x, A_hat))
            if i < last:
                x = F.relu(x)
        return x.unsqueeze(0)
