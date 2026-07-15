"""Learnable per-point GCN feature extractor (BendingGraphs Graphite, patch variant).

Replaces the frozen .npy features with a small graph net trained end-to-end with the
matcher. It produces one descriptor per SPARSE (FPS) point, but computes each from a LOCAL
PATCH of the FULL-resolution mesh around that point (BendingGraphs' patch mode), so the
feature reflects fine local surface geometry rather than the coarse FPS-neighbourhood:

    for each FPS point p:
        patch = the `patch_size` full-mesh vertices nearest p by geodesic distance
        local graph = geodesic-kNN within the patch
        run TAGConv layers on the patch, then max-pool -> one (out_dim) descriptor for p

Output is (1, n, out_dim) -- the shape the denoiser consumes for F, dropping into
_sparse_inputs in place of the loaded feat.

TAGConv is the topology-adaptive polynomial GCN of BendingGraphs (src/models/tag_conv.py):
x' = sum_{k=0..K} A_hat^k x Theta_k with a symmetric-normalised adjacency. Reimplemented
here in dense torch (batched over the n patches) to avoid the torch_geometric dependency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import NETWORK_REGISTRY


def build_patches(verts: torch.Tensor, dist: torch.Tensor, idx: torch.Tensor, patch_size: int):
    """Gather local full-mesh patches for each FPS point. Param-free, so it can run in a
    DataLoader worker (off the main thread) to overlap the topk/gather with GPU compute.

    Returns (D_patch (n,p,p), patch_verts (n,p,3), center_verts (n,3)) -- small tensors, so
    the caller can then drop the full (N,N) dist instead of shipping it to the main process.
    Column 0 of each patch is the FPS centre itself (nearest to itself, geodesic 0)."""
    idx = idx.long()
    p = min(patch_size, dist.shape[0])
    _, patch_idx = torch.topk(dist[idx], p, dim=-1, largest=False)              # (n, p)
    D_patch = dist[patch_idx.unsqueeze(-1), patch_idx.unsqueeze(-2)].float()    # (n, p, p)
    patch_verts = verts[patch_idx].float()                                      # (n, p, 3)
    center_verts = verts[idx].float()                                           # (n, 3)
    return D_patch, patch_verts, center_verts


class TAGConv(nn.Module):
    """Topology-adaptive graph conv: sum_{k=0..K} (A_hat^k x) Theta_k.

    Batched: x is (..., p, in_ch) and A_hat a dense normalised adjacency (..., p, p); a
    separate linear per hop plays the role of Theta_k, with a single bias on the k=0 term.
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
            xk = A_hat @ xk                       # propagate one more hop (batched matmul)
            out = out + self.lins[k](xk)
        return out


@NETWORK_REGISTRY.register()
class GCNFeatureExtractor(nn.Module):
    """Per-FPS-point GCN descriptor from local full-mesh patches.

    forward(verts, dist, idx) -> (1, n, out_dim), where verts (N, 3) and dist (N, N) are the
    FULL-mesh coords and geodesic matrix, and idx (n,) are the full-mesh indices of the FPS
    points. Patches are gathered on dist's device (typically CPU) so the full (N, N) matrix
    never has to move to the GPU; only the small per-patch tensors do.

    Args:
        out_dim: per-point feature dim (BendingGraphs FEATURE_SIZE, default 64).
        node_in: patch node features. 'xyz' = patch coords relative to the centre point,
            scale-normalised by patch radius (local, translation-invariant); 'anchor' =
            geodesic distance from each patch vertex to the centre (intrinsic radial coord).
        hidden: hidden widths between node_in and out_dim (mirror Graphite 16->32).
        hops: per-layer TAGConv hop counts K; len(hops) == len(hidden) + 1.
        patch_size: full-mesh vertices per patch (the local receptive field).
        knn: neighbours per vertex when building the within-patch geodesic graph.
        sigma: edge-weight bandwidth exp(-d^2/sigma^2); None -> median neighbour distance.
    """
    def __init__(self, out_dim: int = 64, node_in: str = 'xyz',
                 hidden: tuple = (16, 32), hops: tuple = (1, 2, 3),
                 patch_size: int = 64, knn: int = 8,
                 sigma: float | None = None, eps: float = 1e-8):
        super().__init__()
        if node_in not in ('xyz', 'anchor'):
            raise ValueError(f"node_in must be 'xyz' or 'anchor', got {node_in!r}")
        if len(hops) != len(hidden) + 1:
            raise ValueError('len(hops) must equal len(hidden) + 1')
        self.node_in = node_in
        self.patch_size = patch_size
        self.knn = knn
        self.sigma = sigma
        self.eps = eps

        in_ch = 3 if node_in == 'xyz' else 1
        widths = [in_ch, *hidden, out_dim]
        self.convs = nn.ModuleList(
            TAGConv(widths[i], widths[i + 1], hops[i]) for i in range(len(hops)))
        # LayerNorm over the feature dim (not BatchNorm): the extractor runs on one shape at
        # a time, so per-shape batch stats would normalise each shape differently and wreck
        # cross-shape feature alignment. LayerNorm is per-point, train/eval-identical, and
        # matches the denoiser's normalisation.
        self.norms = nn.ModuleList(nn.LayerNorm(w) for w in widths[1:])

    def _adjacency(self, D: torch.Tensor) -> torch.Tensor:
        """Symmetric-normalised geodesic-kNN adjacency from batched patch geodesics.

        D: (n, p, p) -> A_hat (n, p, p)."""
        p = D.shape[-1]
        k = min(self.knn, p - 1)
        # k smallest geodesics per row, dropping the self column (distance 0 on diagonal)
        nn_d, nn_idx = torch.topk(D, k + 1, dim=-1, largest=False)
        nn_d, nn_idx = nn_d[..., 1:], nn_idx[..., 1:]
        sigma = self.sigma if self.sigma is not None else nn_d.median().clamp_min(self.eps)
        w = torch.exp(-(nn_d ** 2) / (sigma ** 2 + self.eps))       # (n, p, k)

        A = torch.zeros_like(D).scatter_(-1, nn_idx, w)
        A = torch.maximum(A, A.transpose(-1, -2))                   # symmetrise
        A = A + torch.eye(p, device=D.device, dtype=D.dtype)        # self-loops
        d_inv_sqrt = A.sum(-1).clamp_min(self.eps).pow(-0.5)        # (n, p)
        return d_inv_sqrt.unsqueeze(-1) * A * d_inv_sqrt.unsqueeze(-2)

    def _node_features(self, patch_verts, center_verts, D_patch):
        """patch_verts (n,p,3), center_verts (n,3), D_patch (n,p,p) -> (n,p,in_ch)."""
        if self.node_in == 'xyz':
            rel = patch_verts - center_verts.unsqueeze(1)           # centre-relative coords
            scale = rel.norm(dim=-1).amax(-1, keepdim=True).clamp_min(self.eps)  # (n,1)
            return rel / scale.unsqueeze(-1)
        # 'anchor': geodesic to the centre (patch column 0 is the centre point itself)
        radial = D_patch[..., :1]                                   # (n,p,1)
        return radial / radial.amax(-2, keepdim=True).clamp_min(self.eps)

    def forward(self, patches) -> torch.Tensor:
        """patches = (D_patch (n,p,p), patch_verts (n,p,3), center_verts (n,3)) from
        build_patches, typically produced in a DataLoader worker. Returns (1, n, out_dim)."""
        dev = self.convs[0].lins[0].weight.device
        D_patch, patch_verts, center_verts = (t.to(dev, non_blocking=True) for t in patches)

        A_hat = self._adjacency(D_patch)
        x = self._node_features(patch_verts, center_verts, D_patch)
        last = len(self.convs) - 1
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = norm(conv(x, A_hat))
            if i < last:
                x = F.relu(x)
        return x.amax(dim=1).unsqueeze(0)                # max-pool patch -> (1, n, out_dim)

    def extract(self, verts: torch.Tensor, dist: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Convenience: patchify (on the main thread) then run. Used where no worker-side
        patchification is available (e.g. the diffusion model's _sparse_inputs)."""
        return self.forward(build_patches(verts, dist, idx, self.patch_size))
