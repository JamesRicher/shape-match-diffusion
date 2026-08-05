"""Graph/geometry utilities for the MPNN denoiser (networks/mpnn).

Pure functions + one small module, all batched dense — graphs are (B, n, k) index
tensors gathered with `gather_neighbors`, never edge lists (n=512, small k: dense
gathers beat scatter ops and avoid the torch-geometric dependency).

Distances are intrinsic geodesics in sqrt-area units (dataset convention). kNN
distances are normalised per shape by the mean k-th-NN radius before the RBF
embedding, so the embedding is sampling-density- and scale-invariant and one fixed
RBF range serves every dataset/n_sparse.
"""
import torch
import torch.nn as nn


def knn_from_dist(D: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """k nearest neighbours (self excluded) from a dense distance matrix.

    Args:
        D: (B, n, n) symmetric distances, zero diagonal.
        k: neighbours per node (k < n).
    Returns:
        idx:  (B, n, k) neighbour indices, nearest first.
        dist: (B, n, k) the corresponding distances.
    """
    n = D.shape[-1]
    D_noself = D + torch.eye(n, device=D.device, dtype=D.dtype) * torch.finfo(D.dtype).max
    dist, idx = D_noself.topk(k, dim=-1, largest=False)
    return idx, dist


def normalise_knn_dist(dist: torch.Tensor) -> torch.Tensor:
    """Divide by the per-shape mean k-th-NN radius -> dimensionless, ~O(1) values.

    dist: (B, n, k) kNN distances. The normaliser is a per-(B,) scalar, so the
    operation is permutation-invariant and identical for both shapes' calls.
    """
    radius = dist[..., -1].mean(dim=-1, keepdim=True).clamp_min(1e-12)  # (B, 1)
    return dist / radius.unsqueeze(-1)


def gather_neighbors(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather per-node neighbour features.

    h: (B, n_src, D); idx: (B, n_dst, k) indices into n_src. Returns (B, n_dst, k, D).
    """
    B, _, D = h.shape
    n_dst, k = idx.shape[1], idx.shape[2]
    flat = idx.reshape(B, n_dst * k, 1).expand(-1, -1, D)
    return torch.gather(h, 1, flat).reshape(B, n_dst, k, D)


def feature_knn(F_dst: torch.Tensor, F_src: torch.Tensor, k: int) -> torch.Tensor:
    """Cross-shape candidate partners by feature distance.

    For each dst point, the k src points with the closest (euclidean) descriptor.
    F_dst: (B, n_dst, d); F_src: (B, n_src, d). Returns idx (B, n_dst, k) into src.
    """
    d2 = torch.cdist(F_dst, F_src)
    return d2.topk(k, dim=-1, largest=False).indices


class RBFEmbed(nn.Module):
    """Scalar -> K Gaussian radial-basis activations over a fixed [lo, hi] range.

    The expansion (rather than feeding the raw scalar) is what gives the filter
    nets distance-band selectivity from init. Centres/width are fixed buffers, not
    parameters — the filter nets downstream do the learning.
    """

    def __init__(self, n_basis: int, lo: float, hi: float):
        super().__init__()
        centers = torch.linspace(lo, hi, n_basis)
        self.register_buffer("centers", centers, persistent=False)
        spacing = (hi - lo) / max(n_basis - 1, 1)
        self.gamma = 1.0 / (2.0 * spacing ** 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (...) -> (..., n_basis)."""
        return torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)


# --------------------------------------------------------------------------- #
# unit tests.  Run: python -m networks.mpnn.geometry
# --------------------------------------------------------------------------- #
def _run_tests() -> None:
    torch.manual_seed(0)
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f'[{"PASS" if ok else "FAIL"}] {name:42s}' + (f'  {detail}' if detail else ""))

    B, n, k = 3, 32, 6
    pts = torch.randn(B, n, 3)
    D = torch.cdist(pts, pts)
    D = 0.5 * (D + D.transpose(-1, -2))

    # --- kNN: no self, correct neighbours vs brute force ------------------- #
    idx, dist = knn_from_dist(D, k)
    no_self = (idx != torch.arange(n).view(1, n, 1)).all()
    brute = (D + torch.eye(n) * 1e9).sort(-1).values[..., :k]
    check("knn excludes self + matches brute force",
          no_self and torch.allclose(dist, brute, atol=1e-6))

    # --- kNN permutation equivariance --------------------------------------- #
    p = torch.randperm(n)
    idx_p, dist_p = knn_from_dist(D[:, p][:, :, p], k)
    inv = torch.empty_like(p)
    inv[p] = torch.arange(n)
    check("knn permutation equivariance",
          torch.allclose(dist_p, dist[:, p], atol=1e-6)
          and (idx_p == inv[idx[:, p]]).all())

    # --- normalisation: scale invariance ------------------------------------ #
    nd1 = normalise_knn_dist(dist)
    nd2 = normalise_knn_dist(knn_from_dist(3.7 * D, k)[1])
    check("dist normalisation scale-invariant", torch.allclose(nd1, nd2, atol=1e-5),
          f"mean={nd1.mean():.3f}")

    # --- gather_neighbors matches a loop ------------------------------------ #
    h = torch.randn(B, n, 5)
    g = gather_neighbors(h, idx)
    ok = all(torch.equal(g[b, i, j], h[b, idx[b, i, j]])
             for b in range(B) for i in (0, n // 2) for j in (0, k - 1))
    check("gather_neighbors correct", ok)

    # --- feature_knn sanity: nearest feature is found ------------------------ #
    Fx = torch.randn(B, n, 8)
    ci = feature_knn(Fx, Fx, 1)
    # against itself with distinct rows, the 0-distance match is the point itself
    check("feature_knn self-match", (ci.squeeze(-1) == torch.arange(n)).all())

    # --- RBF: shape, peak at centre, decay ---------------------------------- #
    rbf = RBFEmbed(8, 0.0, 3.0)
    out = rbf(torch.tensor([0.0, 3.0]))
    check("rbf shape + endpoint peaks", out.shape == (2, 8)
          and abs(out[0, 0] - 1) < 1e-6 and abs(out[1, -1] - 1) < 1e-6)

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    _run_tests()
