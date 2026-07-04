import torch
from utils.registry import METRIC_REGISTRY


@METRIC_REGISTRY.register()
def geodesic_distortion(p2p, dist_a, dist_b, sqrt_area_a, sqrt_area_b):
    """
    Global-isometry distortion of the map A -> B: |d_A(i, j) - d_B(p2p(i), p2p(j))|,
    with both geodesic distance matrices normalized by sqrt of surface area.

    Args:
        p2p (torch.Tensor): point-to-point map A -> B. shape [Va]
        dist_a (torch.Tensor): geodesic distance matrix of A. shape [Va, Va]
        dist_b (torch.Tensor): geodesic distance matrix of B. shape [Vb, Vb]
        sqrt_area_a (torch.Tensor): sqrt of surface area of A.
        sqrt_area_b (torch.Tensor): sqrt of surface area of B.
    Returns:
        (sum, mean) of the distortion over all Va^2 vertex pairs.
    """
    d_a = dist_a / sqrt_area_a
    d_b_mapped = dist_b[p2p][:, p2p] / sqrt_area_b
    diff = (d_a - d_b_mapped).abs()
    return diff.sum(), diff.mean()


@METRIC_REGISTRY.register()
def dirichlet_energy(p2p, L_a, verts_b, sqrt_area_b):
    """
    Cotangent Dirichlet energy E = (1/2) f^T L_A f of the mapping
    f(v) = verts_B[p2p(v)] / sqrt(area_B), using the cached cotangent stiffness
    matrix L_A of mesh A. Lower is smoother / more locally isometric.

    Args:
        p2p (torch.Tensor): point-to-point map A -> B. shape [Va]
        L_a (torch.Tensor): sparse cotangent stiffness matrix of A. shape [Va, Va]
        verts_b (torch.Tensor): vertices of B. shape [Vb, 3]
        sqrt_area_b (torch.Tensor): sqrt of surface area of B.
    Returns:
        energy (torch.Tensor): scalar Dirichlet energy, summed over coordinates.
    """
    f = verts_b[p2p] / sqrt_area_b
    return 0.5 * (f * torch.sparse.mm(L_a, f)).sum()
