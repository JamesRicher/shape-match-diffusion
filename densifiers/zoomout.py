"""ZoomOut-style spectral densifier (Melzi et al. 2019).

Upsamples the sparse map by alternating between the pointwise map and a functional map while
growing the spectral band k_start -> k_final. Starting from a nearest-anchor bootstrap it
repeatedly (i) fits a k x k functional map from the current dense p2p, (ii) reads back a
refined dense p2p by nearest-neighbour in the k-dim spectral embedding. The low-k stages
error-correct the anchors (low-rank averaging); growing k restores detail. Output is a genuine
full-resolution vertex map, not a band-limited one -- the truncation only shapes the
interpolation between anchors, which is all the sparse input constrains anyway.

FM convention matches utils/texture_util: for a Y->X map p2p, Cxy = evecs_trans_y @ evecs_x[p2p]
with evecs_trans = evecs.T * mass.
"""
import torch

from utils.registry import DENSIFIER_REGISTRY
from .base_densifier import BaseDensifier, DensifyContext


@DENSIFIER_REGISTRY.register()
class ZoomOutDensifier(BaseDensifier):
    """Spectral upsampling of the sparse map; needs eigenbases (ctx.evecs_*, ctx.mass_*)."""

    def __init__(self, opt=None):
        super().__init__(opt)
        o = self.opt
        self.k_start = o.get('k_start', 20)     # initial spectral band
        self.k_step = o.get('k_step', 10)       # band growth per iteration
        self.k_final = o.get('k_final', 100)    # final band (capped at #evecs available)

    def densify(self, sparse_p2p: torch.Tensor, ctx: DensifyContext) -> torch.Tensor:
        assert ctx.evecs_x is not None and ctx.evecs_y is not None and ctx.mass_y is not None, \
            "ZoomOutDensifier needs eigenbases: set ret_evecs=True and num_evecs>=k_final"
        assert ctx.dist_y is not None, "ZoomOutDensifier needs ctx.dist_y for the bootstrap"
        device = ctx.evecs_y.device
        evecs_x = ctx.evecs_x.to(device)                      # (N_x, K)
        evecs_y = ctx.evecs_y.to(device)                      # (N_y, K)
        ety = evecs_y.t() * ctx.mass_y.to(device).unsqueeze(0)  # (K, N_y) = evecs_trans_y
        K = evecs_x.shape[1]
        k_final = min(self.k_final, K)
        assert self.k_start <= k_final, f"k_start={self.k_start} exceeds available {k_final} evecs"

        # bootstrap: nearest-anchor (geodesic Voronoi) dense map to seed the first FM
        idx_x, idx_y = ctx.idx_x.to(device), ctx.idx_y.to(device)
        nearest = ctx.dist_y.to(device)[:, idx_y].argmin(dim=1)   # (N_y,) closest anchor
        p2p = idx_x[sparse_p2p.to(device)[nearest]]              # (N_y,) Y vertex -> X vertex

        # spectral bands to sweep (ensure k_final is hit even if not step-aligned)
        ks = list(range(self.k_start, k_final + 1, self.k_step))
        if ks[-1] != k_final:
            ks.append(k_final)

        for k in ks:
            Cxy = ety[:k] @ evecs_x[p2p][:, :k]                  # (k, k)  p2p -> FM
            query = evecs_y[:, :k] @ Cxy                         # (N_y, k) target embedding
            p2p = torch.cdist(query, evecs_x[:, :k]).argmin(dim=1)  # FM -> p2p (NN into X)
        return p2p
