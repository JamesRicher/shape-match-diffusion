"""Shared ZoomOut spectral refinement (Melzi et al. 2019) and the ZoomOut densifier.

`zoomout_refine` upsamples/cleans a dense Y->X vertex map by alternating between the pointwise
map and a functional map while growing the spectral band k_start -> k_final. Low-k stages
error-correct the seed (low-rank averaging over the eigenbasis); larger k restores detail. It is
used two ways:

  * ZoomOutDensifier -- seeds from a nearest-anchor (geodesic Voronoi) bootstrap of the sparse map.
  * FunctionalMapDensifier (zoomout_refine: true) -- seeds from its descriptor+landmark p2p readout,
    a far better starting point than the Voronoi bootstrap.

FM convention (matches utils/texture_util): for a Y->X map p2p, Cxy = (evecs_y.T * mass_y) @ evecs_x[p2p].
"""
import torch

from utils.registry import DENSIFIER_REGISTRY
from .base_densifier import BaseDensifier


def zoomout_refine(p2p, evecs_x, evecs_y, mass_y, k_start, k_step, k_final, chunk=512):
    """Iterative spectral (ZoomOut) refinement of a dense Y->X vertex map.

    Args:
        p2p:      (N_y,) LongTensor seed map, Y vertex -> X vertex. Not modified.
        evecs_x:  (N_x, K) X eigenfunctions; evecs_y: (N_y, K) Y eigenfunctions.
        mass_y:   (N_y,) Y lumped mass (for evecs_trans_y = evecs_y.T * mass_y).
        k_start, k_step, k_final: spectral band swept low -> high (k_final capped at K).
        chunk:    Y rows per NN block (memory bound).
    Returns:
        (N_y,) LongTensor refined map. FM convention Cxy = (evecs_y.T * mass_y) @ evecs_x[p2p].
    """
    dev = evecs_y.device
    p2p = p2p.to(dev).long()
    ety = evecs_y.t() * mass_y.to(dev).unsqueeze(0)          # (K, N_y) = evecs_trans_y
    k_final = min(k_final, evecs_x.shape[1])
    if k_start > k_final:
        return p2p
    ks = list(range(k_start, k_final + 1, k_step))
    if ks[-1] != k_final:                                    # always hit k_final exactly
        ks.append(k_final)

    for k in ks:
        Cxy = ety[:k] @ evecs_x[p2p][:, :k]                 # (k, k)  p2p -> functional map
        query = evecs_y[:, :k] @ Cxy                        # (N_y, k) target embedding
        nxt = torch.empty_like(p2p)                         # C -> p2p (NN into X, chunked over Y)
        for lo in range(0, query.shape[0], chunk):
            sl = slice(lo, min(lo + chunk, query.shape[0]))
            nxt[sl] = torch.cdist(query[sl], evecs_x[:, :k]).argmin(dim=1)
        p2p = nxt
    return p2p


@DENSIFIER_REGISTRY.register()
class ZoomOutDensifier(BaseDensifier):
    """Spectral upsampling of the sparse map from a nearest-anchor bootstrap; needs eigenbases
    (ctx.evecs_*, ctx.mass_y) and ctx.dist_y. densify() returns a full-resolution Y->X vertex map."""

    def __init__(self, opt=None):
        super().__init__(opt)
        o = self.opt
        self.k_start = o.get('k_start', 20)     # initial spectral band
        self.k_step = o.get('k_step', 10)       # band growth per iteration
        self.k_final = o.get('k_final', 100)    # final band (capped at #evecs available)
        self.chunk = o.get('chunk', 512)        # Y rows per NN block

    def densify(self, sparse_p2p, ctx):
        assert ctx.evecs_x is not None and ctx.evecs_y is not None and ctx.mass_y is not None, \
            "ZoomOutDensifier needs eigenbases: set ret_evecs=True and num_evecs>=k_final"
        assert ctx.dist_y is not None, "ZoomOutDensifier needs ctx.dist_y for the bootstrap"
        dev = ctx.evecs_y.device
        idx_x, idx_y = ctx.idx_x.to(dev), ctx.idx_y.to(dev)
        # bootstrap: nearest-anchor (geodesic Voronoi) dense map to seed the first FM
        nearest = ctx.dist_y.to(dev)[:, idx_y].argmin(dim=1)         # (N_y,) closest anchor
        p2p = idx_x[sparse_p2p.to(dev)[nearest]]                     # (N_y,) Y vertex -> X vertex
        return zoomout_refine(p2p, ctx.evecs_x.to(dev), ctx.evecs_y.to(dev), ctx.mass_y,
                              self.k_start, self.k_step, self.k_final, chunk=self.chunk)
