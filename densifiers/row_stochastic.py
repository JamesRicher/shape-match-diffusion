"""Row-stochastic densifier: soft, non-bijective lift of the sparse map.

A dense correspondence need not be a bijection (the FAUST_r .vts GT is itself many-to-one), so
this densifier drops the doubly-stochastic/column constraint the sparse diffusion uses and
keeps only a per-Y-vertex distribution over X. With row marginals alone there is nothing to
iterate -- the solve is a single temperatured softmax over a cost, and all the design lives in
building that cost so it stays anchored to the diffusion output:

    prior g(y, x)  = sum_k softmax_k(-d_geo_Y(y, anchor_k)/tau) * exp(-d_geo_X(x, match_k)^2 / 2 sigma_x^2)
    data  h(y, x)  = exp(-feat_weight * ||f_y(y) - f_x(x)||^2 / 2 sigma_f^2)
    S(y, .)        = row_normalise(g * h)      p2p(y) = argmax_x S(y, x)

The prior is a soft-Voronoi blend of the sparse matches: each Y vertex's nearest FPS anchors
vote for their matched X vertices, spread into a geodesic Gaussian on X. The data term (frozen
or GCN features) sharpens within that region -- which is what lets this beat the piecewise-
constant nearest-anchor floor. It reduces exactly to NearestAnchorDensifier at K=1, sigma_x->0,
feat_weight=0. Non-learned, out of the loss (steps.md Step 3).
"""
import torch

from utils.registry import DENSIFIER_REGISTRY
from .base_densifier import BaseDensifier, DensifyContext


@DENSIFIER_REGISTRY.register()
class RowStochasticDensifier(BaseDensifier):
    """Soft per-Y-vertex distribution over X, anchored by the sparse map; needs ctx.dist_x/dist_y
    and (optionally) ctx.feat_x/feat_y. densify() returns the hard row-argmax p2p; soft_map()
    returns the top-c row-stochastic weights for soft uses (e.g. texture transfer)."""

    def __init__(self, opt=None):
        super().__init__(opt)
        o = self.opt
        self.K = o.get('K', 6)                    # nearest FPS anchors blended per Y vertex
        self.tau_y = o.get('tau_y', None)         # Y-side blend temperature (None -> cell radius)
        self.sigma_x = o.get('sigma_x', None)     # X geodesic Gaussian width (None -> cell radius)
        self.sigma_f = o.get('sigma_f', None)     # feature bandwidth (None -> median feat dist)
        self.feat_weight = o.get('feat_weight', 1.0)  # 0 disables the data term (prior only)
        self.chunk = o.get('chunk', 512)          # Y-vertex rows scored per block (memory bound)
        self.top_c = o.get('top_c', 8)            # candidates kept per row in soft_map
        self.eps = o.get('eps', 1e-12)

    def _prepare(self, sparse_p2p, ctx):
        assert ctx.dist_x is not None and ctx.dist_y is not None, \
            "RowStochasticDensifier needs ctx.dist_x and ctx.dist_y"
        dev = ctx.dist_y.device
        dist_x, dist_y = ctx.dist_x.to(dev), ctx.dist_y
        idx_x, idx_y = ctx.idx_x.to(dev).long(), ctx.idx_y.to(dev).long()
        sparse_p2p = sparse_p2p.to(dev).long()

        d_ya = dist_y[:, idx_y]                                  # (N_y, n) Y vertex -> Y anchor
        K = min(self.K, idx_y.shape[0])
        d_k, a_idx = torch.topk(d_ya, K, dim=1, largest=False)  # (N_y, K) nearest anchors

        # length scale: median distance from a Y vertex to its nearest anchor (~cell radius).
        # Auto-adapts to n_sparse density, so tau/sigma_x need no manual retune per resolution.
        ls = d_k[:, 0].median().clamp_min(self.eps)
        tau_y = self.tau_y if self.tau_y is not None else ls
        sigma_x = self.sigma_x if self.sigma_x is not None else ls

        w = torch.softmax(-d_k / tau_y, dim=1)                  # (N_y, K) Y-side anchor weights
        m = idx_x[sparse_p2p]                                    # (n,) X vertex each anchor maps to
        Gker = torch.exp(-(dist_x[:, m] ** 2) / (2 * sigma_x ** 2 + self.eps))  # (N_x, n)

        pre = dict(dev=dev, a_idx=a_idx, w=w, m=m, Gker=Gker, fx=None, fy=None, inv_f=0.0)

        use_feat = self.feat_weight > 0 and ctx.feat_x is not None and ctx.feat_y is not None
        if use_feat:
            fx, fy = ctx.feat_x.to(dev).float(), ctx.feat_y.to(dev).float()
            if self.sigma_f is not None:
                sigma_f = torch.as_tensor(float(self.sigma_f), device=dev)
            else:  # median feature distance from a Y vertex to its nearest anchor's X match
                sigma_f = (fy - fx[m[a_idx[:, 0]]]).norm(dim=-1).median().clamp_min(self.eps)
            pre.update(fx=fx, fy=fy, inv_f=self.feat_weight / (2 * sigma_f ** 2 + self.eps))
        return pre

    def _score_chunk(self, sl, pre):
        """Unnormalised scores S = g * h for Y rows in slice sl. Returns (r, N_x)."""
        a_r, w_r = pre['a_idx'][sl], pre['w'][sl]               # (r, K), (r, K)
        Gk = pre['Gker'][:, a_r]                                # (N_x, r, K)
        g = torch.einsum('xrk,rk->rx', Gk, w_r)                # (r, N_x) soft-Voronoi prior
        if pre['fx'] is None:
            return g
        d2 = torch.cdist(pre['fy'][sl], pre['fx']) ** 2         # (r, N_x) feature distance^2
        return g * torch.exp(-pre['inv_f'] * d2)               # prior * data likelihood

    def densify(self, sparse_p2p: torch.Tensor, ctx: DensifyContext) -> torch.Tensor:
        pre = self._prepare(sparse_p2p, ctx)
        N_y = ctx.dist_y.shape[0]
        p2p = torch.empty(N_y, dtype=torch.long, device=pre['dev'])
        for lo in range(0, N_y, self.chunk):
            sl = slice(lo, min(lo + self.chunk, N_y))
            s = self._score_chunk(sl, pre)                     # (r, N_x)
            best_val, best_x = s.max(dim=1)
            # underflow guard: a row with no prior mass falls back to its nearest anchor's match,
            # so the result is never worse than NearestAnchorDensifier.
            fallback = pre['m'][pre['a_idx'][sl][:, 0]]
            p2p[sl] = torch.where(best_val > self.eps, best_x, fallback)
        return p2p

    # ------------------------------------------------------------------ #
    # soft map (top-c row-stochastic weights, for texture transfer etc.)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def soft_map(self, sparse_p2p: torch.Tensor, ctx: DensifyContext):
        """Returns (cand (N_y, c), weight (N_y, c)): the c highest-probability X vertices per Y
        vertex and their row-normalised weights (rows sum to 1). c = min(top_c, N_x)."""
        pre = self._prepare(sparse_p2p, ctx)
        N_y, N_x = ctx.dist_y.shape[0], ctx.dist_x.shape[0]
        c = min(self.top_c, N_x)
        cand = torch.empty(N_y, c, dtype=torch.long, device=pre['dev'])
        weight = torch.empty(N_y, c, device=pre['dev'])
        for lo in range(0, N_y, self.chunk):
            sl = slice(lo, min(lo + self.chunk, N_y))
            s = self._score_chunk(sl, pre)                     # (r, N_x)
            vals, idx = torch.topk(s, c, dim=1)                # (r, c)
            cand[sl] = idx
            weight[sl] = vals / vals.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return cand, weight
