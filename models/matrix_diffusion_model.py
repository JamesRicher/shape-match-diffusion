"""Matrix diffusion matcher: DDPM in logit space over the assignment (notes/noising.md).

Wraps a MatrixDenoiser under opt['networks']['denoiser']. Training noises the clean logit
target u0 = logit_target(P0*) with the VP forward marginal q_sample, projects to a
doubly-stochastic read-in P_t = Π_S(u_t), predicts the clean logits u0_hat, and takes
assignment-space row-CE on the projected prediction. Inference runs a DDIM (predict-x0)
reverse process in logit space, then Hungarian-snaps the final DS matrix to a sparse
point-to-point map. Densification to the full mesh is deferred (steps.md Step 3): dev
evaluation is the sparse geodesic error over the FPS points.
"""
from collections import OrderedDict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from utils.registry import MODEL_REGISTRY
from utils.logger import get_root_logger
from utils.sinkhorn import logit_target, q_sample, cosine_alpha_bar, log_sinkhorn
from densifiers import build_densifier, DensifyContext
from metrics.geo_metric import calculate_geodesic_error
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class MatrixDiffusionModel(BaseModel):
    """Logit-space assignment diffusion; see module docstring."""

    def __init__(self, opt):
        super().__init__(opt)
        cfg = opt.get('diffusion', {})
        self.eta = cfg.get('eta', 0.1)                  # logit_target label-smoothing
        self.proj_iters = cfg.get('proj_iters', 5)      # Π_S Sinkhorn iterations
        self.schedule_s = cfg.get('schedule_s', 0.008)  # cosine ᾱ offset
        self.sample_steps = cfg.get('sample_steps', 50) # reverse steps at inference
        self.final_iters = cfg.get('final_iters', 20)   # Sinkhorn iters for the final DS snap
        # zero the per-point features so the ONLY cross-shape signal is P_t (the alpha*log P_t
        # skip + geodesic pull-backs). Turns the single-pair overfit into a genuine test of the
        # P_t pathway: with features present the bilinear readout solves the match from features
        # alone and loss_vs_t is flat at every t (see overfit-gate-feature-shortcut memory).
        self.ablate_features = cfg.get('ablate_features', False)

        # optional map densifier (sparse p2p -> dense whole-shape p2p). A non-learned
        # post-process kept out of the loss (steps.md Step 3); None => sparse-only.
        self.densifier = build_densifier(opt.get('densifier'))

        # which eval stats validation reports. Sparse (FPS-point geodesic error) is the fast
        # dev metric; dense whole-shape MGE is the reporting metric and needs a densifier.
        ev = opt.get('eval', {})
        self.report_sparse = ev.get('sparse', True)
        self.report_dense = ev.get('dense', self.densifier is not None)

        # optional Phase-3 diagnostics (steps.md Step 7), run at validation when enabled
        diag = opt.get('diagnostics', {})
        self.diag_loss_vs_t = diag.get('loss_vs_t', False)   # is P_t actually used?
        self.diag_divergence = diag.get('divergence', False) # do prior draws diverge?
        self.diag_bins = diag.get('bins', 10)
        self.diag_repeats = diag.get('repeats', 16)
        self.diag_samples = diag.get('samples', 8)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _sparse_inputs(self, data):
        """Pull the sparse tokens, add a batch dim, move to device.
        Returns F_x, F_y (B,n,d_f); D_x, D_y (B,n,n); P0 (B,n_y,n_x) or None.
        P0 is None under independent-FPS eval (no bijective sparse GT); sampling paths
        ignore it, and the training/diagnostic paths that need it always have gt_perm."""
        dx, dy = data['first'], data['second']
        xs, ys = dx['sparse'], dy['sparse']
        b = lambda z: (z.unsqueeze(0) if z.dim() == 2 else z).to(self.device).float()
        D_x, D_y = b(xs['dist']), b(ys['dist'])
        if 'extractor' in self.networks:                            # learnable GCN features
            ext = self.networks['extractor']                        # local full-mesh patches
            F_x = ext.extract(dx['verts'], dx['dist'], xs['idx'])   # per FPS point (1, n, d)
            F_y = ext.extract(dy['verts'], dy['dist'], ys['idx'])
        else:                                                       # frozen .npy features
            F_x, F_y = b(xs['feat']), b(ys['feat'])
        if self.ablate_features:                                    # P_t-only diagnostic
            F_x, F_y = torch.zeros_like(F_x), torch.zeros_like(F_y)
        gt = data.get('gt_perm')
        P0 = b(gt) if gt is not None else None
        return (F_x, F_y, D_x, D_y, P0)

    def _row_logprob(self, u):
        """Π_S(u) as row-normalised log-probabilities (rows sum to 1 exactly, for CE).

        log_sinkhorn ends on a column pass, so its rows carry the truncation residual; a
        final row-normalisation makes each row a clean log-distribution for row-CE."""
        logP = log_sinkhorn(u, n_iters=self.proj_iters)
        return logP - torch.logsumexp(logP, dim=-1, keepdim=True)

    def _forward_ce(self, F_x, F_y, D_x, D_y, P0, u0, t):
        """One noised forward at time t: returns (row-CE loss, row log-probs).
        Shared by the training step and the loss-vs-t diagnostic."""
        u_t = q_sample(u0, t, s=self.schedule_s)                   # VP forward marginal
        P_t = log_sinkhorn(u_t, n_iters=self.proj_iters).exp()     # Π_S read-in (DS)
        u0_hat = self.networks['denoiser'](P_t, F_x, F_y, D_x, D_y, t)
        logP = self._row_logprob(u0_hat)                           # row log-distribution
        loss = -(P0 * logP).sum(-1).mean()                         # assignment-space row-CE
        return loss, logP

    # ------------------------------------------------------------------ #
    # training step
    # ------------------------------------------------------------------ #
    def feed_data(self, data):
        F_x, F_y, D_x, D_y, P0 = self._sparse_inputs(data)
        u0 = logit_target(P0, self.eta)                            # clean logits
        t = torch.rand(P0.shape[0], device=self.device)           # continuous t ~ U[0,1]
        loss, logP = self._forward_ce(F_x, F_y, D_x, D_y, P0, u0, t)
        self.loss_metrics = OrderedDict(l_ce=loss)
        self.P0_hat = logP.exp().detach()

    # ------------------------------------------------------------------ #
    # sampling / inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, F_x, F_y, D_x, D_y, steps=None, return_trajectory=False):
        """DDIM (predict-x0) reverse process in logit space. Returns P0 (B, n_y, n_x) DS.

        The read-in projection uses tau=1 (proj_iters) to match training exactly -- the
        denoiser only ever saw temperate P_t, so annealing the read-in would be
        off-distribution. Sharpening toward a permutation comes from the reverse
        trajectory (u_t -> the sharp clean logits u0) and the final Hungarian snap, not
        from the projection temperature.

        return_trajectory: when True, also return (per-step running hard maps, per-step
        t values) for the sample-variety diagnostic (vis/sample_variety.py). Each map is
        a cheap row-argmax snap of the running predict-x0 estimate u0_hat, shape (B, n_y).
        The default (False) return is just the DS P0, so `sample(...)[0]` callers are
        unaffected; only pass True when you want the trajectory.
        """
        steps = steps or self.sample_steps
        net = self.networks['denoiser']
        B, n = F_x.shape[0], F_x.shape[1]
        u = torch.randn(B, n, n, device=self.device)               # ᾱ(1)=0 prior
        ts = torch.linspace(1.0, 0.0, steps + 1, device=self.device)

        traj = []
        for i in range(steps):
            t_i, t_prev = ts[i], ts[i + 1]
            P_t = log_sinkhorn(u, n_iters=self.proj_iters).exp()   # tau=1, as in training
            u0_hat = net(P_t, F_x, F_y, D_x, D_y, t_i.reshape(1).expand(B))
            if return_trajectory:                                  # cheap running snap
                traj.append(self._row_logprob(u0_hat).argmax(-1))  # (B, n_y): current match

            ab_t = cosine_alpha_bar(t_i, self.schedule_s)
            ab_p = cosine_alpha_bar(t_prev, self.schedule_s)
            eps_hat = (u - ab_t.sqrt() * u0_hat) / (1.0 - ab_t).clamp_min(1e-8).sqrt()
            u = ab_p.sqrt() * u0_hat + (1.0 - ab_p).clamp_min(0.0).sqrt() * eps_hat

        P0 = log_sinkhorn(u, n_iters=self.final_iters).exp()       # converged DS for Hungarian
        if return_trajectory:
            return P0, torch.stack(traj, dim=1), ts[:-1]           # (B,n,n), (B,steps,n_y), (steps,)
        return P0

    @torch.no_grad()
    def validate_single(self, data):
        """Sample, Hungarian-snap. Returns sparse p2p (n_y,): sparse Y-index -> sparse X-index."""
        F_x, F_y, D_x, D_y, _ = self._sparse_inputs(data)
        P0 = self.sample(F_x, F_y, D_x, D_y)[0]                    # (n_y, n_x)
        row_ind, col_ind = linear_sum_assignment(-P0.detach().cpu().numpy())
        p2p = torch.empty(P0.shape[0], dtype=torch.long)
        p2p[torch.as_tensor(row_ind)] = torch.as_tensor(col_ind)
        return p2p

    @staticmethod
    def _densify_context(data):
        """Build a DensifyContext from a dataset item's full-mesh fields (un-batched, dim-2
        tensors under the batch_size=1 single collate). Optional fields (feats, spectral ops)
        are left None when absent, so each densifier reads only what it needs."""
        x, y = data['first'], data['second']
        return DensifyContext(
            idx_x=x['sparse']['idx'], idx_y=y['sparse']['idx'],
            n_x=x['dist'].shape[0], n_y=y['dist'].shape[0],
            dist_x=x['dist'], dist_y=y['dist'],
            feat_x=x.get('feat'), feat_y=y.get('feat'),
            evecs_x=x.get('evecs'), evecs_y=y.get('evecs'),
            evals_x=x.get('evals'), evals_y=y.get('evals'),
            mass_x=x.get('mass'), mass_y=y.get('mass'),
        )

    @torch.no_grad()
    def densify_single(self, data):
        """Sample + Hungarian for the sparse p2p, then lift to a dense whole-shape p2p via the
        configured densifier. Returns (n_y,) full-mesh target-vertex -> source-vertex."""
        assert self.densifier is not None, "densify_single needs opt['densifier'] configured"
        sparse_p2p = self.validate_single(data)                    # (n,) sparse Y->X
        return self.densifier.densify(sparse_p2p, self._densify_context(data))

    # ------------------------------------------------------------------ #
    # Phase-3 diagnostics (steps.md Step 7)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def loss_vs_t(self, data, n_bins=10, repeats=16):
        """Row-CE as a function of diffusion time t (fixed t per bin, averaged over noise
        draws). The key P_t-dependence check (2026-07-08_inverted_conditioning.md): if the
        model uses P_t, loss falls toward small t (P_t ~ clean); a flat curve means the
        denoiser ignores P_t and the pipeline is broken. Returns {t: mean_loss}."""
        F_x, F_y, D_x, D_y, P0 = self._sparse_inputs(data)
        u0 = logit_target(P0, self.eta)
        B = P0.shape[0]
        curve = {}
        for tv in torch.linspace(0.05, 0.95, n_bins, device=self.device):
            t = tv.reshape(1).expand(B)
            losses = [self._forward_ce(F_x, F_y, D_x, D_y, P0, u0, t)[0].item()
                      for _ in range(repeats)]
            curve[round(float(tv), 3)] = float(np.mean(losses))
        return curve

    @torch.no_grad()
    def trajectory_divergence(self, data, n_samples=8):
        """Mean pairwise disagreement (fraction of points mapped differently) across
        independent prior draws. On a symmetric pose an equivariant denoiser must spread
        over the symmetry modes (property 4) -> nonzero divergence; a collapsed/P_t-
        ignoring model returns ~0. Returns a scalar in [0, 1]."""
        F_x, F_y, D_x, D_y, _ = self._sparse_inputs(data)
        maps = []
        for _ in range(n_samples):
            P0 = self.sample(F_x, F_y, D_x, D_y)[0]
            row_ind, col_ind = linear_sum_assignment(-P0.detach().cpu().numpy())
            p = np.empty(P0.shape[0], dtype=int); p[row_ind] = col_ind
            maps.append(p)
        maps = np.stack(maps)
        disagree = [np.mean(maps[i] != maps[j])
                    for i in range(len(maps)) for j in range(i + 1, len(maps))]
        return float(np.mean(disagree)) if disagree else 0.0

    # ------------------------------------------------------------------ #
    # validation (sparse dev metric and/or dense whole-shape MGE)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validation(self, dataloader, out_dir=None):
        """Report the sparse FPS-point geodesic error + accuracy (opt['eval']['sparse']) and/or
        the dense whole-shape MGE via the densifier (opt['eval']['dense']), plus optional
        Phase-3 diagnostics (loss-vs-t, trajectory divergence) on the first pair.

        Each pair is sampled once; sparse and dense share that sample (dense only adds the
        non-learned densify + geodesic lookup), so enabling both costs no extra sampling."""
        if self.report_dense and self.densifier is None:
            raise ValueError("opt['eval']['dense'] is set but no opt['densifier'] is configured")
        self.eval()
        logger = get_root_logger()
        errs, accs, dense_errs, first_data = [], [], [], None
        pbar = tqdm(dataloader, desc='diffusion eval')
        for data in pbar:
            if first_data is None:
                first_data = data
            p2p = self.validate_single(data)                       # (n,) sparse Y->X
            post = {}
            if self.report_sparse:
                D_x = data['first']['sparse']['dist']              # (n,n) geodesic on X
                n = p2p.shape[0]
                rows = torch.arange(n)
                errs.append(D_x[rows, p2p].cpu().numpy())          # true match of Y row j is X col j
                accs.append((p2p == rows).float().mean().item())
                post['err'] = float(np.concatenate(errs).mean())
                post['acc'] = float(np.mean(accs))
            if self.report_dense:
                dense_p2p = self.densifier.densify(p2p, self._densify_context(data))  # (N_y,) Y->X vert
                dist_x = data['first']['dist'].cpu().numpy()       # (N_x, N_x) area-normalised geodesic
                corr_x = data['first']['corr'].cpu().numpy()       # (T,) template -> X vertex (GT .vts)
                corr_y = data['second']['corr'].cpu().numpy()
                dense_errs.append(calculate_geodesic_error(
                    dist_x, corr_x, corr_y, dense_p2p.cpu().numpy(), return_mean=False))
                post['dense'] = float(np.concatenate(dense_errs).mean())
            pbar.set_postfix(**post)                               # running averages, not a spinner

        result = {}
        msg = []
        if self.report_sparse:
            errs = np.concatenate(errs)
            result['avg_error'] = float(errs.mean())
            result['acc'] = float(np.mean(accs))
            msg.append(f"sparse avg_error={result['avg_error']:.4f} acc={result['acc']:.3f}")
        if self.report_dense:
            result['dense_error'] = float(np.concatenate(dense_errs).mean())
            msg.append(f"dense MGE={result['dense_error']:.4f}")
        logger.info("Dev: " + " | ".join(msg))

        if first_data is not None and self.diag_loss_vs_t:
            curve = self.loss_vs_t(first_data, self.diag_bins, self.diag_repeats)
            vals = list(curve.values())
            half = len(vals) // 2
            # high-t minus low-t; positive => loss falls toward clean, i.e. P_t is used
            result['loss_t_slope'] = float(np.mean(vals[half:]) - np.mean(vals[:half]))
            for i, lv in enumerate(vals):
                result[f'loss_t_{i:02d}'] = lv
            logger.info(f"Diag loss-vs-t: slope={result['loss_t_slope']:+.4f} "
                        f"(low_t={vals[0]:.3f} high_t={vals[-1]:.3f})")

        if first_data is not None and self.diag_divergence:
            result['traj_divergence'] = self.trajectory_divergence(first_data, self.diag_samples)
            logger.info(f"Diag trajectory divergence: {result['traj_divergence']:.3f}")

        self.train()
        return result
