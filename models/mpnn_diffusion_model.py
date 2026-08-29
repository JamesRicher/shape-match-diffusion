"""MPNN diffusion matcher: the MPNN-denoiser twin of MatrixDiffusionModel.
"""
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from utils.registry import MODEL_REGISTRY
from utils.logger import get_root_logger
from utils.sinkhorn import (logit_target, gaussian_target, gaussian_target_from_dist,
                            safe_log, q_sample, cosine_alpha_bar, log_sinkhorn, row_logprob)
from densifiers import build_densifier, DensifyContext
from metrics.geo_metric import calculate_geodesic_error, plot_pck
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class MPNNDiffusionModel(BaseModel):
    """Logit-space assignment diffusion with the MPNN denoiser; see module docstring."""

    def __init__(self, opt):
        super().__init__(opt)
        cfg = opt.get('diffusion', {})
        self.eta = cfg.get('eta', 0.1)                  # logit_target label-smoothing - unused legacy option now
        self.proj_iters = cfg.get('proj_iters', 5)      # Π_S Sinkhorn iterations
        self.schedule_s = cfg.get('schedule_s', 0.008)  # cosine schedule offset
        self.logsnr_shift = cfg.get('logsnr_shift', 0.0) # uniform log-SNR shift (nats); 0 = plain cosine
        self.sample_steps = cfg.get('sample_steps', 50) # reverse steps at inference
        self.final_iters = cfg.get('final_iters', 20)   # Sinkhorn iters for the final DS snap
        self.reproject = cfg.get('reproject', False)    # DisPOSE style reprojection an additional two times per DDIM step
        self.ablate_features = cfg.get('ablate_features', False)
        self.feature_dropout = cfg.get('feature_dropout', 0.0)
        tg = cfg.get('target', {})
        self.target_type = tg.get('type', 'onehot')
        assert self.target_type in ('onehot', 'gaussian'), f'unknown target {self.target_type}'
        self.target_sigma = tg.get('sigma', 0.03)       # kernel width, sqrt-area units (~anchor spacing)
        self.target_cutoff = tg.get('cutoff', 3.0 * self.target_sigma)
        self.target_floor = tg.get('floor', 2e-4)       # tail mass past the cutoff (~eta/m's budget)
        self.target_ds = tg.get('doubly_stochastic', False)
        self.target_ds_iters = tg.get('ds_iters', 20)

        # Row-stochastic (non doubly-stochastic) variant. The flag is single-sourced on the
        # denoiser (networks.denoiser.row_stochastic), read back here so every Sinkhorn read-in
        # / readout / snap in this model switches to a one-sided row-softmax in lockstep with the
        # denoiser's internal re-gauge and warps. When set, the target must be row-stochastic too
        # (col-marginal-free), so identity-at-init and the CE stay coherent. See notes.
        self.row_stochastic = getattr(self.networks['denoiser'], 'row_stochastic', False)
        if self.row_stochastic:
            assert not self.target_ds, \
                "row_stochastic requires a row-stochastic target (target.doubly_stochastic: false)"
        # Sparse decode: hungarian = bijective assignment; argmax = per-row (non-bijective, the
        # natural decode for a row-stochastic map). Independent of the projection mode so it can
        # be A/B'd on either. Used by validate_single and trajectory_divergence.
        self.decode = cfg.get('decode', 'hungarian')
        assert self.decode in ('hungarian', 'argmax'), f"unknown decode {self.decode}"
        ts = cfg.get('t_sampler')
        self.t_sampler = None if ts is None else {
            't_min_drop': ts.get('t_min_drop', 0.35),   # band floor for feature-dropped steps
            't_min_feat': ts.get('t_min_feat', 0.6),    # band floor for features-on steps
            'uniform_frac': ts.get('uniform_frac', 0.1),
        }

        self.densifier = build_densifier(opt.get('densifier'))

        # which eval stats validation reports. Sparse (FPS-point geodesic error) is the fast
        # dev metric; dense whole-shape MGE is the reporting metric and needs a densifier.
        ev = opt.get('eval', {})
        self.report_sparse = ev.get('sparse', True)
        self.report_dense = ev.get('dense', self.densifier is not None)
        # PCK-curve reporting range.
        self.pck_max = ev.get('pck_max', 0.10)  # geodesic-error upper bound (sqrt-area units)
        self.pck_n = ev.get('pck_n', 100)       # number of thresholds in [0, pck_max]

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
    def _sparse_inputs(self, data, drop_features=False):
        """Pull the sparse tokens, add a batch dim, move to device.
        Returns F_x, F_y (B,n,d_f); D_x, D_y (B,n,n); P0 (B,n_y,n_x) or None.
        P0 is None under independent-FPS eval (no bijective sparse GT); sampling paths
        ignore it, and the training/diagnostic paths that need it always have gt_perm.
        drop_features zeros the feature block (the CFG conditioning-dropout regime); the
        per-step coin lives in feed_data so the t sampler can condition on the outcome."""
        dx, dy = data['first'], data['second']
        xs, ys = dx['sparse'], dy['sparse']
        b = lambda z: (z.unsqueeze(0) if z.dim() == 2 else z).to(self.device).float()
        D_x, D_y = b(xs['dist']), b(ys['dist'])
        if 'extractor' in self.networks:                            # learnable features
            ext = self.networks['extractor']
            if getattr(ext, 'needs_operators', False):              # DiffusionNet: reads cached ops
                F_x = ext.extract(dx, xs['idx'])                    # per FPS point (1, n, d)
                F_y = ext.extract(dy, ys['idx'])
            else:                                                   # GCN: local full-mesh patches
                F_x = ext.extract(dx['verts'], dx['dist'], xs['idx'])
                F_y = ext.extract(dy['verts'], dy['dist'], ys['idx'])
        else:                                                       # frozen .npy features
            F_x, F_y = b(xs['feat']), b(ys['feat'])
        if self.ablate_features or drop_features:                   # P_t-only ablation / CFG dropout
            F_x, F_y = torch.zeros_like(F_x), torch.zeros_like(F_y)
        gt = data.get('gt_perm')
        P0 = b(gt) if gt is not None else None
        return (F_x, F_y, D_x, D_y, P0)

    def _row_logprob(self, u):
        """Row-normalised log-probabilities of u (rows sum to 1 exactly, for CE).

        Row-stochastic mode: a single row log-softmax. DS mode: log_sinkhorn ends on a column
        pass, so its rows carry the truncation residual; a final row-normalisation makes each
        row a clean log-distribution for row-CE."""
        if self.row_stochastic:
            return row_logprob(u)
        logP = log_sinkhorn(u, n_iters=self.proj_iters)
        return logP - torch.logsumexp(logP, dim=-1, keepdim=True)

    def _read_in(self, u):
        """Projected read-in P_t (probabilities) the denoiser conditions on: row-softmax in the
        row-stochastic variant, Sinkhorn Π_S otherwise. Shared by training and sampling."""
        if self.row_stochastic:
            return row_logprob(u).exp()
        return log_sinkhorn(u, n_iters=self.proj_iters).exp()

    def _reproject(self, u):
        """Re-embed logits onto the assignment manifold in the LOG domain (for the sampler's
        reproject option): row log-softmax (row-stochastic) or Sinkhorn (DS)."""
        if self.row_stochastic:
            return row_logprob(u)
        return log_sinkhorn(u, n_iters=self.proj_iters)

    def _final_snap(self, u):
        """Converged read of the final iterate (probabilities) handed to the decoder: row-softmax
        (row-stochastic) or a longer-iteration Sinkhorn (DS)."""
        if self.row_stochastic:
            return row_logprob(u).exp()
        return log_sinkhorn(u, n_iters=self.final_iters).exp()

    def _decode(self, P0):
        """Sparse Y->X p2p (n_y,) from a soft assignment P0 (n_y, n_x). hungarian = bijective
        (Hungarian on the full matrix); argmax = per-row best match (non-bijective, the natural
        row-stochastic decode). Returns a CPU long tensor."""
        if self.decode == 'argmax':
            return P0.argmax(-1).detach().cpu().long()
        row_ind, col_ind = linear_sum_assignment(-P0.detach().cpu().numpy())
        p2p = torch.empty(P0.shape[0], dtype=torch.long)
        p2p[torch.as_tensor(row_ind)] = torch.as_tensor(col_ind)
        return p2p

    def _forward_ce(self, F_x, F_y, D_x, D_y, P0, u0, t):
        """One noised forward at time t: returns (row-CE loss, row log-probs).
        P0 here is the CE weight matrix from _target (the hard permutation for the onehot
        target, the geodesic soft target otherwise). Shared by the training step and the
        loss-vs-t diagnostic."""
        u_t = q_sample(u0, t, s=self.schedule_s, logsnr_shift=self.logsnr_shift)  # VP forward marginal
        P_t = self._read_in(u_t)                                   # read-in (row-softmax or Π_S)
        u0_hat = self.networks['denoiser'](P_t, F_x, F_y, D_x, D_y, t)
        logP = self._row_logprob(u0_hat)                           # row log-distribution
        loss = -(P0 * logP).sum(-1).mean()                         # assignment-space row-CE
        return loss, logP

    # ------------------------------------------------------------------ #
    # training step
    # ------------------------------------------------------------------ #
    def _target(self, P0, D_x, D_cross=None):
        """Training target triple (P_ce, u0, H): CE weights, clean logits, entropy floor.
        onehot: hard-P0 CE + eta-smoothed logits (original behaviour), H = 0. gaussian:
        the geodesic soft target for both, H = its entropy — CE minus H is the KL to the
        target, restoring a zero floor so losses stay comparable across targets/sigmas.

        D_cross (independent-FPS training, gaussian only): (n_y, n_x) query-image -> source-anchor
        distances. When present, the soft target is built from it directly (no P0 permutation,
        since the sparse sets do not correspond); P0 is None on these steps."""
        if self.target_type == 'gaussian':
            if D_cross is not None:
                T = gaussian_target_from_dist(D_cross, self.target_sigma, self.target_cutoff,
                                              self.target_floor, doubly_stochastic=self.target_ds,
                                              ds_iters=self.target_ds_iters)
            else:
                T = gaussian_target(P0, D_x, self.target_sigma, self.target_cutoff,
                                    self.target_floor, doubly_stochastic=self.target_ds,
                                    ds_iters=self.target_ds_iters)
            logT = safe_log(T)
            return T, logT, -(T * logT).sum(-1).mean()
        assert D_cross is None, "independent-FPS training requires the gaussian target"
        return P0, logit_target(P0, self.eta), 0.0

    def _sample_t(self, B, dropped):
        """Continuous train-time t. Uniform on [0,1] without a t_sampler; otherwise banded
        to the regime's work band (see __init__), with a uniform_frac full-range floor."""
        u = torch.rand(B, device=self.device)
        if self.t_sampler is None:
            return u
        t_min = self.t_sampler['t_min_drop' if dropped else 't_min_feat']
        floor = torch.rand(B, device=self.device) < self.t_sampler['uniform_frac']
        lo = torch.where(floor, 0.0, t_min)
        return lo + (1.0 - lo) * u

    def feed_data(self, data):
        drop = self.is_train and self.feature_dropout > 0.0 \
                and torch.rand(1).item() < self.feature_dropout    # CFG conditioning dropout
        F_x, F_y, D_x, D_y, P0 = self._sparse_inputs(data, drop_features=drop)
        D_cross = data.get('gt_cross_dist')                 # independent-FPS training step
        if D_cross is not None:
            D_cross = (D_cross.unsqueeze(0) if D_cross.dim() == 2 else D_cross).to(self.device).float()
        P_ce, u0, H = self._target(P0, D_x, D_cross)
        t = self._sample_t((D_cross if D_cross is not None else P0).shape[0], drop)
        loss, logP = self._forward_ce(F_x, F_y, D_x, D_y, P_ce, u0, t)
        # H is constant w.r.t. parameters: same gradients, logged/optimized value is the KL
        self.loss_metrics = OrderedDict(l_ce=loss - H)
        self.P0_hat = logP.exp().detach()

    # ------------------------------------------------------------------ #
    # sampling / inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, F_x, F_y, D_x, D_y, steps=None, return_trajectory=False, sample_eta=0.0):
        """DDIM (predict-x0) reverse process in logit space. Returns P0 (B, n_y, n_x),
        row-stochastic or doubly-stochastic per the projection mode.
        """
        steps = steps or self.sample_steps
        net = self.networks['denoiser']
        B, n = F_x.shape[0], F_x.shape[1]
        u = torch.randn(B, n, n, device=self.device)               # ᾱ(1)=0 prior
        ts = torch.linspace(1.0, 0.0, steps + 1, device=self.device)

        traj = []
        for i in range(steps):
            t_i, t_prev = ts[i], ts[i + 1]
            # read-in, as in training. Under reproject the step ends on _reproject, so from
            # i>0 the read-in is an idempotent no-op on an already-projected u -- exp it
            # directly and skip the redundant Sinkhorn/softmax.
            P_t = u.exp() if (self.reproject and i > 0) else self._read_in(u)
            u0_hat = net(P_t, F_x, F_y, D_x, D_y, t_i.reshape(1).expand(B))
            if self.reproject:
                u0_hat = self._reproject(u0_hat)
            if return_trajectory:                                  # cheap running snap
                traj.append(self._row_logprob(u0_hat).argmax(-1))  # (B, n_y): current match

            ab_t = cosine_alpha_bar(t_i, self.schedule_s, self.logsnr_shift)
            ab_p = cosine_alpha_bar(t_prev, self.schedule_s, self.logsnr_shift)
            eps_hat = (u - ab_t.sqrt() * u0_hat) / (1.0 - ab_t).clamp_min(1e-8).sqrt()
            # generalized DDIM: sigma=0 -> deterministic (default); sigma at eta=1 -> DDPM.
            # (1-ab_p) is 0 at the final step (t_prev=0 -> ab_p=1) so no noise is added there.
            sigma = sample_eta * ((1.0 - ab_p) / (1.0 - ab_t).clamp_min(1e-8)).sqrt() \
                    * (1.0 - ab_t / ab_p.clamp_min(1e-8)).clamp_min(0.0).sqrt()
            u = ab_p.sqrt() * u0_hat + ((1.0 - ab_p) - sigma ** 2).clamp_min(0.0).sqrt() * eps_hat
            if sample_eta > 0.0:
                u = u + sigma * torch.randn_like(u)
            if self.reproject:                                     # re-embed onto the manifold
                u = self._reproject(u)                             # = log(read(u)), no exp/log round-trip

        P0 = self._final_snap(u)                                   # converged read for the decoder
        if return_trajectory:
            return P0, torch.stack(traj, dim=1), ts[:-1]           # (B,n,n), (B,steps,n_y), (steps,)
        return P0

    @torch.no_grad()
    def validate_single(self, data):
        """Sample, then decode. Returns sparse p2p (n_y,): sparse Y-index -> sparse X-index."""
        F_x, F_y, D_x, D_y, _ = self._sparse_inputs(data)
        P0 = self.sample(F_x, F_y, D_x, D_y)[0]                    # (n_y, n_x)
        return self._decode(P0)

    @torch.no_grad()
    def _dense_gcn_feats(self, verts, dist, chunk=1024):
        """Run the GCN extractor at EVERY full-mesh vertex (a patch per vertex) to get a dense
        (N, d) feature field for the densifier's data term. Chunked over centres so the
        per-patch tensors stay small; the full (N, N) dist stays on its own device and only the
        small per-patch tensors move to the GPU. Returns (N, d) on CPU."""
        ext = self.networks['extractor']
        N = dist.shape[0]
        feats = [ext.extract(verts, dist, torch.arange(lo, min(lo + chunk, N))).squeeze(0).cpu()
                 for lo in range(0, N, chunk)]
        return torch.cat(feats, dim=0)                             # (N, d)

    def _densify_context(self, data):
        """Build a DensifyContext from a dataset item's full-mesh fields (un-batched, dim-2
        tensors under the batch_size=1 single collate). Optional fields (feats, spectral ops)
        are left None when absent, so each densifier reads only what it needs. When the
        densifier wants model features (feat_source gcn/diffnet) and a matching TRAINED extractor
        is present, the frozen .npy feat field is replaced by dense descriptors from that
        extractor (GCN: one patch per vertex; DiffusionNet: one per-vertex forward). The extractor
        is self.networks['extractor'], restored from the joint checkpoint, so these are the fully
        fine-tuned features -- not the pretrained warm start. If no matching extractor is present,
        feat_x/feat_y are cleared to None; a densifier that requires them (FunctionalMapDensifier)
        then raises rather than silently substituting a weaker signal."""
        x, y = data['first'], data['second']
        feat_x, feat_y = x.get('feat'), y.get('feat')
        want_feats = self.densifier is not None and getattr(self.densifier, 'wants_model_feats', False)
        if want_feats:
            ext = self.networks.get('extractor')
            src = self.densifier.feat_source
            is_diffnet = getattr(ext, 'needs_operators', False) if ext is not None else False
            if src == 'diffnet' and is_diffnet:                 # dense DiffusionNet per-vertex
                feat_x, feat_y = ext.extract_dense(x), ext.extract_dense(y)
            elif src == 'gcn' and ext is not None and not is_diffnet:  # dense GCN patch per vertex
                feat_x = self._dense_gcn_feats(x['verts'], x['dist'])
                feat_y = self._dense_gcn_feats(y['verts'], y['dist'])
            else:                                               # cannot produce them -> leave unset
                feat_x, feat_y = None, None                     # never leak a stray .npy feat
        return DensifyContext(
            idx_x=x['sparse']['idx'], idx_y=y['sparse']['idx'],
            n_x=x['dist'].shape[0], n_y=y['dist'].shape[0],
            dist_x=x['dist'], dist_y=y['dist'],
            feat_x=feat_x, feat_y=feat_y,
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
        P_ce, u0, H = self._target(P0, D_x)
        B = P0.shape[0]
        curve = {}
        for tv in torch.linspace(0.05, 0.95, n_bins, device=self.device):
            t = tv.reshape(1).expand(B)
            losses = [self._forward_ce(F_x, F_y, D_x, D_y, P_ce, u0, t)[0].item() - float(H)
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
            maps.append(self._decode(P0).numpy())
        maps = np.stack(maps)
        disagree = [np.mean(maps[i] != maps[j])
                    for i in range(len(maps)) for j in range(i + 1, len(maps))]
        return float(np.mean(disagree)) if disagree else 0.0

    def _pck_report(self, geo_errs, out_dir, tag=''):
        """Build a PCK curve + AUC from a flat array of per-correspondence geodesic errors.

        Writes ``pck<suffix>.png`` / ``pck<suffix>_data.npz`` to out_dir (same layout as
        run_baselines.py, so the curves overlay directly and vis/plot_pck_combined.py reads
        them). Returns a dict of scalar metrics to fold into the validation result.

        Args:
            geo_errs (np.ndarray): flat geodesic errors (sqrt-area normalised).
            out_dir (str, optional): destination for the figure/npz; skipped when None.
            tag (str): '' for the dense/reporting curve, e.g. 'sparse' for a diagnostic one.
                Non-empty tags prefix the metric keys ('sparse_auc') and file names.
        """
        thresholds = np.linspace(0., self.pck_max, self.pck_n)
        label = f'{self.opt["name"]} — {tag or "dense"}'
        auc, fig, pck = plot_pck(geo_errs, threshold=self.pck_max, steps=self.pck_n, label=label)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            suffix = f'_{tag}' if tag else ''
            fig.savefig(os.path.join(out_dir, f'pck{suffix}.png'), dpi=150, bbox_inches='tight')
            np.savez(os.path.join(out_dir, f'pck{suffix}_data.npz'),
                     thresholds=thresholds, pck=pck, errors=geo_errs)
        plt.close(fig)

        prefix = f'{tag}_' if tag else ''
        return {f'{prefix}auc': float(auc), f'{prefix}pck_max': self.pck_max,
                f'{prefix}pck_n': self.pck_n}

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
            # diagnostic PCK over the sparse FPS points ('sparse_auc'); keys/files are
            # prefixed so they never clash with the dense reporting curve below.
            result.update(self._pck_report(errs, out_dir, tag='sparse'))
            msg.append(f"sparse auc={result['sparse_auc']:.4f}")
        if self.report_dense:
            dense_errs = np.concatenate(dense_errs)
            result['dense_error'] = float(dense_errs.mean())
            msg.append(f"dense MGE={result['dense_error']:.4f}")
            # the honest whole-shape reporting curve -> pck.png / pck_data.npz + 'auc'
            result.update(self._pck_report(dense_errs, out_dir, tag=''))
            msg.append(f"dense auc={result['auc']:.4f}")
        logger.info("Dev: " + " | ".join(msg))

        # BP's own state, from the last pair sampled. beta/g are what the gates learned;
        # msg_delta is the max message change on the final sweep, i.e. whether n_sweeps
        # is set anywhere near right (large => BP has not converged and more sweeps is
        # the knob; ~0 => the trailing sweeps are idle). Endpoint-only profiling is what
        # let the previous gate collapse go unnoticed for a whole run series.
        bp = getattr(getattr(self.networks['denoiser'], 'bp', None), 'stats', None)
        if bp:
            result.update(bp)
            # strip only the single-site 'bp/' prefix: a multi-site stage namespaces per
            # trunk block ('bp1/', 'bp5/'), and that prefix is the whole point of the run.
            logger.info("BP: " + " ".join(f"{k.removeprefix('bp/')}={v:+.4f}" for k, v in bp.items()))

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
