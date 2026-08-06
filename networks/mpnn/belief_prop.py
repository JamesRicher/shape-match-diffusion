"""Belief-propagation refinement — standalone prototype (notes/BP-idea.md stages 0-6).

Loopy sum-product over a pairwise MRF with variables on ONE shape (the "source"):
each source vertex is a variable whose labels are a small per-vertex candidate set of
target vertices; unaries come from the current assignment logits; pairwise potentials
score isometric distortion (|d_X(i,k) - d_Y(j,l)|) on the source's geodesic-kNN graph.

This is the SINGLE-SIDED, gradient-safe (but usable under no_grad) refiner the note
prescribes as build step 1: run it on a trained model's final logits and sweep beta
against geodesic error for a go/no-go, before wiring anything into the denoiser. The
later additions discussed (residual unaries u - u_bp, bidirectional X+Y passes with an
agreement gate, per-block integration) build ON this core; the propagation math here is
exactly what they reuse.

Not learned: the message update, cavity exclusion (N(i)\\j), normalisation, damping,
sweep count. Seven scalars parameterise it (beta, sigma, delta, tau, alpha, s, g) —
passed explicitly so the driver can sweep them. All log-domain. Dense-batched to match
the repo (no edge lists); the reverse-edge index handles the cavity term.

Orientation: `logits` is (B, n_src, n_tgt); variables live on n_src, labels on n_tgt.
`D_x` is the source geodesic (message graph + edge lengths), `D_y` the target geodesic
(supplies the labels' metric). Slack label ∅ is internal to BP: it absorbs mass for
variables whose true match was never proposed and is DROPPED at reintegration, so it
never reaches the assignment state or the (square, dustbin-free) external Sinkhorn.

References: Yedidia-Freeman-Weiss 2003 (message form); Felzenszwalb-Huttenlocher 2006
(practical vision BP); Besse et al. 2014 (per-node candidate/particle BP); Lawler 1963
(the QAP energy).
"""
import torch

from networks.mpnn.geometry import knn_from_dist

NEG_INF = float("-inf")


def _dedupe_mask(cand: torch.Tensor) -> torch.Tensor:
    """(B, n, Kc) candidate indices -> (B, n, Kc) bool, False on duplicate slots.

    Keeps the first occurrence of each target index per row; later repeats are marked
    invalid (they become inert -inf entries downstream). O(Kc^2), Kc small.
    """
    Kc = cand.shape[-1]
    eq = cand.unsqueeze(-1) == cand.unsqueeze(-2)                 # (B, n, Kc, Kc)
    earlier = torch.tril(torch.ones(Kc, Kc, dtype=torch.bool, device=cand.device), -1)
    is_dup = (eq & earlier).any(-1)
    return ~is_dup


def build_candidate_sets(logits, feats_x, feats_y, k_logit, k_feat):
    """Stage 0: per-vertex candidate labels = top-k logits ∪ feature-kNN, deduped.

    Returns cand_idx (B, n, Kc), cand_mask (B, n, Kc) with Kc = k_logit + k_feat.
    Slack is NOT included here (handled as a separate column in the potentials).
    """
    top_l = logits.topk(min(k_logit, logits.shape[-1]), dim=-1).indices
    d2 = torch.cdist(feats_x, feats_y)
    top_f = d2.topk(min(k_feat, feats_y.shape[1]), dim=-1, largest=False).indices
    cand = torch.cat([top_l, top_f], dim=-1)                     # (B, n, Kc)
    return cand, _dedupe_mask(cand)


def _unary(logits, cand_idx, cand_mask, beta, s):
    """Stage 1: φ (B, n, Kc+1), last column = slack. Held fixed across sweeps."""
    phi_real = beta * torch.gather(logits, -1, cand_idx)         # (B, n, Kc)
    phi_real = phi_real.masked_fill(~cand_mask, NEG_INF)
    phi_slack = torch.full(phi_real.shape[:-1] + (1,), float(s), device=logits.device)
    phi = torch.cat([phi_real, phi_slack], dim=-1)
    return phi - torch.logsumexp(phi, dim=-1, keepdim=True)


def _gather_rows(cand_idx, nbr):
    """cand_idx (B, n, Kc), nbr (B, n, k) -> (B, n, k, Kc): each neighbour's candidates."""
    B, n, Kc = cand_idx.shape
    k = nbr.shape[-1]
    flat = nbr.reshape(B, n * k, 1).expand(-1, -1, Kc)
    return torch.gather(cand_idx, 1, flat).reshape(B, n, k, Kc)


def _pairwise(cand_idx, cand_mask, nbr, D_x, D_y, sigma, delta):
    """Stage 2: log ψ (B, n, k, Kc+1, Kc+1) per directed edge (source a-axis, dest c-axis).

    log ψ = -min((d_Y(a,c) - d_X_edge)^2 / 2σ^2, δ); slack row/col = 0 (compatible with
    everything); rows of invalid source cands and cols of invalid dest cands = -inf.
    """
    B, n, Kc = cand_idx.shape
    k = nbr.shape[-1]
    Kp1 = Kc + 1

    d_x_edge = torch.gather(D_x, -1, nbr)                        # (B, n, k)
    cand_s = _gather_rows(cand_idx, nbr)                        # (B, n, k, Kc) source cands
    cand_d = cand_idx.unsqueeze(2).expand(-1, -1, k, -1)        # (B, n, k, Kc) dest cands

    # d_Y(source_a, dest_c): gather rows of D_y by source cands, then cols by dest cands
    r_flat = cand_s.reshape(B, n * k * Kc)
    Dr = torch.gather(D_y, 1, r_flat.unsqueeze(-1).expand(-1, -1, D_y.shape[-1]))
    Dr = Dr.reshape(B, n, k, Kc, D_y.shape[-1])                 # (B,n,k,Kc_a, m)
    c_exp = cand_d.unsqueeze(3).expand(-1, -1, -1, Kc, -1)      # (B,n,k,Kc_a,Kc_c)
    d_y = torch.gather(Dr, -1, c_exp)                          # (B,n,k,Kc_a,Kc_c)

    diff = d_y - d_x_edge[..., None, None]
    log_psi_real = -torch.clamp(diff.pow(2) / (2.0 * sigma ** 2), max=float(delta))

    log_psi = torch.zeros(B, n, k, Kp1, Kp1, device=D_x.device)
    log_psi[..., :Kc, :Kc] = log_psi_real
    # Invalidate padded SOURCE rows only (removes them from the logsumexp over senders).
    # Dest columns are deliberately NOT masked: masking them would create all -inf
    # columns -> logsumexp over an empty set -> nan gradients. Invalid dest candidates
    # instead produce a meaningless-but-finite message that is dropped at reintegration
    # (cand_mask) and can't leak forward (their unary φ = -inf zeroes them in H).
    mask_s = _gather_rows(cand_mask.long(), nbr).bool()        # (B,n,k,Kc)
    log_psi[..., :Kc, :] = log_psi[..., :Kc, :].masked_fill(~mask_s[..., None], NEG_INF)
    return log_psi


def _reverse_index(nbr):
    """For each directed edge (i, slot) with source s=nbr[i,slot], find slot' with
    nbr[s,slot']=i. Returns rev_slot (B,n,k) and valid (B,n,k) (reverse edge exists)."""
    B, n, k = nbr.shape
    nbr_s = _gather_rows(nbr, nbr)                              # (B,n,k,k): neighbours of each source
    i_idx = torch.arange(n, device=nbr.device).view(1, n, 1, 1)
    match = nbr_s == i_idx                                      # (B,n,k,k)
    valid = match.any(-1)
    rev_slot = match.float().argmax(-1)                        # first slot' matching
    return rev_slot, valid


def _gather_msg(msg, nbr, rev_slot, valid):
    """Reverse messages: out[b,i,slot] = msg[b, nbr[i,slot], rev_slot[i,slot]] (uniform
    where no reverse edge). msg (B,n,k,Kp1) -> (B,n,k,Kp1)."""
    B, n, k, Kp1 = msg.shape
    node = nbr.reshape(B, n * k)
    m1 = torch.gather(msg, 1, node[:, :, None, None].expand(-1, -1, k, Kp1))
    m1 = m1.reshape(B, n, k, k, Kp1)                           # msg at each source, all its slots
    out = torch.gather(m1, 3, rev_slot[..., None, None].expand(-1, -1, -1, 1, Kp1)).squeeze(3)
    return out * valid[..., None]                              # zero (=uniform log) if no reverse


def _sweep(phi, log_psi, nbr, rev_slot, valid, src_mask, n_sweeps, tau, alpha):
    """Stages 3-4: T damped sum-product sweeps. Returns messages msg (B,n,k,Kp1)
    where msg[i,slot] is the message FROM nbr[i,slot] TO i, over i's label domain."""
    B, n, k, Kp1 = phi.shape[0], phi.shape[1], nbr.shape[-1], phi.shape[-1]
    msg = torch.zeros(B, n, k, Kp1, device=phi.device)
    for _ in range(n_sweeps):
        H = phi + msg.sum(2)                                   # (B,n,Kp1) incoming aggregate
        H_src = _gather_rows(H, nbr)                           # (B,n,k,Kp1) H at each source
        rev = _gather_msg(msg, nbr, rev_slot, valid)          # reverse message i->s
        # Sanitise before the subtraction so padded source entries (H_src = -inf) don't
        # form -inf - -inf = nan (torch.where still differentiates the unselected branch).
        z = torch.zeros_like(H_src)
        safe = torch.where(src_mask, H_src, z) - torch.where(src_mask, rev, z)
        h_cav = torch.where(src_mask, safe, torch.full_like(H_src, NEG_INF))
        # transport: new[.,c] = tau * logsumexp_a((h_cav[.,a] + log_psi[.,a,c]) / tau)
        new = tau * torch.logsumexp((h_cav.unsqueeze(-1) + log_psi) / tau, dim=-2)
        new = new - torch.logsumexp(new, dim=-1, keepdim=True)
        msg = alpha * msg + (1.0 - alpha) * new
    return msg


def bp_refine(logits, feats_x, feats_y, D_x, D_y, *,
              k_logit=8, k_feat=8, k_graph=8,
              beta=1.0, sigma=0.05, delta=4.0, tau=1.0, alpha=0.5, s=-4.0, g=1.0,
              n_sweeps=3, return_info=False):
    """Single-sided BP refinement of assignment logits (notes/BP-idea.md stages 0-6).

    logits (B, n_src, n_tgt); feats_* (B, n, d); D_x (B, n_src, n_src); D_y
    (B, n_tgt, n_tgt). Returns refined logits (B, n_src, n_tgt) = logits + g * Δ where
    Δ scatters the belief residual onto candidates (slack dropped). With return_info,
    also returns a dict (beliefs, slack_mass, belief_entropy) for diagnostics.
    """
    cand_idx, cand_mask = build_candidate_sets(logits, feats_x, feats_y, k_logit, k_feat)
    phi = _unary(logits, cand_idx, cand_mask, beta, s)          # (B,n,Kp1)

    nbr, _ = knn_from_dist(D_x, min(k_graph, D_x.shape[-1] - 1))
    log_psi = _pairwise(cand_idx, cand_mask, nbr, D_x, D_y, sigma, delta)
    rev_slot, valid = _reverse_index(nbr)

    # source-domain validity per edge (slack column always valid)
    Kc = cand_idx.shape[-1]
    src_ok = torch.cat([_gather_rows(cand_mask.long(), nbr).bool(),
                        torch.ones(*nbr.shape, 1, dtype=torch.bool, device=nbr.device)], dim=-1)

    msg = _sweep(phi, log_psi, nbr, rev_slot, valid, src_ok, n_sweeps, tau, alpha)

    # Stage 6: reintegrate the RESIDUAL (net geometric evidence = summed incoming
    # messages, unary EXCLUDED). Adding the full belief would double-count the unary
    # (φ = β·logit) and, being a normalised log-prob (≤ 0), would push candidates
    # below the untouched non-candidate logits. Each message is zero-logsumexp, so the
    # residual is centred evidence: it raises geometrically-supported candidates and
    # lowers unsupported ones, relative to their row. Slack column dropped.
    resid = msg.sum(2)                                          # (B,n,Kp1)
    r_real = resid[..., :Kc].masked_fill(~cand_mask, 0.0)       # 0 where invalid/dup
    # Mean-centre over each vertex's valid candidates: messages are normalised (≤ 0),
    # so an uncentred residual would down-shift every candidate relative to the
    # untouched non-candidate logits and the argmax would flee off-candidate. Centring
    # makes BP a zero-sum re-ranking AMONG candidates (raise supported, lower the rest).
    cnt = cand_mask.sum(-1, keepdim=True).clamp_min(1)
    mean = (r_real.sum(-1, keepdim=True) / cnt)
    r_real = (r_real - mean).masked_fill(~cand_mask, 0.0)
    delta_mat = torch.zeros_like(logits)
    delta_mat.scatter_add_(-1, cand_idx, r_real)
    refined = logits + g * delta_mat

    if not return_info:
        return refined
    # Stage 5: beliefs (for diagnostics only — the slack mass and entropy of P(·|i))
    belief = phi + resid
    belief = belief - torch.logsumexp(belief, dim=-1, keepdim=True)
    p = belief.exp().clamp_min(1e-12)
    return refined, {"beliefs": belief, "slack_mass": p[..., Kc],
                     "belief_entropy": -(p * p.log()).sum(-1)}


# --------------------------------------------------------------------------- #
# self-test: planted isometric problem (no checkpoint needed).
# Run: python -m networks.mpnn.belief_prop
# --------------------------------------------------------------------------- #
def _planted(B=2, n=48, d=8, seed=0, logit_scale=1.5, noise=1.0, feat_noise=1.5,
             drop_frac=0.0):
    """Isometric pair: target = source under a known permutation, same metric.

    logits are weakly correct (correct label gets +logit_scale, plus Gaussian noise) so
    the raw argmax is imperfect; geometry must clean it up. Optionally drop_frac of
    vertices have their correct label pushed out of reach (tests slack). Returns
    logits, feats_x, feats_y, D_x, D_y, perm (B,n) source->target ground truth.
    """
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(B, n, 3, generator=g)
    D_x = torch.cdist(pts, pts)
    perm = torch.stack([torch.randperm(n, generator=g) for _ in range(B)])  # source->target
    inv = torch.argsort(perm, dim=1)                                        # target->source
    # Isometric target metric: with source i ↔ target perm[i], we need
    # d_Y(t1,t2) = d_X(inv[t1], inv[t2]) so that d_Y(perm[i],perm[j]) = d_X(i,j).
    D_y = torch.gather(torch.gather(D_x, 1, inv[:, :, None].expand(-1, -1, n)),
                       2, inv[:, None, :].expand(-1, n, -1))
    # features: target t shares a latent with its true source inv[t]
    latent = torch.randn(B, n, d, generator=g)
    feats_x = latent + feat_noise * torch.randn(B, n, d, generator=g)
    feats_y = torch.gather(latent, 1, inv[:, :, None].expand(-1, -1, d)) \
        + feat_noise * torch.randn(B, n, d, generator=g)
    # weakly-correct logits toward the true target
    onehot = torch.zeros(B, n, n)
    onehot.scatter_(-1, perm[:, :, None], 1.0)
    logits = logit_scale * onehot + noise * torch.randn(B, n, n, generator=g)
    return logits, feats_x, feats_y, D_x, D_y, perm


def _acc(logits, perm):
    return (logits.argmax(-1) == perm).float().mean().item()


def _run_tests():
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f'[{"PASS" if ok else "FAIL"}] {name:46s}' + (f'  {detail}' if detail else ""))

    torch.manual_seed(0)
    logits, fx, fy, Dx, Dy, perm = _planted()
    sigma = Dx[Dx > 0].median().item() * 0.15                   # ~a fraction of typical edge len
    base = _acc(logits, perm)

    # --- BP improves accuracy over the raw logits ------------------------- #
    ref = bp_refine(logits, fx, fy, Dx, Dy, sigma=sigma, beta=1.0, n_sweeps=3, g=4.0)
    after = _acc(ref, perm)
    check("BP raises match accuracy", after > base + 0.05, f"{base:.3f} -> {after:.3f}")

    # --- finite / shape ---------------------------------------------------- #
    check("output finite + shape", torch.isfinite(ref).all() and ref.shape == logits.shape)

    # --- test-time sweep-count: more inference -> not worse (mostly rising) - #
    accs = [_acc(bp_refine(logits, fx, fy, Dx, Dy, sigma=sigma, n_sweeps=T, g=4.0), perm)
            for T in (0, 1, 2, 4, 8)]
    rising = accs[-1] >= accs[1] and accs[-1] >= base
    check("accuracy rises with sweep count", rising,
          " ".join(f"T{T}={a:.2f}" for T, a in zip((0, 1, 2, 4, 8), accs)))

    # --- identity at g=0 --------------------------------------------------- #
    r0 = bp_refine(logits, fx, fy, Dx, Dy, sigma=sigma, g=0.0)
    check("g=0 is identity", torch.allclose(r0, logits, atol=1e-6))

    # --- slack absorbs unproposed matches: with tiny candidate sets, dropped #
    #     vertices should carry high slack mass rather than false confidence   #
    _, info = bp_refine(logits, fx, fy, Dx, Dy, sigma=sigma, k_logit=3, k_feat=3,
                        return_info=True, n_sweeps=3)
    # correct label out of candidate set -> those rows lean on slack
    cand, cmask = build_candidate_sets(logits, fx, fy, 3, 3)
    hit = (cand == perm[..., None]).any(-1)                     # (B,n) true label proposed?
    miss_slack = info["slack_mass"][~hit].mean().item() if (~hit).any() else 0.0
    hit_slack = info["slack_mass"][hit].mean().item()
    check("slack higher on unproposed matches", miss_slack > hit_slack,
          f"miss={miss_slack:.3f} hit={hit_slack:.3f}")

    # --- gradient flows (autograd-safe for later in-loop use) -------------- #
    lg = logits.clone().requires_grad_(True)
    bp_refine(lg, fx, fy, Dx, Dy, sigma=sigma, g=2.0).sum().backward()
    check("gradients flow", lg.grad is not None and torch.isfinite(lg.grad).all()
          and lg.grad.abs().sum() > 0)

    passed, total = sum(results), len(results)
    print(f"\n{passed}/{total} checks passed")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    _run_tests()
