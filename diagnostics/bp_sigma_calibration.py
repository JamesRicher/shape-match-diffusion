"""BP potential calibration -- pick sigma (and a beta scale) from DATA, not from a guess.

Prerequisite to diagnostics/bp_postprocess_sweep.py. That sweep's default grid
(sigma in 0.02..0.1) is a guess, and sigma is the scalar most able to make BP a silent
no-op: too large and every candidate pair looks equally isometric, too small and the
correct pair is truncated away with the wrong ones. This script measures the distortion
distribution the pairwise potential actually sees and reports the sigma that best
separates the ground-truth pair from its competitors.

Measured, on the message graph BP would build (geodesic kNN of the SOURCE shape):

  1. GT distortion   |d_tgt(gt(i), gt(s)) - d_src(i, s)| over kNN edges -- the isometry
     defect of the datasets themselves. sigma below this scale penalises the truth.
  2. GT pair rank    the distortion percentile of the correct candidate pair among all
     valid pairs on an edge. SIGMA-FREE: if the correct pair is not near the top, no
     sigma rescues it and the potential is uninformative here.
  3. p_gt(sigma)     softmax weight of the correct pair under log psi over a sigma grid.
     The argmax is the recommended sigma.
  4. delta load      fraction of pairs truncated at delta at the chosen sigma.
  5. beta scale      spread of one sweep's geometric evidence vs the spread of the
     unary logits, so beta puts data and geometry on the same nat scale (model mode).
  6. coverage        fraction of vertices whose true match is in the candidate set at
     all -- BP's hard ceiling; low coverage means candidate refresh, not more sweeps.

Two modes. Model-free (default) needs only the dataset: candidate sets are the true
match plus its nearest target neighbours, i.e. the near-miss confusions, so coverage is
100 percent by construction and the sigma curve is a pure geometry measurement. With
--with_model the candidate sets, coverage and logit scale are the real ones from a
trained checkpoint. Run model-free first; it is seconds per pair and does not need a GPU.

Convention matches bp_postprocess_sweep: variables on Y (source = message graph),
labels on X (target). Test-phase sparse sets correspond, so the GT map is the identity.

Usage:
    python -m diagnostics.bp_sigma_calibration -c <config.yaml> [--num_pairs 20]
    python -m diagnostics.bp_sigma_calibration -c <config.yaml> --with_model \
        [--checkpoint <pth>] [--out fig.png]
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from datasets import build_dataset
from diagnostics.bp_postprocess_sweep import _load
from evaluate import apply_override
from networks.mpnn.belief_prop import _gather_rows, build_candidate_sets, edge_distortion
from networks.mpnn.geometry import knn_from_dist
from utils.logger import get_root_logger
from utils.options import load_yaml
from utils.sinkhorn import safe_log

NEG_INF = float("-inf")


def _gt_slot(cand_idx):
    """Slot of each vertex's TRUE label in its own candidate set (GT map = identity).

    Returns slot (B, n) long and hit (B, n) bool -- False where the true match was never
    proposed, in which case slot is meaningless and the vertex is excluded downstream.
    """
    n = cand_idx.shape[1]
    truth = torch.arange(n, device=cand_idx.device).view(1, n, 1)
    match = cand_idx == truth
    return match.float().argmax(-1), match.any(-1)


def _nearmiss_candidates(D_tgt, Kc):
    """Near-miss candidates: each true match plus its Kc-1 nearest target neighbours.

    Self sits at slot 0 (zero diagonal), so the true label is always proposed. This is
    the HARDEST distractor set for a distance potential -- competitors a short geodesic
    hop from the truth, whose edge lengths barely differ -- and it needs no model. Read
    the p_gt it produces as a lower bound on discriminability.
    """
    idx = D_tgt.topk(Kc, dim=-1, largest=False).indices          # (B, n, Kc)
    return idx, torch.ones_like(idx, dtype=torch.bool)


def _random_candidates(D_tgt, Kc, seed=0):
    """Truth plus Kc-1 uniformly random targets: the EASY reference set.

    Distractors are anywhere on the shape, so their edge lengths are unrelated to the
    truth's. Brackets the near-miss lower bound from above; the real candidate sets sit
    between the two.
    """
    B, n, _ = D_tgt.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand = torch.randint(0, n, (B, n, Kc - 1), generator=g).to(D_tgt.device)
    truth = torch.arange(n, device=D_tgt.device).view(1, n, 1).expand(B, -1, -1)
    idx = torch.cat([truth, rand], dim=-1)
    from networks.mpnn.belief_prop import _dedupe_mask
    return idx, _dedupe_mask(idx)


def _valid_pair_mask(cand_mask, nbr):
    """(B, n, k, Kc_a, Kc_c) True where both the source and dest candidate are valid."""
    mask_a = _gather_rows(cand_mask.long(), nbr).bool()          # (B, n, k, Kc)
    mask_c = cand_mask.unsqueeze(2).expand(-1, -1, nbr.shape[-1], -1)
    return mask_a.unsqueeze(-1) & mask_c.unsqueeze(-2)


def _pair_index(diff, gt_a, gt_c, nbr):
    """Pull the GT-pair entry out of a (B, n, k, Kc, Kc) edge tensor -> (B, n, k).

    gt_a is indexed at the edge's SOURCE neighbour, gt_c at the edge's centre vertex.
    """
    a = _gather_rows(gt_a.unsqueeze(-1), nbr).squeeze(-1)        # (B, n, k) source's slot
    c = gt_c.unsqueeze(-1).expand(-1, -1, nbr.shape[-1])         # (B, n, k) centre's slot
    out = diff.gather(3, a[..., None, None].expand(-1, -1, -1, 1, diff.shape[-1]))
    return out.squeeze(3).gather(-1, c.unsqueeze(-1)).squeeze(-1)


def _rank_percentile(diff, valid, gt_val):
    """Fraction of valid competitors on each edge STRICTLY better (less distorted) than
    the GT pair. 0 = the correct pair is the most isometric option; 0.5 = uninformative."""
    d = diff.abs()
    better = ((d < gt_val[..., None, None]) & valid).flatten(3).sum(-1).float()
    total = valid.flatten(3).sum(-1).float().clamp_min(1)
    return better / total


def _p_gt(diff, valid, gt_a, gt_c, nbr, sigma, delta):
    """Softmax weight of the GT pair among valid pairs under log psi at this sigma."""
    log_psi = -torch.clamp(diff.pow(2) / (2.0 * sigma ** 2), max=float(delta))
    log_psi = log_psi.masked_fill(~valid, NEG_INF)
    lse = torch.logsumexp(log_psi.flatten(3), dim=-1)            # (B, n, k)
    return (_pair_index(log_psi, gt_a, gt_c, nbr) - lse).exp()


def _geometry_spread(diff, valid, cand_mask, sigma, delta):
    """Spread (max - mean over a vertex's candidates) of ONE sweep's geometric evidence.

    Sweep 1 from uniform messages: msg(c) = logsumexp_a log psi(a, c), normalised, summed
    over the vertex's incoming edges. This is the nat-scale beta has to match.
    """
    log_psi = -torch.clamp(diff.pow(2) / (2.0 * sigma ** 2), max=float(delta))
    log_psi = log_psi.masked_fill(~valid, NEG_INF)
    msg = torch.logsumexp(log_psi, dim=-2)                       # (B, n, k, Kc) over senders
    msg = msg - torch.logsumexp(msg, dim=-1, keepdim=True)
    resid = msg.sum(2)                                           # (B, n, Kc)
    return _spread(resid, cand_mask)


def _spread(vals, mask):
    """max - mean over each vertex's valid candidates -> (B, n)."""
    v = vals.masked_fill(~mask, NEG_INF)
    top = v.max(-1).values
    cnt = mask.sum(-1).clamp_min(1)
    mean = (vals.masked_fill(~mask, 0.0).sum(-1) / cnt)
    return top - mean


def _q(x, qs=(50, 75, 90, 95, 99)):
    return {f"p{q}": float(np.percentile(x, q)) for q in qs}


def _pred_slot(cand_idx, pred):
    """Slot of the CURRENTLY PREDICTED label in each vertex's candidate set -> (slot, hit).

    hit is False where the argmax label was not itself proposed as a candidate (possible
    since candidates are feature-kNN union logit top-k over a subsample); those vertices
    are excluded, as there is no head-to-head to score.
    """
    match = cand_idx == pred.unsqueeze(-1)
    return match.float().argmax(-1), match.any(-1)


def _wrong_vertex_stats(cand, cmask, nbr, D_src, D_tgt, pred, sigmas, delta):
    """THE GO/NO-GO GATE: can the pairwise potential reverse the model's ACTUAL mistakes?

    The near-miss / random tables score the truth against SYNTHETIC distractors. Neither is
    the decision BP has to make, which is: at a vertex the argmax got wrong, does the
    geometry prefer the true label over the one the model actually chose? Restricted to
    wrong vertices whose truth AND prediction are both in the candidate set.

    Two families, per sigma:

    head-to-head  log psi(truth) - log psi(pred) on each incident edge, under two
                  conditions for the NEIGHBOUR's label -- 'oracle' (neighbour at its true
                  label: the evidence available once neighbours are already right) and
                  'self' (neighbour at its own predicted label: what sweep 1 actually sees
                  from the current map). Reported as edge-level and vertex-level (summed
                  over the vertex's k edges) win rates -- vertex-level is the decision.

    message       one BP message from uniform priors, m(c) = logsumexp_a log psi(a, c),
                  summed over the vertex's k incident edges, i.e. the ACCUMULATED geometric
                  evidence over its candidate set. p_true is the softmax weight of the true
                  label and rank is the fraction of candidates scoring above it. This is
                  the one that matters: a single edge only pins a candidate to a geodesic
                  circle, so weak per-edge evidence can still localise once k edges are
                  intersected. lift is over 1/Kc (a label choice), NOT the 1/Kc^2 of the
                  pair tables -- the two are not comparable.
    """
    n = cand.shape[1]
    truth = torch.arange(n, device=cand.device)
    t_slot, t_hit = _gt_slot(cand)
    p_slot, p_hit = _pred_slot(cand, pred)
    wrong = (pred != truth) & t_hit & p_hit                      # (1, n) scorable mistakes
    diff = edge_distortion(cand, nbr, D_src, D_tgt)
    valid = _valid_pair_mask(cmask, nbr)
    # only edges whose neighbour also has both labels available
    nb_ok = _gather_rows((t_hit & p_hit).long().unsqueeze(-1), nbr).squeeze(-1).bool()
    edge_ok = wrong.unsqueeze(-1) & nb_ok                        # (1, n, k)
    out = {"n_wrong": int(wrong.sum().item()), "sigma": {}}
    if out["n_wrong"] == 0:
        return out

    for s in sigmas:
        log_psi = -torch.clamp(diff.pow(2) / (2.0 * s ** 2), max=float(delta))
        log_psi = log_psi.masked_fill(~valid, NEG_INF)
        d = {}
        for name, a_slot in (("oracle", t_slot), ("self", p_slot)):
            gap = (_pair_index(log_psi, a_slot, t_slot, nbr)
                   - _pair_index(log_psi, a_slot, p_slot, nbr))  # (1, n, k) >0 favours truth
            gap = torch.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)
            e = gap[edge_ok]
            v = (gap * edge_ok).sum(-1)[wrong]                   # summed over incident edges
            d[f"edge_win_{name}"] = float((e > 0).float().mean().item()) if e.numel() else float("nan")
            d[f"vertex_win_{name}"] = float((v > 0).float().mean().item())
        # accumulated single-sweep message from uniform priors
        msg = torch.logsumexp(log_psi, dim=-2)                   # (1, n, k, Kc) over senders
        msg = msg - torch.logsumexp(msg, dim=-1, keepdim=True)
        M = (msg * edge_ok.unsqueeze(-1)).sum(2)                 # (1, n, Kc) over edges
        M = M.masked_fill(~cmask, NEG_INF)
        logp = M - torch.logsumexp(M, dim=-1, keepdim=True)
        p_true = logp.gather(-1, t_slot.unsqueeze(-1)).squeeze(-1).exp()[wrong]
        m_true = M.gather(-1, t_slot.unsqueeze(-1))
        rank = ((M > m_true) & cmask).sum(-1).float() / cmask.sum(-1).clamp_min(1).float()
        d["msg_p_true"] = float(p_true.mean().item())
        d["msg_chance"] = float((1.0 / cmask.sum(-1).clamp_min(1).float())[wrong].mean().item())
        d["msg_lift"] = d["msg_p_true"] / max(d["msg_chance"], 1e-9)
        d["msg_rank"] = float(rank[wrong].mean().item())
        d["msg_top1"] = float((rank[wrong] == 0).float().mean().item())
        out["sigma"][s] = d
    return out


def _report_gate(out, sigmas, logger):
    """Print the wrong-argmax go/no-go block (no-op unless model mode produced it)."""
    if "wrong_argmax" not in out:
        return
    w = out["wrong_argmax"]
    logger.info(f"\n=== GO/NO-GO: the potential vs the model's OWN mistakes "
                f"({w['n_wrong']} wrong vertices over {w['n_pairs']} pairs) ===")
    logger.info("truth-vs-predicted head-to-head (win > 0.5 = geometry prefers the truth); "
                "msg = accumulated 1-sweep evidence over the candidate set")
    logger.info(f"{'sigma':>8} {'edge_win':>9} {'vtx_win':>8} {'vtx_win_s':>10} "
                f"{'msg_lift':>9} {'msg_top1':>9} {'msg_rank':>9}")
    for s in sigmas:
        r = w["sigma"][f"{s:g}"]
        logger.info(f"{s:>8.4f} {r['edge_win_oracle']:>9.3f} {r['vertex_win_oracle']:>8.3f} "
                    f"{r['vertex_win_self']:>10.3f} {r['msg_lift']:>9.2f} "
                    f"{r['msg_top1']:>9.3f} {r['msg_rank']:>9.3f}")
    best_s = max(sigmas, key=lambda s: w["sigma"][f"{s:g}"]["vertex_win_oracle"])
    b = w["sigma"][f"{best_s:g}"]
    if b["vertex_win_oracle"] < 0.55:
        logger.info(f"NO-GO: at best (sigma={best_s:g}) geometry prefers the truth over the "
                    f"model's own choice on only {b['vertex_win_oracle']:.3f} of wrong "
                    f"vertices -- the potential cannot reverse these mistakes; do not build "
                    f"the in-loop version on this evidence.")
    else:
        logger.info(f"GO: at sigma={best_s:g} geometry prefers the truth on "
                    f"{b['vertex_win_oracle']:.3f} of wrong vertices (self-consistent "
                    f"{b['vertex_win_self']:.3f}), msg lift {b['msg_lift']:.2f}x, truth "
                    f"ranked top on {b['msg_top1']:.3f}")


def _candidate_stats(cand, cmask, nbr, D_src, D_tgt, sigmas, delta, keep, pred=None):
    """All per-pair statistics for one candidate set. Returns a dict of arrays/scalars.

    pred (B, n), the current argmax map, splits coverage into the vertices already
    matched correctly (where a proposed truth changes nothing) and the wrong ones --
    the latter is BP's actual ceiling on this pair.
    """
    slot, hit = _gt_slot(cand)
    diff = edge_distortion(cand, nbr, D_src, D_tgt)              # (1, n, k, Kc, Kc)
    valid = _valid_pair_mask(cmask, nbr)
    # edges usable for GT-pair statistics: the truth is proposed at BOTH endpoints
    edge_ok = hit.unsqueeze(-1) & _gather_rows(hit.long().unsqueeze(-1), nbr).squeeze(-1).bool()
    gt_val = _pair_index(diff, slot, slot, nbr).abs()            # (1, n, k)
    n_valid = valid.flatten(3).sum(-1).float()

    out = {"coverage": hit.float().mean().item(),
           "ranks": keep(_rank_percentile(diff, valid, gt_val)[edge_ok]),
           "argmax_acc": None if pred is None else (pred == torch.arange(
               cand.shape[1], device=cand.device)).float().mean().item(),
           "coverage_wrong": None,
           "chance": (1.0 / n_valid.clamp_min(1))[edge_ok].mean().item(),
           "p_gt": {}, "trunc": {}, "geo_spread": {}}
    if pred is not None:
        wrong = pred != torch.arange(cand.shape[1], device=cand.device)
        out["coverage_wrong"] = (hit[wrong].float().mean().item() if wrong.any() else
                                 float("nan"))
    for s in sigmas:
        out["p_gt"][s] = keep(_p_gt(diff, valid, slot, slot, nbr, s, delta)[edge_ok])
        over = ((diff.pow(2) / (2.0 * s ** 2) > delta) & valid).sum().item()
        out["trunc"][s] = over / max(valid.sum().item(), 1)
        out["geo_spread"][s] = keep(_geometry_spread(diff, valid, cmask, s, delta))
    return out


@torch.no_grad()
def run(test_set, model, num_pairs, k_graph, k_cand, sigmas, delta, max_keep=400_000,
        pair_seed=None, gate_only=False):
    logger = get_root_logger()
    N = min(num_pairs, len(test_set))
    # Sequential pairs are product(range(n), repeat=2), so the first N all share ONE source
    # shape -- sigma would be calibrated to a single geometry. Acute on heterogeneous sets
    # (DT4D inter: every pair has crypto at one end, and the first N share one crypto frame).
    if pair_seed is None:
        pairs = list(range(N))
        if N < len(test_set):
            logger.info("WARNING: sequential pairs are one source shape vs the rest; pass "
                        "--pair_seed for a representative sigma")
    else:
        pairs = sorted(np.random.default_rng(pair_seed).choice(
            len(test_set), size=N, replace=False).tolist())
    Kc = 2 * k_cand                                              # width of the real sets
    # gate_only: skip the nearmiss/random/model bracket tables entirely and compute just the
    # wrong-argmax go/no-go, which needs only the model's own candidate sets
    modes = ["model"] if gate_only else (
        (["model"] if model is not None else []) + ["nearmiss", "random"])
    acc = {m: [] for m in modes}
    gt_dist, logit_spread, wrong_acc = [], [], []
    rng = np.random.default_rng(0)

    def keep(t):
        v = t.detach().cpu().numpy().ravel()
        return v if v.size <= max_keep else rng.choice(v, max_keep, replace=False)

    for idx in tqdm(pairs, desc="BP sigma calibration"):
        data = test_set[idx]
        logits = None
        if model is not None:
            F_x, F_y, D_x, D_y, _ = model._sparse_inputs(data)
            P0 = model.sample(F_x, F_y, D_x, D_y)
            logits = safe_log(P0[0] if isinstance(P0, tuple) else P0)   # (1, n_y, n_x)
        else:
            b = lambda z: (z.unsqueeze(0) if z.dim() == 2 else z).float()
            D_x = b(data["first"]["sparse"]["dist"])
            D_y = b(data["second"]["sparse"]["dist"])
        D_src, D_tgt = D_y, D_x                                  # variables on Y, labels on X

        nbr, d_src_edge = knn_from_dist(D_src, min(k_graph, D_src.shape[-1] - 1))
        # the dataset's own isometry defect on the message edges (GT map = identity)
        gt_dist.append(keep((torch.gather(D_tgt, -1, nbr) - d_src_edge).abs()))

        for m in modes:
            pred = None
            if m == "model":
                cand, cmask = build_candidate_sets(logits, F_y, F_x, k_cand, k_cand)
                logit_spread.append(keep(_spread(torch.gather(logits, -1, cand), cmask)))
                pred = logits.argmax(-1)
                wrong_acc.append(_wrong_vertex_stats(cand, cmask, nbr, D_src, D_tgt,
                                                     pred, sigmas, delta))
            elif m == "nearmiss":
                cand, cmask = _nearmiss_candidates(D_tgt, Kc)
            else:
                cand, cmask = _random_candidates(D_tgt, Kc, seed=idx)
            if not gate_only:                                    # the three bracket tables
                acc[m].append(_candidate_stats(cand, cmask, nbr, D_src, D_tgt,
                                               sigmas, delta, keep, pred))

    cat = lambda vs: np.concatenate(vs)
    out = {"num_pairs": N, "pair_seed": pair_seed, "pairs": pairs,
           "k_graph": k_graph, "Kc": Kc, "delta": delta,
           "mode": "model" if model is not None else "model-free",
           "gt_distortion": _q(cat(gt_dist)), "candidates": {}}
    raw = {"gt_dist": cat(gt_dist), "modes": {}}
    for m in (() if gate_only else modes):
        st = acc[m]
        ranks = cat([p["ranks"] for p in st])
        d = {"coverage": float(np.mean([p["coverage"] for p in st])),
             "chance": float(np.mean([p["chance"] for p in st])),
             "gt_pair_rank_mean": float(ranks.mean()),
             "gt_pair_rank_best_frac": float((ranks == 0).mean()),
             "sigma": {}}
        if st[0]["argmax_acc"] is not None:
            d["argmax_acc"] = float(np.mean([p["argmax_acc"] for p in st]))
            d["coverage_wrong"] = float(np.nanmean([p["coverage_wrong"] for p in st]))
        for s in sigmas:
            d["sigma"][f"{s:g}"] = {
                "p_gt": float(cat([p["p_gt"][s] for p in st]).mean()),
                "trunc_frac": float(np.mean([p["trunc"][s] for p in st])),
                "geo_spread": float(np.median(cat([p["geo_spread"][s] for p in st]))),
            }
        d["sigma_argmax"] = float(max(sigmas, key=lambda s: d["sigma"][f"{s:g}"]["p_gt"]))
        out["candidates"][m] = d
        raw["modes"][m] = {"ranks": ranks,
                           "p_gt": [d["sigma"][f"{s:g}"]["p_gt"] for s in sigmas]}

    # ---- the go/no-go gate: BP vs the model's ACTUAL mistakes (model mode only) ----
    scored = [w for w in wrong_acc if w["n_wrong"] > 0]
    if scored:
        n_wrong = int(np.sum([w["n_wrong"] for w in scored]))
        keys = list(scored[0]["sigma"][sigmas[0]].keys())
        # weight each pair by its wrong-vertex count, so pairs with few mistakes don't
        # dominate the mean
        wts = np.array([w["n_wrong"] for w in scored], dtype=float)
        out["wrong_argmax"] = {"n_wrong": n_wrong, "n_pairs": len(scored), "sigma": {
            f"{s:g}": {k: float(np.average([w["sigma"][s][k] for w in scored], weights=wts))
                       for k in keys} for s in sigmas}}

    if gate_only:                                                # nothing below applies
        out["sigma_from_gt_distortion"] = out["gt_distortion"]["p90"]
        _report_gate(out, sigmas, logger)
        return out, raw

    # the primary mode decides sigma: the real candidate sets if we have them
    primary = "model" if model is not None else "nearmiss"
    prim = out["candidates"][primary]
    best = prim["sigma_argmax"]
    out["primary"] = primary
    out["sigma_recommended"] = best
    out["sigma_from_gt_distortion"] = out["gt_distortion"]["p90"]
    if logit_spread:
        ls = float(np.median(cat(logit_spread)))
        out["logit_spread"] = ls
        out["beta_recommended"] = prim["sigma"][f"{best:g}"]["geo_spread"] / max(ls, 1e-6)

    # ------------------------------- report -------------------------------- #
    logger.info(f"\nmode={out['mode']}  pairs={N}  k_graph={k_graph}  Kc={Kc}  delta={delta}")
    logger.info("GT distortion on message edges (sqrt-area units): "
                + "  ".join(f"{k}={v:.4f}" for k, v in out["gt_distortion"].items()))
    for m in modes:
        d = out["candidates"][m]
        logger.info(f"\n[{m}] coverage {d['coverage']:.3f}   chance p_gt {d['chance']:.4f}   "
                    f"GT-pair rank {d['gt_pair_rank_mean']:.3f} (0=best), most isometric on "
                    f"{d['gt_pair_rank_best_frac']:.3f} of edges")
        if "argmax_acc" in d:
            logger.info(f"      argmax acc {d['argmax_acc']:.3f} -> BP ceiling {d['coverage']:.3f} "
                        f"(truth proposed for {d['coverage_wrong']:.3f} of the WRONG vertices)")
        logger.info(f"{'sigma':>8} {'p_gt':>8} {'lift':>7} {'trunc':>7} {'geo_spread':>11}")
        for s in sigmas:
            r = d["sigma"][f"{s:g}"]
            flag = "  <--" if s == d["sigma_argmax"] else ""
            logger.info(f"{s:>8.4f} {r['p_gt']:>8.4f} {r['p_gt'] / max(d['chance'], 1e-9):>7.2f} "
                        f"{r['trunc_frac']:>7.3f} {r['geo_spread']:>11.3f}{flag}")

    _report_gate(out, sigmas, logger)

    logger.info(f"\nprimary candidate set: {primary}")
    if "beta_recommended" in out:
        logger.info(f"logit spread (median) {out['logit_spread']:.3f} nats vs geometric "
                    f"evidence {prim['sigma'][f'{best:g}']['geo_spread']:.3f} nats "
                    f"-> beta ~ {out['beta_recommended']:.3f}")
    # bracket both the discriminability argmax and the robustness choice (p90 of the
    # dataset's own distortion) -- they disagree whenever truncation is doing the work
    p90 = out["sigma_from_gt_distortion"]
    grid = sorted({round(v, 5) for v in (best / 2, best, best * 2, p90)})
    beta = out.get("beta_recommended", 1.0)
    logger.info(f"sigma from GT distortion (p90, robustness choice): {p90:.4f}")
    logger.info(f"\nseed the go/no-go sweep with:\n  --sigmas "
                + " ".join(f"{v:g}" for v in grid)
                + f" --betas {beta / 2:.3g} {beta:.3g} {beta * 2:.3g}")

    if best in (min(sigmas), max(sigmas)):
        logger.info("NOTE: the argmax sits on the edge of the sigma grid -- widen --sigmas "
                    "before trusting it.")
    if prim["sigma"][f"{best:g}"]["trunc_frac"] > 0.5:
        logger.info("NOTE: over half of all candidate pairs are truncated flat at delta at "
                    "the chosen sigma, so log psi is mostly constant; check the p90 choice too.")
    if prim["gt_pair_rank_mean"] > 0.25:
        logger.info("WARNING: the correct pair is not consistently the most isometric one; "
                    "the distance-only potential is weak here and no sigma fixes that.")
    return out, raw


def _figure(out, raw, sigmas, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(raw["gt_dist"], bins=80, range=(0, np.percentile(raw["gt_dist"], 99)),
               color="tab:blue")
    ax[0].axvline(out["sigma_recommended"], color="k", ls="--",
                  label=f"sigma*={out['sigma_recommended']:g}")
    ax[0].axvline(out["sigma_from_gt_distortion"], color="tab:red", ls=":",
                  label=f"p90={out['sigma_from_gt_distortion']:.4f}")
    ax[0].set_title("GT distortion on message edges")
    ax[0].set_xlabel("|d_tgt - d_src|"); ax[0].legend()

    for m, d in out["candidates"].items():
        ax[1].plot(sigmas, raw["modes"][m]["p_gt"], "o-", label=m)
        ax[1].axhline(d["chance"], color="grey", ls=":", lw=0.8)
        ax[2].hist(raw["modes"][m]["ranks"], bins=50, histtype="step", density=True,
                   label=f"{m} (mean {d['gt_pair_rank_mean']:.3f})")
    ax[1].axvline(out["sigma_recommended"], color="k", ls="--")
    ax[1].set_xscale("log"); ax[1].set_xlabel("sigma"); ax[1].set_ylabel("p(GT pair)")
    ax[1].set_title("discriminability vs sigma (dotted = chance)"); ax[1].legend()

    ax[2].set_title("GT pair distortion rank percentile")
    ax[2].set_xlabel("fraction of pairs more isometric than GT"); ax[2].legend()
    fig.suptitle(f"BP potential calibration [{out['mode']}, {out['num_pairs']} pairs]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    get_root_logger().info(f"figure -> {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--gate_only", action="store_true",
                   help="run ONLY the wrong-argmax go/no-go gate, skipping the nearmiss / "
                        "random / model bracket tables (implies --with_model)")
    p.add_argument("--with_model", action="store_true",
                   help="use a trained checkpoint's real candidate sets and logit scale")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num_pairs", type=int, default=20)
    p.add_argument("--pair_seed", type=int, default=None,
                   help="sample --num_pairs test pairs at random with this seed; without it "
                        "the first N are used, and those all share one source shape")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, repeatable (e.g. --set datasets.test.inter_class=false)")
    p.add_argument("--k_graph", type=int, default=8)
    p.add_argument("--k_cand", type=int, default=8,
                   help="per-source candidates (model mode uses k_logit=k_feat=k_cand)")
    p.add_argument("--sigmas", type=float, nargs="+",
                   default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2])
    p.add_argument("--delta", type=float, default=4.0)
    p.add_argument("--out", default=None, help="figure path (png); no figure if omitted")
    p.add_argument("--json", default=None, help="write the summary dict here")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.with_model or args.gate_only:
        model, test_set, ckpt = _load(args.config, args.checkpoint, args.device, args.overrides)
        get_root_logger().info(f"loaded {ckpt}")
    else:
        model = None
        opt = load_yaml(args.config)
        for spec in args.overrides:
            apply_override(opt, spec)
        test_set = build_dataset(opt["datasets"]["test"])
    summary, raw = run(test_set, model, args.num_pairs, args.k_graph, args.k_cand,
                       sorted(args.sigmas), args.delta, pair_seed=args.pair_seed,
                       gate_only=args.gate_only)
    # after run(), so a missing output dir can't throw away a completed calibration
    for path in (args.out, args.json):
        if path and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
    if args.out and not args.gate_only:                          # figure needs the bracket tables
        _figure(summary, raw, sorted(args.sigmas), args.out)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        get_root_logger().info(f"summary -> {args.json}")
