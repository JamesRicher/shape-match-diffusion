"""Offline analysis of a selector.py run -- tune the selector with NO re-sampling.

Reads the saved (n_pairs, K) score matrices from a selector .npz and:
  1. sweeps the isometry<->coverage combination weight,
  2. sweeps K (by slicing the first K' sample columns -- free, no re-run),
  3. reports per-metric within-pair rank correlation,
all against the reference row (feature-NN / single K=1 / oracle).

Weight and K are HYPERPARAMETERS, so tune on a `--split train` npz, then apply the frozen weight
to the `--split test` npz for the reported number:

  # tune on train (find the best weight + see recovery vs K):
  python -m diagnostics.analyze_selector <..._train_K75....npz>
  # report on test with the frozen weight (and optionally a frozen K):
  python -m diagnostics.analyze_selector <..._test_K75....npz> --apply-weight 0.6 --k 32

combined score = w*z(isometry) + (1-w)*z(coverage), z-scored per pair across the K samples
(cycle is dropped by default -- it has ~zero within-pair signal; add it with --with-cycle).
"""
import argparse

import numpy as np
from scipy.stats import spearmanr


# --------------------------------------------------------------------------- #
def _load(path):
    d = np.load(path)
    if 'score_mge' not in d:
        raise SystemExit(f"{path}\n  has no raw score matrices (produced by an older selector.py). "
                         f"Re-run selector.py to get score_mge / score_isometry / score_coverage.")
    return d


def _z(a):
    """Per-pair z-score across the K-sample axis (axis 1)."""
    s = a.std(1, keepdims=True)
    return (a - a.mean(1, keepdims=True)) / np.where(s > 1e-12, s, 1.0)


def _select_mge(score, mge):
    """Per-pair MGE of the argmin-score sample."""
    return mge[np.arange(len(mge)), score.argmin(1)]


def _combined(iso, cov, cyc, w, with_cycle):
    c = w * _z(iso) + (1.0 - w) * _z(cov)
    return c + _z(cyc) if with_cycle else c


def _stats(sel, single_m, oracle_m):
    rec = (single_m - sel.mean()) / (single_m - oracle_m) if single_m > oracle_m else float('nan')
    return {'mge': float(sel.mean()), 'gross': float(np.mean(sel > 0.1)), 'recovery': float(rec)}


def _fmt(name, st):
    return f"{name:>16} {st['mge']:>10.4f} {st['gross']*100:>9.1f}% {st['recovery']*100:>11.1f}%"


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz', help='a selector.py .npz (train for tuning, test for reporting)')
    ap.add_argument('--apply-weight', type=float, default=None,
                    help='report the combined selector at this fixed iso-weight (freeze from train)')
    ap.add_argument('--k', type=int, default=None, help='use only the first K samples (freeze from train)')
    ap.add_argument('--with-cycle', action='store_true', help='include the (usually dead) cycle score')
    ap.add_argument('--grid', type=int, default=21, help='iso-weight grid points in [0,1]')
    args = ap.parse_args()

    d = _load(args.npz)
    mge = d['score_mge']; iso = d['score_isometry']; cov = d['score_coverage']; cyc = d['score_cycle']
    if args.k:                                              # freeze K by slicing sample columns
        mge, iso, cov, cyc = (a[:, :args.k] for a in (mge, iso, cov, cyc))
    P, K = mge.shape
    single = mge[:, 0]; single_m = single.mean()
    oracle = mge.min(1); oracle_m = oracle.mean()
    nn = d['feature_nn'] if 'feature_nn' in d else None

    print(f"\nnpz: {args.npz}\n  pairs={P}  K={K}" + (f"  (sliced to K={args.k})" if args.k else ""))
    print(f"\n{'method':>16} {'dense MGE':>10} {'gross>0.1':>10} {'oracle-recov':>12}")
    print('-' * 52)
    if nn is not None:
        print(_fmt('feature-NN', _stats(nn, single_m, oracle_m)))
    print(_fmt('single K=1', _stats(single, single_m, oracle_m)))
    print(_fmt('oracle', _stats(oracle, single_m, oracle_m)))

    # per-metric within-pair rank correlation with true MGE (what selection can exploit)
    print("\nwithin-pair Spearman(score, MGE)  (higher = better predictor; selection needs this):")
    for nm, s in [('isometry', iso), ('coverage', cov), ('cycle', cyc)]:
        rs = [spearmanr(s[p], mge[p]).correlation for p in range(P)
              if mge[p].std() > 1e-9 and s[p].std() > 1e-9]
        print(f"    {nm:>9}: {np.mean(rs):+.3f}" if rs else f"    {nm:>9}:   n/a")

    # ---- apply a frozen weight (the reporting path) ---------------------------
    if args.apply_weight is not None:
        st = _stats(_select_mge(_combined(iso, cov, cyc, args.apply_weight, args.with_cycle), mge),
                    single_m, oracle_m)
        print(f"\nFROZEN selector  (w_iso={args.apply_weight:g}, K={K}, "
              f"{'iso+cov+cycle' if args.with_cycle else 'iso+cov'}):")
        print(_fmt('selector', st))
        return

    # ---- tuning: sweep the iso<->coverage weight ------------------------------
    print(f"\niso-weight sweep  (w=1 -> isometry only, w=0 -> coverage only):")
    print(f"{'w_iso':>16} {'dense MGE':>10} {'gross>0.1':>10} {'oracle-recov':>12}")
    print('-' * 52)
    best = None
    for w in np.linspace(0.0, 1.0, args.grid):
        st = _stats(_select_mge(_combined(iso, cov, cyc, w, args.with_cycle), mge), single_m, oracle_m)
        if best is None or st['mge'] < best[1]['mge']:
            best = (w, st)
        if np.isclose(w * (args.grid - 1) % max(1, (args.grid - 1) // 10), 0):   # ~10 rows
            print(_fmt(f'{w:.2f}', st))
    w_star, st_star = best
    print('-' * 52)
    print(f"BEST  w_iso={w_star:.3f}  -> MGE {st_star['mge']:.4f}, gross {st_star['gross']*100:.1f}%, "
          f"recovery {st_star['recovery']*100:.1f}%")

    # ---- K sweep at the best weight (free; deployment cost) -------------------
    print(f"\nK sweep at w_iso={w_star:.3f}  (selector vs oracle as K grows):")
    print(f"{'K':>16} {'sel MGE':>10} {'sel gross':>10} {'oracle MGE':>12}")
    print('-' * 52)
    for kp in sorted({1, 2, 4, 8, 16, 32, K} & set(range(1, K + 1))):
        m, i, c, y = (a[:, :kp] for a in (mge, iso, cov, cyc))
        sel = _select_mge(_combined(i, c, y, w_star, args.with_cycle), m)
        print(f"{kp:>16} {sel.mean():>10.4f} {np.mean(sel > 0.1)*100:>9.1f}% {m.min(1).mean():>12.4f}")

    print(f"\n-> freeze w_iso={w_star:.3f} (and a K from the sweep), then run this on the TEST npz "
          f"with --apply-weight {w_star:.3f} [--k <K>].")


if __name__ == '__main__':
    main()
