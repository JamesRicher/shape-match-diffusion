"""Print pgfplots \\addplot blocks for report PCK figures from eval results.

Each results/<tag>/pck_data.npz stores the raw per-correspondence geodesic errors, so the
curve is recomputed here at whatever upper bound the figure uses (0.1 or 0.2) regardless
of the pck_max the eval ran with. The AUC in the legend entry is taken over the plotted
range, normalised to the unit interval -- the convention the comparison curves use.

Example:
    python -m scripts.pck_to_tikz --max 0.2 \\
        experiments/final/faust_mpnn_512_final_cold_co/results/faust_FINAL_seed0_self \\
        experiments/final/faust_mpnn_512_final_cold_co/results/faust_FINAL_seed0_self_nobp
"""
import argparse
import os
import sys

import numpy as np

# so `python scripts/pck_to_tikz.py` works as well as `python -m scripts.pck_to_tikz`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.geo_metric import _pck_curve  # noqa: E402

DEFAULT_COLORS = ['cPLOT1', 'cPLOT2', 'cPLOT3', 'cPLOT4', 'cPLOT5', 'cPLOT6']


def load_errors(path, tag, key=None):
    """Per-correspondence geodesic errors from a results dir, a pck npz, or a benchmark npz.

    evaluate.py writes them as 'errors'; diagnostics/nn_baseline_dense.py writes one array per
    ablation arm ('nn_error', 'hungarian_error', 'diffusion_error'), so those need --key.
    """
    if os.path.isdir(path):
        suffix = f'_{tag}' if tag else ''
        path = os.path.join(path, f'pck{suffix}_data.npz')
    data = np.load(path)
    key = key or 'errors'
    if key not in data:
        raise KeyError(f'{path} has no "{key}" array (holds {list(data.keys())}); pass --key '
                       f'to pick an arm, e.g. --key hungarian_error')
    errors = data[key]
    if errors.size == 0:
        raise ValueError(f'{path}: "{key}" is empty -- that arm was skipped in the run')
    return errors


def pck_curve(errors, upper, steps):
    """PCK sampled on [0, upper] plus its AUC, normalised to the unit interval.

    Delegates to the repo's own _pck_curve so the figure, the eval stats and the comparison
    curves all share one convention -- the AUC is over the PLOTTED range, so a 0.2 panel's
    legend value is not comparable to the 0.1-range auc in stats.json.
    """
    thresholds = np.linspace(0., upper, steps)
    pck, auc = _pck_curve(errors, thresholds)
    return thresholds, pck, float(auc)


def default_label(path, tag):
    """<model> / <result-tag> from a results dir, escaped, for when --label is omitted."""
    d = os.path.normpath(path if os.path.isdir(path) else os.path.dirname(path))
    parts = d.split(os.sep)
    label = parts[-1]
    if len(parts) >= 3 and parts[-2] == 'results':
        label = f'{parts[-3]} / {label}'
    if tag:
        label = f'{label} [{tag}]'
    return label.replace('_', r'\_')   # --label is passed through verbatim instead


def addplot(thresholds, pck, color, label, auc, width_macro):
    """One \\addplot block with the curve inlined, matching the report's table style."""
    rows = ''.join(f'{repr(float(x))} {y:.6f} \\\\\n'
                   for x, y in zip(thresholds, pck))
    return (f'\\addplot [color={color}, smooth, line width={width_macro}]\n'
            f'table[row sep=crcr]{{%\n{rows}    }};\n'
            f'\\addlegendentry{{\\textcolor{{black}}{{{label}: {auc:.2f}}}}}\n')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', help='results/<tag>/ dirs (or pck_data.npz paths)')
    p.add_argument('--max', type=float, default=0.10,
                   help='upper geodesic-error bound of the figure (0.1 or 0.2)')
    p.add_argument('--steps', type=int, default=20,
                   help='number of sampled thresholds, matching the comparison curves')
    p.add_argument('--tag', default='',
                   help="'' for the dense curve, 'sparse' for pck_sparse_data.npz")
    p.add_argument('--key', action='append', default=None,
                   help='npz array to read per run, repeatable (default "errors"; use e.g. '
                        'hungarian_error for a nn_baseline_dense arm)')
    p.add_argument('--label', action='append', default=None,
                   help='legend name for the corresponding run, repeatable (default: dir name)')
    p.add_argument('--color', action='append', default=None,
                   help=f'pgfplots colour per run, repeatable (default: {DEFAULT_COLORS[0]}, ...)')
    p.add_argument('--line_width', default='\\pckLineWidth', help='line width macro or length')
    p.add_argument('-o', '--out', default=None, help='write the blocks here instead of stdout')
    args = p.parse_args()

    labels = args.label or []
    colors = args.color or []
    keys = args.key or []
    blocks = []
    for i, run in enumerate(args.runs):
        errors = load_errors(run, args.tag, keys[i] if i < len(keys) else None)
        thresholds, pck, auc = pck_curve(errors, args.max, args.steps)
        label = labels[i] if i < len(labels) else default_label(run, args.tag)
        color = colors[i] if i < len(colors) else DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        blocks.append(addplot(thresholds, pck, color, label, auc, args.line_width))
        print(f'% {label}: AUC={auc:.4f} over [0, {args.max:g}], '
              f'PCK@{args.max:g}={pck[-1]:.4f}, n={errors.size}')

    out = '\n'.join(blocks)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(out)
        print(f'% wrote {args.out}')
    else:
        print(out)


if __name__ == '__main__':
    main()
