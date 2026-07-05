import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # headless: safe on a remote machine with no display
import matplotlib.pyplot as plt


def _read_metrics(csv_path):
    """Read results/metrics.csv into ``{tag: {'iter': [...], 'epoch': [...], 'value': [...]}}``."""
    series = defaultdict(lambda: {'iter': [], 'epoch': [], 'value': []})
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            s = series[row['tag']]
            s['iter'].append(int(row['iter']))
            s['epoch'].append(int(row['epoch']) if row['epoch'] not in ('', None) else None)
            s['value'].append(float(row['value']))
    return series


def plot_training_curves(results_dir):
    """Render loss_curve.png and val_curves.png from ``results_dir/metrics.csv``.

    Loss terms (``Loss/*``, plus ``LR``) are plotted against iteration; validation
    metrics (``Val/*``) against epoch. Only tags present in the CSV are drawn, so
    this works on a partially-complete run. Returns the list of files written.
    """
    csv_path = os.path.join(results_dir, 'metrics.csv')
    if not os.path.isfile(csv_path):
        return []
    series = _read_metrics(csv_path)
    written = []

    # --- loss curve (vs iteration), LR on a secondary axis if present ---------- #
    loss_tags = sorted(t for t in series if t.startswith('Loss/'))
    if loss_tags:
        fig, ax = plt.subplots()
        for tag in loss_tags:
            s = series[tag]
            ax.plot(s['iter'], s['value'], label=tag.split('/', 1)[1])
        ax.set_xlabel('iteration')
        ax.set_ylabel('loss')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        if 'LR' in series:
            ax2 = ax.twinx()
            ax2.plot(series['LR']['iter'], series['LR']['value'], color='gray',
                     ls='--', alpha=0.6, label='LR')
            ax2.set_ylabel('learning rate')
            ax2.legend(loc='lower left')
        ax.set_title('Training loss')
        out = os.path.join(results_dir, 'loss_curve.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        written.append(out)

    # --- validation curves (vs epoch): avg_error (left) and auc (right) -------- #
    val_tags = sorted(t for t in series if t.startswith('Val/'))
    if val_tags:
        fig, ax = plt.subplots()
        ax.set_xlabel('epoch')
        ax.grid(True, alpha=0.3)
        lines = []
        if 'Val/avg_error' in series:
            s = series['Val/avg_error']
            lines += ax.plot(s['epoch'], s['value'], color='tab:red', marker='.', label='avg_error')
            ax.set_ylabel('avg geodesic error', color='tab:red')
            ax.tick_params(axis='y', labelcolor='tab:red')
        if 'Val/auc' in series:
            s = series['Val/auc']
            ax2 = ax.twinx()
            lines += ax2.plot(s['epoch'], s['value'], color='tab:blue', marker='.', label='auc')
            ax2.set_ylabel('AUC', color='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
        # any other Val/* tags on the primary axis
        for tag in val_tags:
            if tag not in ('Val/avg_error', 'Val/auc'):
                s = series[tag]
                lines += ax.plot(s['epoch'], s['value'], marker='.', label=tag.split('/', 1)[1])
        ax.legend(lines, [ln.get_label() for ln in lines], loc='best')
        ax.set_title('Validation metrics')
        out = os.path.join(results_dir, 'val_curves.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        written.append(out)

    return written


def parse_args():
    parser = argparse.ArgumentParser(description='Render training/validation curves from metrics.csv.')
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('-c', '--config', help='config whose experiment name locates results/')
    g.add_argument('-n', '--name', help='experiment name (subdir of experiments/)')
    g.add_argument('--results-dir', help='path to a results/ dir containing metrics.csv')
    return parser.parse_args()


def _results_dir_from_args(args):
    if args.results_dir:
        return args.results_dir
    from utils.options import load_yaml, resolve_experiment_paths
    opt = {'name': args.name} if args.name else load_yaml(args.config)
    resolve_experiment_paths(opt)
    return opt['path']['results']


def main():
    args = parse_args()
    results_dir = _results_dir_from_args(args)
    written = plot_training_curves(results_dir)
    if written:
        print('wrote ' + ', '.join(written))
    else:
        print(f'no metrics.csv found in {results_dir}')


if __name__ == '__main__':
    main()
