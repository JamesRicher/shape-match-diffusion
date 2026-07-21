import argparse

import train
import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model then evaluate it on the test set, in one process.')
    parser.add_argument('-c', '--config', required=True, help='path to a YAML config file')
    parser.add_argument('-n', '--name', default=None, help='override experiment name (subdir of experiments/)')
    parser.add_argument('-e', '--epochs', type=int, default=None, help='override number of training epochs')
    parser.add_argument('--device', default=None, help="'cuda' / 'cpu'; auto-detected when omitted")
    parser.add_argument('--resume', default=None, help='path to a checkpoint to resume training from')
    parser.add_argument('--num_workers', type=int, default=0, help='dataloader workers')
    parser.add_argument('--debug', action='store_true', help='run a couple of iterations for a quick smoke test')
    # for evaluation: default (None) evaluates the just-trained final model (final.pth)
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint to evaluate (default: the run\'s final.pth)')
    parser.add_argument('--eval_tag', default=None,
                        help='subdir of results/ for the evaluation '
                             '(default: the test dataset name); keeps evaluations on '
                             'different datasets from overwriting each other')
    # forwarded to evaluate.evaluate() for the qualitative texture-transfer figures
    parser.add_argument('--num_qual', type=int, default=10,
                        help='number of random test pairs to render texture-transfer '
                             'figures for (results/qual/); 0 disables')
    parser.add_argument('--qual_seed', type=int, default=0,
                        help='RNG seed for picking the qualitative pairs')
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. train (writes checkpoints, experiment_info.json, and the training curves
    #    loss_curve.png / val_curves.png — the per-run loss and validation history)
    train_opt = train.build_opt(args)
    train.train(train_opt, args)

    # 2. evaluate the final model on the test set (writes results/stats.json and the
    #    test PCK curve pck.png)
    eval_opt, ckpt = evaluate.build_opt(args)
    evaluate.evaluate(eval_opt, ckpt, args)


if __name__ == '__main__':
    main()
