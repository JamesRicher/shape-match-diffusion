import csv
import os
import time

from utils.logger import get_root_logger


class MetricLogger:
    """Logs training/validation scalars to a tidy CSV and (optionally) TensorBoard.

    The CSV (``results/metrics.csv``) is the source of truth for the rendered PNG
    curves and can be re-plotted at any time; the TensorBoard event files (under
    ``experiments/<name>/tb/``) give live/interactive monitoring. TensorBoard is
    optional: if the package is not installed, CSV logging continues and a single
    warning is emitted.

    Tags are free-form; the plotter groups by prefix, so use ``Loss/*`` for loss
    terms and ``Val/*`` for validation metrics (e.g. ``Val/avg_error``).
    """

    FIELDS = ['wall_time', 'iter', 'epoch', 'tag', 'value']

    def __init__(self, results_dir, tb_dir=None):
        os.makedirs(results_dir, exist_ok=True)
        self.csv_path = os.path.join(results_dir, 'metrics.csv')
        # truncate: one CSV per training run (fresh curves)
        self._csv_file = open(self.csv_path, 'w', newline='')
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(self.FIELDS)
        self._csv_file.flush()

        self.tb = None
        if tb_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb = SummaryWriter(tb_dir)
            except Exception as e:  # tensorboard not installed / import error
                get_root_logger().warning(
                    f'TensorBoard unavailable ({type(e).__name__}: {e}); logging CSV only. '
                    f'`pip install tensorboard` to enable it.')

    def log(self, tag, value, step, epoch=None):
        """Record one scalar at global step ``step`` (we use the iteration count)."""
        value = float(value)
        self._writer.writerow([time.time(), step, '' if epoch is None else epoch, tag, value])
        self._csv_file.flush()
        if self.tb is not None:
            self.tb.add_scalar(tag, value, step)

    def log_many(self, scalars, step, epoch=None):
        """Record a dict of ``{tag: value}`` at the same step."""
        for tag, value in scalars.items():
            self.log(tag, value, step, epoch=epoch)

    def close(self):
        self._csv_file.close()
        if self.tb is not None:
            self.tb.close()
