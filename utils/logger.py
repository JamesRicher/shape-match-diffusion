import time
import logging


def get_root_logger(name: str = 'shapematch', log_level: int = logging.INFO) -> logging.Logger:
    """Return a process-wide logger with a single stream handler.

    Repeated calls return the same configured logger (handlers are only added once).
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(handler)
        logger.setLevel(log_level)
        logger.propagate = False
    return logger


class AvgTimer:
    """Tracks the average wall-clock time between successive ``record()`` calls."""

    def __init__(self):
        self.start = time.time()
        self.total = 0.0
        self.count = 0

    def record(self):
        now = time.time()
        self.total += now - self.start
        self.count += 1
        self.start = now

    def get_avg_time(self) -> float:
        return self.total / max(self.count, 1)
