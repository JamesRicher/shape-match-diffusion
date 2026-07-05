from functools import partial

from utils.registry import METRIC_REGISTRY

import metrics.geo_metric
import metrics.map_metric

def build_metric(opt):
    """
    Builds a metric from options.

    Args:
        opt (dict): Must contain
            type (str): Metric type.
        Any remaining keys are bound to the metric as keyword arguments (e.g.
        ``plot_pck``'s ``threshold`` / ``steps``), so config overrides actually
        take effect rather than being silently dropped.
    """
    opt = dict(opt)  # don't mutate the caller's dict
    opt_type = opt.pop("type")
    metric = METRIC_REGISTRY.get(opt_type)
    return partial(metric, **opt) if opt else metric