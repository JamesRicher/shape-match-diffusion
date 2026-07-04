from utils.registry import METRIC_REGISTRY

import metrics.geo_metric
import metrics.map_metric

def build_metric(opt):
    """
    Builds a metric from options

    Args:
        opt (dict):  Must contian:
            type (str): Metric type
    """

    opt_type = opt.pop("type")
    metric = METRIC_REGISTRY.get(opt_type)
    return metric