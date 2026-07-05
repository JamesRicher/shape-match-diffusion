import numpy as np
import matplotlib.pyplot as plt
from utils.registry import METRIC_REGISTRY

# np.trapz was renamed to np.trapezoid in NumPy 2.0.
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

@METRIC_REGISTRY.register()
def calculate_geodesic_error(dist_x, corr_x, corr_y, p2p, return_mean=True):
    """
    Calculate the geodesic error between predicted correspondence and gt correspondence
    NOTE: the errors returned are not normalised by total SA, this is done at
        the call site if needed, though ULRSSM preprocessing makes this a no-op

    Args:
        dist_x (np.ndarray): Geodesic distance matrix of shape x. shape [Vx, Vx]
        corr_x (np.ndarray): Ground truth correspondences of shape x. shape [V]
        corr_y (np.ndarray): Ground truth correspondences of shape y. shape [V]
        p2p (np.ndarray): Point-to-point map (shape y -> shape x). shape [Vy]
        return_mean (bool, optional): Average the geodesic error. Default True.
    Returns:
        avg_geodesic_error (np.ndarray): Average geodesic error.
    """
    ind21 = np.stack([corr_x, p2p[corr_y]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[dist_x.shape[0], dist_x.shape[0]])
    geo_err = np.take(dist_x, ind21)
    if return_mean:
        return geo_err.mean()
    else:
        return geo_err


def _pck_curve(geo_err, thresholds):
    """PCK values (fraction of correspondences below each threshold) and normalized AUC.

    Args:
        geo_err (np.ndarray): geodesic error list (any shape, flattened).
        thresholds (np.ndarray): monotone thresholds, shape [steps].
    Returns:
        pcks (np.ndarray): shape [steps]. auc (float): AUC over the unit interval.
    """
    geo_err = np.ravel(geo_err)
    pcks = (geo_err[None, :] <= thresholds[:, None]).mean(axis=1)
    # AUC over a normalized [0, 1] x-axis so it is comparable across thresholds.
    auc = _trapezoid(pcks, np.linspace(0., 1., thresholds.shape[0]))
    return pcks, auc


@METRIC_REGISTRY.register()
def plot_pck(geo_err, threshold=0.10, steps=40, label=None, ax=None):
    """
    plot pck curve and compute auc.
    Args:
        geo_err (np.ndarray): geodesic error list.
        threshold (float, optional): threshold upper bound. Default 0.10.
        steps (int, optional): number of steps between [0, threshold]. Default 40.
        label (str, optional): legend label for the curve (e.g. an experiment name).
        ax (matplotlib.axes.Axes, optional): axis to draw on. A new figure is created
            when omitted; pass a shared axis to overlay several experiments.
    Returns:
        auc (float): area under curve.
        fig (matplotlib.pyplot.figure): pck curve.
        pcks (np.ndarray): pcks.
    """
    assert threshold > 0 and steps > 0
    thresholds = np.linspace(0., threshold, steps)
    pcks, auc = _pck_curve(geo_err, thresholds)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure

    curve_label = f"{label} (AUC={auc:.3f})" if label else f"AUC={auc:.3f}"
    ax.plot(thresholds, pcks, label=curve_label)
    ax.set_xlim(0., threshold)
    ax.set_ylim(0., 1.)
    ax.set_xlabel("geodesic error")
    ax.set_ylabel("PCK")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return auc, fig, pcks


@METRIC_REGISTRY.register()
def plot_pck_multi(experiments, threshold=0.10, steps=40, title=None):
    """
    Overlay PCK curves for several experiments on a single axis, each annotated
    with its AUC. Built on top of ``plot_pck`` by sharing one axis.

    Args:
        experiments (dict[str, np.ndarray]): maps experiment label -> geodesic error
            array. Insertion order is preserved in the legend.
        threshold (float, optional): threshold upper bound. Default 0.10.
        steps (int, optional): number of steps between [0, threshold]. Default 40.
        title (str, optional): figure title.
    Returns:
        aucs (dict[str, float]): label -> auc.
        fig (matplotlib.pyplot.figure): overlaid pck curves.
        pcks (dict[str, np.ndarray]): label -> pcks.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    aucs, pcks = {}, {}
    for label, geo_err in experiments.items():
        auc, _, pck = plot_pck(geo_err, threshold=threshold, steps=steps,
                               label=label, ax=ax)
        aucs[label] = auc
        pcks[label] = pck

    if title:
        ax.set_title(title)
    return aucs, fig, pcks
