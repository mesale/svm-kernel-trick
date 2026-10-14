"""Reusable helpers for the SVM / kernel-trick notebook."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import pairwise_distances


def plot_decision_boundary(model, X, y, title="", ax=None, grid_steps=300, pad=0.5):
    """Plot a fitted SVC pipeline's decision regions, margins and support vectors.

    Parameters
    ----------
    model : fitted ``Pipeline`` whose last step is named ``"svc"`` (the scaler is
        applied inside the pipeline, so ``X`` is in the ORIGINAL feature units).
    X, y  : the training data the model was fitted on. Support vectors are looked
        up through ``svc.support_`` (row indices into the training set).

    Drawn: background = predicted class, solid line = decision function 0
    (the boundary), dashed lines = decision function -1 / +1 (the margin edges),
    black rings = support vectors. Support vectors lie ON or INSIDE the margin
    (or on the wrong side of it, when C is finite).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_steps),
                         np.linspace(y_min, y_max, grid_steps))
    grid = np.c_[xx.ravel(), yy.ravel()]

    zz = model.decision_function(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz > 0, alpha=0.18, levels=[-0.5, 0.5, 1.5],
                cmap=ListedColormap(["#3b4cc0", "#b40426"]))
    ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors="k",
               linestyles=["--", "-", "--"], linewidths=[1, 2, 1])

    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=22,
               edgecolors="white", linewidths=0.4, zorder=3)

    svc = model.named_steps["svc"] if hasattr(model, "named_steps") else model
    sv = X[svc.support_]
    ax.scatter(sv[:, 0], sv[:, 1], s=80, facecolors="none", edgecolors="k",
               linewidths=0.8, zorder=4, label=f"support vectors ({len(sv)})")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def median_pairwise_distance(X):
    """Median Euclidean distance between all distinct pairs of rows of ``X``."""
    D = pairwise_distances(np.asarray(X))
    return float(np.median(D[np.triu_indices_from(D, k=1)]))


def suggested_gamma(X):
    """Median-heuristic starting value for the RBF ``gamma``: 1 / median distance.

    Only a starting point for the grid search, not a replacement for it.
    """
    return 1.0 / median_pairwise_distance(X)
