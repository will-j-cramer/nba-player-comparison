"""k-means clusters on the 2D map, and plain-language names for them."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import KMeans


def kmeans_labels(xy: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Cluster labels 0..k-1, numbered by cluster size (0 = largest), deterministic."""
    if not 2 <= k <= len(xy):
        raise ValueError(f"k must be between 2 and the number of points ({len(xy)})")
    raw = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(xy)
    by_size = np.argsort(-np.bincount(raw, minlength=k), kind="stable")
    relabel = np.empty(k, dtype=int)
    relabel[by_size] = np.arange(k)
    return relabel[raw]


def cluster_names(
    z: np.ndarray,
    labels: np.ndarray,
    stat_labels: Sequence[str],
    include: Sequence[bool] | None = None,
    n_traits: int = 2,
) -> dict[int, str]:
    """Name each cluster by its most extreme average z-scores, e.g. "high AST%, low 3PA/100".

    `include` selects which stats may name a cluster (e.g. only the distance stats).
    """
    include_mask = np.ones(z.shape[1], bool) if include is None else np.asarray(include, bool)
    names = {}
    for c in np.unique(labels):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a stat missing for a whole cluster
            mean = np.nanmean(z[labels == c], axis=0)
        mean = np.where(include_mask, np.nan_to_num(mean), 0.0)
        top = [
            j for j in np.argsort(-np.abs(mean), kind="stable")[:n_traits] if abs(mean[j]) >= 0.25
        ]
        traits = [f"{'high' if mean[j] > 0 else 'low'} {stat_labels[j]}" for j in top]
        names[int(c)] = ", ".join(traits) if traits else "average profile"
    return names
