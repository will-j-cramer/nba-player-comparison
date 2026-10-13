"""Builds the 2D embedding with UMAP over the engine's distance (see metric.py)."""

from __future__ import annotations

import warnings

import numpy as np

from nba_similarity.mapping.metric import encode_missing, typical_distance

DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_SEED = 42


def build_embedding(
    x: np.ndarray,
    weights: np.ndarray,
    *,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """(n, 2) coordinates for the standardized z-matrix `x` (NaN = missing).

    Deterministic for a given seed (UMAP then runs single-threaded).
    """
    import umap  # heavy import (numba JIT); only needed when building the map

    if len(x) <= n_neighbors:
        raise ValueError(f"need more than {n_neighbors} player-seasons to build a map")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=typical_distance,
        metric_kwds={"w": np.asarray(weights, dtype=np.float64)},
        random_state=seed,
        n_components=2,
    )
    with warnings.catch_warnings():
        # "n_jobs overridden to 1 by setting random_state" etc.
        warnings.simplefilter("ignore", UserWarning)
        xy = reducer.fit_transform(encode_missing(np.asarray(x, dtype=np.float64)))
    return np.asarray(xy, dtype=float)
