"""The engine's euclidean distance as a UMAP metric, including its missing-value rule.

On the standardized z-matrix (mean 0, variance 1 per stat), for each weighted stat j:
- both seasons have it:   w_j * (x_j - y_j)^2
- only one has it (v):    w_j * (v^2 + 1)  -- the expected distance to an unknown typical
                                              value, the same "typical" rule the engine uses
- neither has it:         w_j * 2
Nothing is filled in. UMAP rejects NaN, so missing values are passed as MISSING, a sentinel
far outside any z-score.
"""

from __future__ import annotations

import numba
import numpy as np

MISSING = -1e30
_IS_MISSING = -1e29  # anything below this is the sentinel


def encode_missing(x: np.ndarray) -> np.ndarray:
    """Replace NaN with the MISSING sentinel (UMAP input must be finite)."""
    return np.where(np.isnan(x), MISSING, x)


@numba.njit(fastmath=True)
def typical_distance(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    d = 0.0
    for j in range(x.shape[0]):
        if w[j] == 0.0:
            continue
        a = x[j]
        b = y[j]
        a_missing = a < _IS_MISSING
        b_missing = b < _IS_MISSING
        if not a_missing and not b_missing:
            d += w[j] * (a - b) ** 2
        elif a_missing and b_missing:
            d += w[j] * 2.0
        elif a_missing:
            d += w[j] * (b * b + 1.0)
        else:
            d += w[j] * (a * a + 1.0)
    return np.sqrt(d)


def typical_distance_reference(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Plain numpy version (NaN = missing), used to test the numba metric."""
    both = ~np.isnan(x) & ~np.isnan(y)
    only_x = ~np.isnan(x) & np.isnan(y)
    only_y = np.isnan(x) & ~np.isnan(y)
    neither = np.isnan(x) & np.isnan(y)
    d = (
        np.sum(w[both] * (x[both] - y[both]) ** 2)
        + np.sum(w[only_x] * (x[only_x] ** 2 + 1))
        + np.sum(w[only_y] * (y[only_y] ** 2 + 1))
        + np.sum(w[neither] * 2)
    )
    return float(np.sqrt(d))
