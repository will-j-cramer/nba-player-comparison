from __future__ import annotations

import numpy as np

from nba_similarity.mapping.metric import (
    MISSING,
    encode_missing,
    typical_distance,
)


def typical_distance_reference(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Plain numpy version of the metric (NaN = missing)."""
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


def _numba(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return typical_distance(encode_missing(x), encode_missing(y), w)


def test_matches_numpy_reference_with_gaps() -> None:
    rng = np.random.default_rng(0)
    w = rng.uniform(0, 1, 12)
    for _ in range(50):
        x, y = rng.normal(size=12), rng.normal(size=12)
        x[rng.random(12) < 0.2] = np.nan
        y[rng.random(12) < 0.2] = np.nan
        assert np.isclose(_numba(x, y, w), typical_distance_reference(x, y, w))


def test_symmetric_and_zero_on_identical_complete_rows() -> None:
    x = np.array([0.5, -1.0, 2.0])
    y = np.array([1.0, np.nan, 0.0])
    w = np.ones(3)
    assert _numba(x, x, w) == 0.0
    assert np.isclose(_numba(x, y, w), _numba(y, x, w))


def test_missing_stat_counts_as_typical_not_zero() -> None:
    w = np.array([1.0, 1.0])
    x = np.array([2.0, 0.0])
    y = np.array([np.nan, 0.0])
    # expected squared distance to an unknown typical value: z^2 + 1, not (2 - 0)^2 = 4
    assert np.isclose(_numba(x, y, w), np.sqrt(2.0**2 + 1))
    assert np.isclose(_numba(y, y, w), np.sqrt(2.0))  # both missing: 2


def test_zero_weight_stats_are_ignored() -> None:
    w = np.array([0.0, 1.0])
    assert _numba(np.array([5.0, 1.0]), np.array([-5.0, 1.0]), w) == 0.0
    assert _numba(np.array([np.nan, 1.0]), np.array([3.0, 1.0]), w) == 0.0


def test_encode_missing_uses_sentinel() -> None:
    assert encode_missing(np.array([1.0, np.nan]))[1] == MISSING
