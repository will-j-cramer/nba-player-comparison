from __future__ import annotations

import numpy as np
import pytest

from nba_similarity.mapping.embed import build_embedding


def _groups(n_per: int = 80, d: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.array([[3.0] * d, [-3.0] * d, [3.0, -3.0] * (d // 2)])
    x = np.vstack([c + rng.normal(scale=0.4, size=(n_per, d)) for c in centers])
    return x, np.repeat(np.arange(3), n_per)


@pytest.fixture(scope="module")
def embedded() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, groups = _groups()
    x[::7, 0] = np.nan  # pre-1980-style gaps are still placed
    return x, groups, build_embedding(x, np.ones(x.shape[1]), seed=1)


def test_finite_2d_coordinates(embedded: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    x, _, xy = embedded
    assert xy.shape == (len(x), 2) and np.isfinite(xy).all()


def test_similar_rows_stay_together(embedded: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    _, groups, xy = embedded
    centroids = np.array([xy[groups == g].mean(axis=0) for g in range(3)])
    nearest = np.argmin(((xy[:, None, :] - centroids[None]) ** 2).sum(-1), axis=1)
    assert (nearest == groups).mean() > 0.9


def test_deterministic_for_a_seed(embedded: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    x, _, xy = embedded
    assert np.allclose(build_embedding(x, np.ones(x.shape[1]), seed=1), xy)


def test_too_few_rows_raises() -> None:
    with pytest.raises(ValueError, match="more than 15"):
        build_embedding(np.zeros((10, 3)), np.ones(3))
