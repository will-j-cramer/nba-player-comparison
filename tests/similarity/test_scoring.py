from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from nba_similarity.similarity.scoring import masked_scores, projected_scores
from tests.similarity.conftest import random_population

D = 12
EQUAL_W = np.full(D, 1 / D)


def stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.nanmean(X, axis=0), np.nanvar(X, axis=0)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_identical_seasons_score_100(metric: str) -> None:
    X = random_population()
    s = masked_scores(X, X[0], EQUAL_W, *stats(X), metric)  # type: ignore[arg-type]
    assert s.score[0] == pytest.approx(100)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_contributions_sum_to_score(metric: str) -> None:
    X = random_population()
    X[::7, 3] = np.nan  # some missing values
    w = np.random.default_rng(1).dirichlet(np.ones(D))
    s = masked_scores(X, X[5], w, *stats(X), metric)  # type: ignore[arg-type]
    np.testing.assert_allclose(np.nansum(s.contributions, axis=1), s.score)


def test_euclidean_zero_means_typical_distance() -> None:
    """Averaged over every player-season, the score is exactly 0 (baseline = mean distance)."""
    X = random_population()
    s = masked_scores(X, X[3] * 2.5, EQUAL_W, *stats(X), "euclidean")
    assert s.score.mean() == pytest.approx(0, abs=1e-9)


def test_cosine_matches_sklearn_with_equal_weights() -> None:
    X = random_population()
    s = masked_scores(X, X[0], EQUAL_W, *stats(X), "cosine")
    np.testing.assert_allclose(s.score, 100 * cosine_similarity(X[:1], X)[0])


def test_euclidean_ranking_matches_sklearn_distances() -> None:
    X = random_population()
    s = masked_scores(X, X[0], EQUAL_W, *stats(X), "euclidean")
    distances = euclidean_distances(X[:1], X)[0]
    np.testing.assert_array_equal(
        np.argsort(-s.score, kind="stable"), np.argsort(distances, kind="stable")
    )


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_missing_feature_is_ignored_not_zero_filled(metric: str) -> None:
    """A candidate missing a stat is scored on the shared stats only."""
    X = random_population()
    query, cand = X[0].copy(), X[1].copy()
    cand[2] = np.nan
    mean, var = stats(X)
    with_gap = masked_scores(cand[None, :], query, EQUAL_W, mean, var, metric)  # type: ignore[arg-type]
    keep = np.arange(D) != 2
    reference = masked_scores(
        cand[None, keep],
        query[keep],
        EQUAL_W[keep],
        mean[keep],
        var[keep],
        metric,  # type: ignore[arg-type]
    )
    assert with_gap.score[0] == pytest.approx(reference.score[0])
    assert with_gap.contributions[0, 2] == 0
    assert with_gap.shared[0] == pytest.approx(11 / 12)


def test_nothing_shared_gives_nan() -> None:
    X = np.full((1, D), np.nan)
    s = masked_scores(X, np.zeros(D), EQUAL_W, np.zeros(D), np.ones(D), "euclidean")
    assert np.isnan(s.score[0]) and s.shared[0] == 0


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_full_rank_projection_equals_unprojected(metric: str) -> None:
    X = random_population()
    X = X - X.mean(axis=0)  # PCA (and so projected cosine) works from the data's centre
    mean, var = stats(X)
    plain = masked_scores(X, X[0], EQUAL_W, mean, var, metric)  # type: ignore[arg-type]
    sqrt_w = np.sqrt(EQUAL_W)
    Xw = X * sqrt_w
    projected = projected_scores(
        Xw,
        Xw[0],
        np.eye(D),
        np.cov(Xw, rowvar=False, ddof=0),
        Xw.mean(axis=0),
        metric,  # type: ignore[arg-type]
    )
    np.testing.assert_allclose(projected.score, plain.score, atol=1e-9)
    np.testing.assert_allclose(projected.contributions, plain.contributions, atol=1e-9)


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_reduced_projection_contributions_sum_to_score(metric: str) -> None:
    from sklearn.decomposition import PCA

    X = random_population()
    pca = PCA(n_components=4).fit(X)
    P = pca.components_.T @ pca.components_
    s = projected_scores(X, X[0], P, np.cov(X, rowvar=False, ddof=0), X.mean(axis=0), metric)  # type: ignore[arg-type]
    np.testing.assert_allclose(s.contributions.sum(axis=1), s.score)
    assert s.score[0] == pytest.approx(100)
