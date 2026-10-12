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


def test_missing_stat_counts_as_typical_not_skipped() -> None:
    """A candidate lacking a stat earns 0 points for it, but it stays in the denominator."""
    X = random_population()
    query, cand = X[0].copy(), X[1].copy()
    cand[2] = np.nan
    mean, var = stats(X)
    s = masked_scores(cand[None, :], query, EQUAL_W, mean, var, "euclidean")
    expected = EQUAL_W * ((query - mean) ** 2 + var)
    keep = np.arange(D) != 2
    gaps = EQUAL_W[keep] * (cand[keep] - query[keep]) ** 2
    assert s.score[0] == pytest.approx(100 * (expected[keep] - gaps).sum() / expected.sum())
    assert s.contributions[0, 2] == 0
    assert s.shared[0] == pytest.approx(11 / 12)


def test_missing_a_distinctive_stat_costs_more() -> None:
    """Skipping the query's standout stat (Curry's 3PA rate) costs more than an average one."""
    X = random_population()
    mean, var = stats(X)
    query = np.zeros(D)
    query[2] = 3.0  # distinctive
    twin_missing_distinctive = query.copy()
    twin_missing_distinctive[2] = np.nan
    twin_missing_average = query.copy()
    twin_missing_average[5] = np.nan
    pair = np.vstack([twin_missing_distinctive, twin_missing_average])
    s = masked_scores(pair, query, EQUAL_W, mean, var, "euclidean")
    assert s.score[0] < s.score[1] < 100


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_missing_is_not_zero_filled(metric: str) -> None:
    X = random_population()
    mean, var = stats(X)
    query = X[0]
    missing, zero_filled = X[1].copy(), X[1].copy()
    missing[2], zero_filled[2] = np.nan, 0.0
    s = masked_scores(np.vstack([missing, zero_filled]), query, EQUAL_W, mean, var, metric)  # type: ignore[arg-type]
    assert s.score[0] != pytest.approx(s.score[1])


def test_cosine_missing_stat_lowers_the_score() -> None:
    X = random_population()
    mean, var = stats(X)
    query = X[0]
    complete = query * 1.0
    gap = query.copy()
    gap[4] = np.nan
    s = masked_scores(np.vstack([complete, gap]), query, EQUAL_W, mean, var, "cosine")
    assert s.score[0] == pytest.approx(100)
    assert s.score[1] < 100
    np.testing.assert_allclose(np.nansum(s.contributions, axis=1), s.score)


def test_stats_the_query_lacks_are_ignored() -> None:
    X = random_population()
    mean, var = stats(X)
    query = X[0].copy()
    query[3] = np.nan  # e.g. a 1975 query without 3PA rate
    s = masked_scores(X, query, EQUAL_W, mean, var, "euclidean")
    assert s.score[0] == pytest.approx(100)  # itself, compared on the 11 stats it has
    assert (s.contributions[:, 3] == 0).all()


def test_nothing_shared_scores_zero_and_shares_nothing() -> None:
    X = np.full((1, D), np.nan)
    s = masked_scores(X, np.zeros(D), EQUAL_W, np.zeros(D), np.ones(D), "euclidean")
    assert s.score[0] == 0 and s.shared[0] == 0


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


@pytest.mark.filterwarnings("ignore:Mean of empty slice", "ignore:Degrees of freedom")
@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_a_stat_with_no_values_anywhere_is_ignored(metric: str) -> None:
    """Regression: an all-NaN column (NaN mean/var) must not turn every score into NaN."""
    X = random_population()
    X[:, 4] = np.nan
    mean, var = np.nanmean(X, axis=0), np.nanvar(X, axis=0)  # NaN for column 4
    s = masked_scores(X, X[0], EQUAL_W, mean, var, metric)  # type: ignore[arg-type]
    assert np.isfinite(s.score).all()
    assert s.score[0] == pytest.approx(100)
