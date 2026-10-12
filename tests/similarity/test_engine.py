from __future__ import annotations

import numpy as np
import pytest

from nba_similarity.similarity.config import SimilarityConfig
from nba_similarity.similarity.engine import SimilarityEngine
from tests.similarity.conftest import NAMES, col, make_matrix, random_population

SCORING = [col(f) for f in ("pts", "usg_pct", "fg3a_rate", "fta_rate")]
DEFENSE = [col(f) for f in ("stl_pct", "blk_pct")]


def engine_with_twins(**config: object) -> SimilarityEngine:
    """Rows 0-1: the same player in two near-identical seasons; row 2: a different, close player."""
    z = random_population(200)
    z[1] = z[0] + 0.01
    z[2] = z[0] + 0.2
    ids = ["star", "star", "lookalike"] + [f"p{i}" for i in range(3, 200)]
    seasons = [2020, 2021, 2020] + [2020] * 197
    return SimilarityEngine(make_matrix(z, ids, seasons), SimilarityConfig(**config))  # type: ignore[arg-type]


def test_excludes_own_seasons_by_default() -> None:
    matches = engine_with_twins().find_similar("star", 2020, n=5)
    assert matches[0].player_id == "lookalike"
    assert all(m.player_id != "star" for m in matches)


def test_can_include_own_seasons() -> None:
    matches = engine_with_twins(include_same_player=True).find_similar("star", 2020, n=5)
    assert (matches[0].player_id, matches[0].season) == ("star", 2021)
    assert ("star", 2020) not in {(m.player_id, m.season) for m in matches}  # never itself


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("pca", [None, 4, 0.8])
def test_scores_are_0_to_100_and_sorted(metric: str, pca: int | float | None) -> None:
    engine = engine_with_twins(metric=metric, pca_components=pca)
    matches = engine.find_similar("star", 2020, n=25)
    assert len(matches) == 25
    scores = [m.score for m in matches]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s <= 100 for s in scores)


def test_category_weights_change_who_matches() -> None:
    z = np.zeros((4, len(NAMES)))
    z[0, SCORING] = 2.0
    z[0, DEFENSE] = 2.0
    z[1, SCORING] = 2.0  # same scorer, different defender
    z[2, DEFENSE] = 2.0  # same defender, different scorer
    z[3] = -1.0
    engine = SimilarityEngine(make_matrix(z), SimilarityConfig(standardize=False))
    scorer_first = engine.with_config(category_weights={"scoring": 5}).find_similar("p0", 2020)
    defender_first = engine.with_config(category_weights={"defense": 5}).find_similar("p0", 2020)
    assert scorer_first[0].player_id == "p1"
    assert defender_first[0].player_id == "p2"


def test_min_shared_excludes_seasons_missing_too_much() -> None:
    z = random_population(50)
    z[1] = z[0]
    z[1, SCORING] = np.nan  # missing 20% of the weight (the whole scoring category)
    engine = SimilarityEngine(make_matrix(z))
    assert engine.find_similar("p0", 2020, n=1)[0].player_id == "p1"  # 80% shared >= 75%
    strict = engine.with_config(min_shared=0.9)
    assert strict.find_similar("p0", 2020, n=1)[0].player_id != "p1"
    # with scoring weighted 0, the gap no longer counts against p1
    unweighted = engine.with_config(min_shared=0.9, category_weights={"scoring": 0})
    assert unweighted.find_similar("p0", 2020, n=1)[0].player_id == "p1"


def test_pca_only_compares_seasons_with_all_query_features() -> None:
    z = random_population(50)
    z[1] = z[0]
    z[1, col("usg_pct")] = np.nan
    engine = SimilarityEngine(make_matrix(z), SimilarityConfig(pca_components=0.9))
    ids = [m.player_id for m in engine.find_similar("p0", 2020, n=49)]
    assert "p1" not in ids and len(ids) == 48


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
def test_full_rank_pca_matches_no_pca(metric: str) -> None:
    base = engine_with_twins(metric=metric)
    full = base.with_config(pca_components=len(NAMES))
    a = base.find_similar("star", 2020, n=10)
    b = full.find_similar("star", 2020, n=10)
    assert [(m.player_id, m.season) for m in a] == [(m.player_id, m.season) for m in b]
    np.testing.assert_allclose([m.score for m in a], [m.score for m in b], atol=1e-6)


def test_similar_traits_explain_the_match() -> None:
    z = random_population(30) * 0.3
    z[0, col("ast_pct")] = 3.5  # a distinctive passer
    z[1] = z[0]
    z[1, col("ast_pct")] = 3.3  # ...matched by another distinctive passer
    z[1, col("blk_pct")] = np.nan  # a stat only one side has is never named
    engine = SimilarityEngine(make_matrix(z), SimilarityConfig(top_features=4))
    match = engine.find_similar("p0", 2020, n=1)[0]
    assert match.player_id == "p1"
    features = [c.feature for c in match.similar_traits]
    assert features[0] == "ast_pct"
    assert "blk_pct" not in features and len(features) == 4
    points = [c.points for c in match.similar_traits]
    assert points == sorted(points, reverse=True)
    top = match.similar_traits[0]
    assert (top.query_z, top.match_z, top.query_raw) == (3.5, 3.3, 35.0)
    assert match.differences == ()


def test_far_apart_stat_is_a_difference_not_a_similarity() -> None:
    """Like Harden vs Jokic on rebounding: closer than typical, but not alike."""
    z = random_population(200) * 0.5
    z[0] = 0.0
    z[0, col("ast_pct")] = 3.5
    z[0, col("drb_pct")] = 3.0
    z[1] = 0.0
    z[1, col("ast_pct")] = 3.4
    z[1, col("drb_pct")] = 0.8  # 2.2 SD apart
    engine = SimilarityEngine(make_matrix(z), SimilarityConfig(standardize=False))
    match = engine.find_similar("p0", 2020, n=1)[0]
    assert match.player_id == "p1"
    assert match.contributions["drb_pct"] > 0  # it still earns points...
    assert "drb_pct" not in [c.feature for c in match.similar_traits]  # ...but isn't "alike"
    assert match.differences[0].feature == "drb_pct"
    assert match.similar_traits[0].feature == "ast_pct"
    wider = engine.with_config(close_z=2.5).find_similar("p0", 2020, n=1)[0]
    assert "drb_pct" in [c.feature for c in wider.similar_traits]


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("pca", [None, 0.8])
def test_contributions_sum_to_score(metric: str, pca: float | None) -> None:
    engine = engine_with_twins(metric=metric, pca_components=pca)
    for match in engine.find_similar("star", 2020, n=10):
        assert sum(match.contributions.values()) == pytest.approx(match.score, abs=1e-6)


def test_zero_weight_category_never_explains() -> None:
    engine = engine_with_twins(category_weights={"rebounding": 0}, top_features=12)
    for match in engine.find_similar("star", 2020, n=5):
        named = {c.category for c in match.similar_traits + match.differences}
        assert "rebounding" not in named
        assert "orb_pct" not in match.contributions


def test_missing_stat_penalizes_incomplete_twin() -> None:
    """An identical season missing a stat ranks below an identical complete one."""
    z = random_population(50)
    z[1] = z[0]
    z[2] = z[0]
    z[2, col("fg3a_rate")] = np.nan  # e.g. a pre-1980 season
    engine = SimilarityEngine(make_matrix(z))
    first, second = engine.find_similar("p0", 2020, n=2)
    assert (first.player_id, second.player_id) == ("p1", "p2")
    assert first.score == pytest.approx(100) and second.score < 100


def test_unknown_query_raises() -> None:
    with pytest.raises(KeyError, match="not in the feature matrix"):
        engine_with_twins().find_similar("nobody", 2020)
