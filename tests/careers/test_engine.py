from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from nba_similarity.careers.data import load_career_data
from nba_similarity.careers.engine import CareerConfig, CareerEngine


def comps(league: sqlite3.Connection, **config: object):
    data = load_career_data(league)
    result = CareerEngine(data, CareerConfig(n=50, **config)).find_comps("query", 21)
    return result, {c.player_id: c for c in result.comps}, [c.player_id for c in result.comps]


def test_window_covers_ages_19_to_21(league: sqlite3.Connection) -> None:
    result, _, _ = comps(league)
    assert result.ages == (19, 20, 21) and result.query_seasons == (2005, 2006, 2007)
    assert result.query_experience == 3


def test_same_trajectory_ranks_first_and_opposite_trajectory_ranks_low(
    league: sqlite3.Connection,
) -> None:
    _, by_id, order = comps(league)
    assert order[0] == "twin" and by_id["twin"].score > 95
    assert order.index("faller") > order.index("gap")  # same average, opposite shape
    assert by_id["faller"].score < by_id["twin"].score - 20


def test_missing_and_partial_seasons_count_less_but_stay_comparable(
    league: sqlite3.Connection,
) -> None:
    _, by_id, _ = comps(league)
    twin, gap, partial = by_id["twin"], by_id["gap"], by_id["partial"]
    assert gap.window_seasons == (2002, None, 2004)
    assert gap.score < twin.score and partial.score < twin.score
    assert gap.score < partial.score  # a 300-minute season beats no season
    assert gap.shared < 1 and np.isclose(partial.shared, 1)


def test_contributions_sum_to_the_score_and_explain_by_age(league: sqlite3.Connection) -> None:
    _, by_id, _ = comps(league)
    twin, faller = by_id["twin"], by_id["faller"]
    assert np.isclose(sum(twin.contributions.values()), twin.score)
    assert {age for age, _ in twin.contributions} == {19, 20, 21}
    assert faller.differences  # rising vs falling: they differ at every age
    for trait in faller.differences:
        assert trait.gap > 1.0 and trait.ages == (19, 20, 21)
        assert (trait.query_raw, trait.comp_raw) == (6.0, 2.0)  # values at the latest age


def test_outcomes_bust_pending_and_projection(league: sqlite3.Connection) -> None:
    result, by_id, _ = comps(league, horizon=3)
    assert [s.outcome for s in by_id["bust"].outcome.seasons] == ["out"] * 3
    assert [s.outcome for s in by_id["twin"].outcome.seasons] == ["qualified"] * 3
    assert not by_id["young"].outcome.observable  # 2016+ hasn't been played
    proj = result.projection
    assert proj.n_comps >= 4 and [a.age for a in proj.ages] == [22, 23, 24]
    first = proj.ages[0]
    assert np.isclose(sum(first.shares.values()), 1)
    assert 0 < first.shares["out"] < 1  # the bust counts, weighted by its score
    stat = result.stats.index("ts_pct")
    assert first.pct[stat, 0] <= first.pct[stat, 1] <= first.pct[stat, 2]


def test_censored_careers_are_not_candidates(league: sqlite3.Connection) -> None:
    league.execute("UPDATE career_players SET censored_through_age = 19 WHERE player_id = 'twin'")
    _, by_id, _ = comps(league)
    assert "twin" not in by_id
    _, by_id, _ = comps(league, window=1)  # only age 21: the censored ages aren't needed
    assert "twin" in by_id


def test_experience_gap_filter(league: sqlite3.Connection) -> None:
    _, by_id, _ = comps(league, max_experience_gap=0)
    assert "gap" not in by_id  # 2 NBA seasons through age 21 vs the query's 3
    assert "twin" in by_id


def test_query_needs_a_season_at_the_age(league: sqlite3.Connection) -> None:
    engine = CareerEngine(load_career_data(league))
    with pytest.raises(ValueError, match="age 25"):
        engine.find_comps("query", 25)
    with pytest.raises(KeyError):
        engine.find_comps("nobody", 21)
