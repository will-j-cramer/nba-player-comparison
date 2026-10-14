from __future__ import annotations

import math
import sqlite3

import numpy as np
import pytest

from nba_similarity.careers.data import load_career_data
from nba_similarity.careers.lookup import find_career_player
from nba_similarity.careers.outcomes import percentile, todays_terms, weighted_quantiles
from nba_similarity.careers.windows import slot_weights, window_ages
from nba_similarity.similarity.lookup import PlayerNotFoundError


def test_window_ages_start_at_debut() -> None:
    assert window_ages(19, 21, 3) == [19, 20, 21]
    assert window_ages(20, 21, 3) == [20, 21]
    assert window_ages(19, 25, 2) == [24, 25]
    with pytest.raises(ValueError):
        window_ages(19, 21, 0)


def test_slot_weights_favor_the_latest_age() -> None:
    assert np.allclose(slot_weights(3, 0.6), [0.36, 0.6, 1.0])


def test_weighted_quantiles() -> None:
    values = np.array([1.0, 2.0, 3.0, np.nan])
    assert np.allclose(weighted_quantiles(values, np.ones(4), [0.5]), [2.0])
    heavy = weighted_quantiles(values, np.array([1.0, 1.0, 10.0, 1.0]), [0.5])
    assert heavy[0] > 2.5
    assert np.isnan(weighted_quantiles(np.array([np.nan]), np.ones(1))).all()


def test_todays_terms_inverts_the_sqrt_transform(league: sqlite3.Connection) -> None:
    data = load_career_data(league)
    j = data.stats.index("blk_pct")  # sqrt transform
    mean = data.season_mean.at[data.last_season, "blk_pct"]
    assert math.isclose(todays_terms(data, j, 0.0), mean**2)
    assert percentile(0.0) == 50


def test_lookup_by_name(league: sqlite3.Connection) -> None:
    data = load_career_data(league)
    assert find_career_player(data, "Player twin") == "twin"
    assert find_career_player(data, "twin") == "twin"
    with pytest.raises(PlayerNotFoundError, match="did you mean"):
        find_career_player(data, "Player tw")
