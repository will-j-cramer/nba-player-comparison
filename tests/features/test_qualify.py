from __future__ import annotations

import pandas as pd
import pytest

from nba_similarity.features.qualify import min_minutes, qualify, schedule_games


@pytest.mark.parametrize(
    ("season", "expected"), [(2023, 500), (1999, 305), (2012, 402), (2020, 433), (2021, 439)]
)
def test_threshold_scales_with_schedule(season: int, expected: int) -> None:
    assert min_minutes(schedule_games(season, True, 82)) == expected


def test_in_progress_season_uses_games_played_so_far() -> None:
    assert schedule_games(2027, False, 20) == 20
    assert min_minutes(20) == 122
    assert schedule_games(2027, True, 20) == 82  # once final, the full schedule applies


def test_qualify_filters_by_season_threshold() -> None:
    df = pd.DataFrame(
        {
            "player_id": ["a", "b", "c", "d"],
            "season": [2023, 2023, 1999, 1999],
            "g": [82, 60, 50, 50],
            "mp": [499, 500, 304, 305],
        }
    )
    kept = qualify(df, final_seasons={1999, 2023})
    assert list(kept["player_id"]) == ["b", "d"]
    assert list(kept["min_minutes"]) == [500, 305]


def test_missing_minutes_do_not_qualify() -> None:
    df = pd.DataFrame({"player_id": ["a"], "season": [2023], "g": [82], "mp": [None]})
    assert qualify(df, final_seasons={2023}).empty
