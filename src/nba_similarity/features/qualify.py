"""Minimum-minutes filter that excludes small-sample seasons."""

from __future__ import annotations

import pandas as pd

from nba_similarity.features.definitions import FULL_SCHEDULE, MIN_MINUTES, SHORT_SCHEDULES


def schedule_games(season: int, is_final: bool, max_games_played: int | None) -> int:
    """Games per team in `season`; a season in progress uses games played so far."""
    if not is_final and max_games_played:
        return min(max_games_played, FULL_SCHEDULE)
    return SHORT_SCHEDULES.get(season, FULL_SCHEDULE)


def min_minutes(games: int, minutes: int = MIN_MINUTES) -> int:
    """`minutes` (default MIN_MINUTES) scaled to a `games`-game schedule (500 → 305 for 50)."""
    return round(minutes * games / FULL_SCHEDULE)


def season_minimums(
    df: pd.DataFrame, final_seasons: set[int], minutes: int = MIN_MINUTES
) -> dict[int, int]:
    """Each season's minimum minutes: `minutes` scaled to that season's schedule."""
    max_games = df.groupby("season")["g"].max()
    return {
        int(season): min_minutes(
            schedule_games(int(season), int(season) in final_seasons, _int_or_none(games)),
            minutes,
        )
        for season, games in max_games.items()
    }


def qualify(df: pd.DataFrame, final_seasons: set[int]) -> pd.DataFrame:
    """Keep rows whose minutes meet their season's threshold; adds a `min_minutes` column."""
    thresholds = season_minimums(df, final_seasons)
    out = df.assign(min_minutes=df["season"].map(thresholds))
    minutes = pd.to_numeric(out["mp"], errors="coerce").fillna(0)
    return out[minutes >= out["min_minutes"]].reset_index(drop=True)


def _int_or_none(value: object) -> int | None:
    return None if pd.isna(value) else int(value)  # type: ignore[arg-type]
