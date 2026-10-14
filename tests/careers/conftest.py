"""A synthetic league with planted careers, for career-comp tests.

Every season has the same background of one-season players of every age from 19 to 32, so a
given stat level maps to nearly the same z-score in every season, and every age has a
population for the score's baseline. The planted careers (levels by age 19, 20, 21):
- query:   2005-07, 2, 4, 6 (rising)
- twin:    2000-02, 2, 4, 6, then full seasons through 2009
- faller:  2001-03, 6, 4, 2 (the opposite trajectory), then full seasons
- gap:     2002 and 2004 only in the window (no NBA season at 20), then full seasons
- partial: 2003-05 like the twin, but only 300 minutes at age 20
- bust:    2000-02, 2, 4, 6.5, then out of the league
- young:   2013-15, 2, 4, 6: outcome still pending (the last season is 2015)
"""

from __future__ import annotations

import dataclasses
import sqlite3
from collections import defaultdict

import pytest

from nba_similarity.features.pipeline import build_features
from nba_similarity.ingestion.records import PlayerSeason
from nba_similarity.ingestion.schema import init_db
from nba_similarity.ingestion.store import write_season
from tests.features.conftest import make_player

SEASONS = range(2000, 2016)


def player(
    pid: str, season: int, age: int, level: float, mp: int = 2000, g: int = 70
) -> PlayerSeason:
    return dataclasses.replace(make_player(pid, season, mp, level), age=age, g=g)


def career(pid: str, first: int, levels: dict[int, float], after: int = 0) -> list[PlayerSeason]:
    """Seasons at the given ages (19 at `first`), then `after` more full seasons at level 5."""
    rows = [player(pid, first + age - 19, age, lv) for age, lv in levels.items()]
    last_age = max(levels)
    rows += [player(pid, first + a - 19, a, 5.0) for a in range(last_age + 1, last_age + 1 + after)]
    return rows


PLANTED = [
    *career("query", 2005, {19: 2, 20: 4, 21: 6}),
    *career("twin", 2000, {19: 2, 20: 4, 21: 6}, after=7),
    *career("faller", 2001, {19: 6, 20: 4, 21: 2}, after=7),
    *career("gap", 2002, {19: 2, 21: 6}, after=7),
    *career("partial", 2003, {19: 2, 21: 6}, after=7),
    player("partial", 2004, 20, 4, mp=300, g=10),  # same minutes per game, fewer games
    *career("bust", 2000, {19: 2, 20: 4, 21: 6.5}),
    *career("young", 2013, {19: 2, 20: 4, 21: 6}),
]


def build_league(conn: sqlite3.Connection) -> None:
    init_db(conn)
    by_season: dict[int, list[PlayerSeason]] = defaultdict(list)
    for ps in PLANTED:
        by_season[ps.season].append(ps)
    for season in SEASONS:
        background = [
            player(f"bg{season}_{i}", season, 19 + i % 14, 1.0 + i % 10, mp=1000 + 150 * (i % 7))
            for i in range(42)
        ]
        write_season(conn, season, background + by_season[season], is_final=True)
    build_features(conn)


@pytest.fixture
def league() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    build_league(conn)
    return conn
