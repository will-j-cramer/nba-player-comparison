from __future__ import annotations

import sqlite3

import pytest

from nba_similarity.ingestion.records import (
    ADVANCED_COLUMNS,
    PER100_COLUMNS,
    PlayerSeason,
    Stint,
)
from nba_similarity.ingestion.schema import init_db
from nba_similarity.ingestion.store import write_season

# Stats that did not exist before 1979-80 / 1977-78 are None in early seasons
# (feature name -> the ingestion column it comes from).
ERA_GAPS_ADVANCED = {"usg_pct", "tov_pct", "fg3a_rate"}
ERA_GAPS_PER100 = {"fg3", "fg3a", "fg3_pct", "tov"}
ERA_GAPS = {"fg3a", "usg_pct", "tov_pct"}  # the features affected


def make_player(player_id: str, season: int, mp: int, level: float) -> PlayerSeason:
    early = season < 1980
    advanced = {c: level for c in ADVANCED_COLUMNS}
    per100 = {c: level for c in PER100_COLUMNS}
    if early:
        advanced.update(dict.fromkeys(ERA_GAPS_ADVANCED, None))
        per100.update(dict.fromkeys(ERA_GAPS_PER100, None))
    return PlayerSeason(
        player_id=player_id,
        name=f"Player {player_id}",
        season=season,
        age=25,
        pos="SF",
        team="BOS",
        g=70,
        gs=50,
        mp=mp,
        stints=(Stint("BOS", 70, mp),),
        per100=per100,
        advanced=advanced,
    )


@pytest.fixture
def ingested_db() -> sqlite3.Connection:
    """Two seasons of synthetic players: 1976 (with era gaps) and 2020 (shortened: 433 min)."""
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    for season, n in ((1976, 12), (2020, 15)):
        players = [make_player(f"p{season}_{i}", season, 600 + 100 * i, float(i)) for i in range(n)]
        players.append(make_player(f"bench{season}", season, 420, 99.0))  # below 500, above 433
        write_season(conn, season, players, is_final=True)
    return conn
