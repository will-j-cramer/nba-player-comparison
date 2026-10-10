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
from nba_similarity.ingestion.store import ingested_seasons, write_season


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    init_db(connection)
    return connection


def player_season(player_id: str, season: int, stints: tuple[str, ...] = ("BOS",)) -> PlayerSeason:
    return PlayerSeason(
        player_id=player_id,
        name=player_id.title(),
        season=season,
        age=25,
        pos="SF",
        team="TOT" if len(stints) > 1 else stints[0],
        g=70,
        gs=70,
        mp=2400,
        stints=tuple(Stint(team, 35, 1200) for team in stints),
        per100=dict.fromkeys(PER100_COLUMNS, 1.0) | {"fg3": None},
        advanced=dict.fromkeys(ADVANCED_COLUMNS, 2.0),
    )


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def test_schema_columns() -> None:
    connection = sqlite3.connect(":memory:")
    init_db(connection)
    assert columns(connection, "per100_stats") == ["player_id", "season", *PER100_COLUMNS]
    assert columns(connection, "advanced_stats") == ["player_id", "season", *ADVANCED_COLUMNS]


def test_write_season_round_trip(conn: sqlite3.Connection) -> None:
    traded = player_season("a", 2023, ("BRK", "PHO"))
    write_season(conn, 2023, [traded, player_season("b", 2023)], is_final=True)

    team = conn.execute("SELECT team, n_teams FROM player_seasons WHERE player_id='a'").fetchone()
    assert team == ("TOT", 2)
    stints = conn.execute(
        "SELECT stint, team FROM player_season_teams WHERE player_id='a' ORDER BY stint"
    ).fetchall()
    assert stints == [(1, "BRK"), (2, "PHO")]
    assert conn.execute("SELECT fg3, pts FROM per100_stats WHERE player_id='a'").fetchone() == (
        None,
        1.0,
    )
    assert ingested_seasons(conn) == {2023: True}


def test_rewriting_a_season_replaces_its_rows_only(conn: sqlite3.Connection) -> None:
    write_season(conn, 2022, [player_season("a", 2022)], is_final=True)
    write_season(conn, 2023, [player_season("a", 2023), player_season("b", 2023)], is_final=False)
    write_season(conn, 2023, [player_season("a", 2023)], is_final=True)

    counts = dict(conn.execute("SELECT season, COUNT(*) FROM player_seasons GROUP BY season"))
    assert counts == {2022: 1, 2023: 1}
    assert conn.execute("SELECT COUNT(*) FROM advanced_stats WHERE season=2023").fetchone() == (1,)
    assert ingested_seasons(conn) == {2022: True, 2023: True}


def test_failed_write_leaves_season_untouched(conn: sqlite3.Connection) -> None:
    write_season(conn, 2023, [player_season("a", 2023)], is_final=True)
    duplicate = [player_season("b", 2023), player_season("b", 2023)]
    with pytest.raises(sqlite3.IntegrityError):
        write_season(conn, 2023, duplicate, is_final=True)

    rows = conn.execute("SELECT player_id FROM player_seasons WHERE season=2023").fetchall()
    assert rows == [("a",)]
