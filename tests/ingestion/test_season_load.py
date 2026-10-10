"""End-to-end load of one real season (2022-23) from saved pages, with no network access."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from nba_similarity.ingestion.parse import parse_page
from nba_similarity.ingestion.pipeline import ingest
from nba_similarity.ingestion.records import ADVANCED_COLUMNS, PER100_COLUMNS
from nba_similarity.ingestion.schema import init_db
from tests.ingestion.conftest import read_fixture

SEASON = 2023


class NoNetworkClient:
    def get(self, url: str) -> str:
        raise AssertionError(f"unexpected network request: {url}")


@pytest.fixture(scope="module")
def conn(tmp_path_factory: pytest.TempPathFactory) -> sqlite3.Connection:
    raw_dir = tmp_path_factory.mktemp("raw")
    for page in ("per_poss", "advanced"):
        html = read_fixture(f"NBA_{SEASON}_{page}.html.gz")
        (raw_dir / f"NBA_{SEASON}_{page}.html").write_text(html, encoding="utf-8")

    connection = sqlite3.connect(":memory:")
    init_db(connection)
    report = ingest(
        [SEASON], connection, NoNetworkClient(), raw_dir, parse_page, today=date(2026, 10, 10)
    )
    assert report.loaded == {SEASON: 539}
    return connection


def table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def test_season_loads_expected_columns(conn: sqlite3.Connection) -> None:
    assert table_columns(conn, "per100_stats") == ["player_id", "season", *PER100_COLUMNS]
    assert table_columns(conn, "advanced_stats") == ["player_id", "season", *ADVANCED_COLUMNS]
    assert table_columns(conn, "player_seasons") == [
        "player_id", "season", "age", "pos", "team", "n_teams", "g", "gs", "mp",
    ]  # fmt: skip


def test_season_has_reasonable_row_count(conn: sqlite3.Connection) -> None:
    for table in ("player_seasons", "per100_stats", "advanced_stats"):
        (count,) = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE season = ?", (SEASON,)
        ).fetchone()
        assert 400 <= count <= 700, table
    (dupes,) = conn.execute(
        "SELECT COUNT(*) FROM (SELECT player_id FROM player_seasons GROUP BY player_id, season "
        "HAVING COUNT(*) > 1)"
    ).fetchone()
    assert dupes == 0


def test_traded_player_is_one_combined_row(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT team, n_teams, g FROM player_seasons WHERE player_id = 'duranke01'"
    ).fetchone()
    assert row == ("TOT", 2, 47)
    stints = conn.execute(
        "SELECT team, g FROM player_season_teams WHERE player_id = 'duranke01' ORDER BY stint"
    ).fetchall()
    assert stints == [("BRK", 39), ("PHO", 8)]


def test_known_values_match_basketball_reference(conn: sqlite3.Connection) -> None:
    jokic = conn.execute(
        "SELECT a.ts_pct, a.usg_pct, a.ast_pct, a.bpm, p.pts "
        "FROM advanced_stats a JOIN per100_stats p USING (player_id, season) "
        "WHERE player_id = 'jokicni01' AND season = ?",
        (SEASON,),
    ).fetchone()
    assert jokic == (0.701, 27.2, 46.6, 13.0, 35.6)
