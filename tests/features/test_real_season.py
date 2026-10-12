"""Features built from the real 2022-23 season (needs fixtures from scripts/make_fixtures.sh)."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from nba_similarity.features.definitions import FEATURE_NAMES
from nba_similarity.features.pipeline import build_features
from nba_similarity.ingestion.parse import parse_page
from nba_similarity.ingestion.pipeline import ingest
from nba_similarity.ingestion.schema import init_db
from tests.ingestion.conftest import read_fixture


class NoNetworkClient:
    def get(self, url: str) -> str:
        raise AssertionError(f"unexpected network request: {url}")


@pytest.fixture(scope="module")
def conn(tmp_path_factory: pytest.TempPathFactory) -> sqlite3.Connection:
    raw_dir = tmp_path_factory.mktemp("raw")
    for page in ("per_poss", "advanced"):
        html = read_fixture(f"NBA_2023_{page}.html.gz")
        (raw_dir / f"NBA_2023_{page}.html").write_text(html, encoding="utf-8")
    connection = sqlite3.connect(":memory:")
    init_db(connection)
    ingest([2023], connection, NoNetworkClient(), raw_dir, parse_page, today=date(2026, 10, 10))
    build_features(connection)
    return connection


def test_qualified_rows_and_full_coverage(conn: sqlite3.Connection) -> None:
    (n,) = conn.execute("SELECT COUNT(*) FROM player_season_features").fetchone()
    assert n == 367  # players with >= 500 minutes in 2022-23
    covered = conn.execute(
        "SELECT feature FROM feature_season_stats WHERE season = 2023 AND covered = 1"
    ).fetchall()
    assert {f for (f,) in covered} == set(FEATURE_NAMES)


def test_jokic_is_an_elite_passer_and_rebounder(conn: sqlite3.Connection) -> None:
    z_ast, z_drb, z_ts = conn.execute(
        "SELECT z_ast_pct, z_drb_pct, z_ts_pct FROM player_season_features "
        "WHERE player_id = 'jokicni01'"
    ).fetchone()
    assert z_ast > 3 and z_drb > 2 and z_ts > 2


def test_ft_pct_basis_damps_small_samples(conn: sqlite3.Connection) -> None:
    """The top FT% z-scores belong to high-volume shooters, not 10-for-10 seasons."""
    top = conn.execute(
        "SELECT player_id FROM player_season_features ORDER BY z_ft_pct DESC LIMIT 5"
    ).fetchall()
    assert ("lillada01",) in top
    (max_z,) = conn.execute("SELECT MAX(z_ft_pct) FROM player_season_features").fetchone()
    assert max_z < 3


def test_sqrt_features_rarely_clip(conn: sqlite3.Connection) -> None:
    (clipped,) = conn.execute(
        "SELECT COUNT(*) FROM player_season_features "
        "WHERE ABS(z_blk_pct) >= 4 OR ABS(z_orb_pct) >= 4 OR ABS(z_fta_rate) >= 4"
    ).fetchone()
    assert clipped == 0
