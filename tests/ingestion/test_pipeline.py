"""Orchestration tests: skip/resume and refresh behaviour, with a fake client and parser.

These exercise the pipeline's control flow only. Parsing of real Basketball-Reference HTML is
tested separately against saved pages in tests/fixtures/.
"""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from nba_similarity.ingestion.cache import Page
from nba_similarity.ingestion.client import PageNotFoundError, RateLimitedError
from nba_similarity.ingestion.parse import ParseError
from nba_similarity.ingestion.pipeline import current_season, ingest, is_final
from nba_similarity.ingestion.records import RawRow
from nba_similarity.ingestion.schema import init_db

TODAY = date(2026, 10, 10)


class FakeClient:
    """Serves '<season>' as the page body; seasons in `missing` return 404."""

    def __init__(self, missing: set[int] | None = None, rate_limit_on: int | None = None):
        self.urls: list[str] = []
        self.missing = missing or set()
        self.rate_limit_on = rate_limit_on

    def get(self, url: str) -> str:
        self.urls.append(url)
        season = int(url.rsplit("NBA_", 1)[1].split("_", 1)[0])
        if season in self.missing:
            raise PageNotFoundError(url)
        if season == self.rate_limit_on:
            raise RateLimitedError(url)
        return str(season)


def fake_parse(html: str, page: Page) -> list[RawRow]:
    return [
        {"player_id": f"p{html}", "name": "Player", "team": "BOS", "g": "1", "mp": "10"},
        {"player_id": "q", "name": "Q", "team": "LAL", "g": "2", "mp": "20"},
    ]


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    init_db(connection)
    return connection


def run(conn: sqlite3.Connection, tmp_path: Path, client: FakeClient, seasons, **kw):
    return ingest(seasons, conn, client, tmp_path / "raw", fake_parse, today=TODAY, **kw)


def test_season_helpers() -> None:
    assert current_season(date(2026, 10, 10)) == 2027
    assert current_season(date(2026, 3, 1)) == 2026
    assert is_final(2026, TODAY) and not is_final(2027, TODAY)
    assert not is_final(2026, date(2026, 6, 15))


def test_loads_seasons_and_caches_html(conn, tmp_path: Path) -> None:
    client = FakeClient()
    report = run(conn, tmp_path, client, [2024, 2025])

    assert report.loaded == {2024: 2, 2025: 2}
    assert len(client.urls) == 4
    assert (tmp_path / "raw" / "NBA_2024_advanced.html").read_text() == "2024"
    assert conn.execute("SELECT COUNT(*) FROM player_seasons").fetchone() == (4,)


def test_rerun_skips_final_seasons_without_fetching(conn, tmp_path: Path) -> None:
    run(conn, tmp_path, FakeClient(), [2024, 2025])
    client = FakeClient()
    report = run(conn, tmp_path, client, [2024, 2025])

    assert report.skipped == [2024, 2025] and report.loaded == {}
    assert client.urls == []


def test_parse_uses_cached_html_when_db_is_fresh(tmp_path: Path) -> None:
    first = sqlite3.connect(":memory:")
    init_db(first)
    run(first, tmp_path, FakeClient(), [2024])

    fresh = sqlite3.connect(":memory:")
    init_db(fresh)
    client = FakeClient()
    report = run(fresh, tmp_path, client, [2024])
    assert report.loaded == {2024: 2}
    assert client.urls == []  # rebuilt from data/raw without network


def test_force_refetches(conn, tmp_path: Path) -> None:
    run(conn, tmp_path, FakeClient(), [2024])
    client = FakeClient()
    report = run(conn, tmp_path, client, [2024], force=True)
    assert report.loaded == {2024: 2}
    assert len(client.urls) == 2


def test_in_progress_season_is_refetched_every_run(conn, tmp_path: Path) -> None:
    run(conn, tmp_path, FakeClient(), [2027])
    client = FakeClient()
    run(conn, tmp_path, client, [2027])
    assert len(client.urls) == 2
    assert conn.execute("SELECT is_final FROM ingested_seasons WHERE season=2027").fetchone() == (
        0,
    )


def test_season_loaded_in_progress_is_refetched_once_final(conn, tmp_path: Path) -> None:
    ingest([2026], conn, FakeClient(), tmp_path / "raw", fake_parse, today=date(2026, 3, 1))
    client = FakeClient()
    run(conn, tmp_path, client, [2026])
    assert len(client.urls) == 2  # stale mid-season cache is not reused
    assert conn.execute("SELECT is_final FROM ingested_seasons WHERE season=2026").fetchone() == (
        1,
    )


def test_missing_season_is_reported_not_fatal(conn, tmp_path: Path) -> None:
    report = run(conn, tmp_path, FakeClient(missing={2027}), [2026, 2027])
    assert report.loaded == {2026: 2}
    assert report.unavailable == [2027]


def test_rate_limit_aborts_but_keeps_completed_seasons(conn, tmp_path: Path) -> None:
    with pytest.raises(RateLimitedError):
        run(conn, tmp_path, FakeClient(rate_limit_on=2025), [2024, 2025])
    assert dict(conn.execute("SELECT season, is_final FROM ingested_seasons")) == {2024: 1}


def parse_without_table(html: str, page: Page) -> list[RawRow]:
    raise ParseError("no <table id='per_poss'> with a body")


def test_upcoming_season_page_without_table_is_unavailable(conn, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    report = ingest([2027], conn, FakeClient(), raw_dir, parse_without_table, today=TODAY)
    assert report.unavailable == [2027] and report.loaded == {}
    assert not list(raw_dir.glob("NBA_2027_*"))  # placeholder page is not kept in the cache
    assert conn.execute("SELECT COUNT(*) FROM ingested_seasons").fetchone() == (0,)


def test_finished_season_page_without_table_raises(conn, tmp_path: Path) -> None:
    with pytest.raises(ParseError):
        ingest([2025], conn, FakeClient(), tmp_path / "raw", parse_without_table, today=TODAY)
