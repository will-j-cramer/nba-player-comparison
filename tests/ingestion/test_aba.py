"""ABA roster ingestion, with hand-written HTML (no scraped data) and a fake client."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from nba_similarity.ingestion.aba import (
    aba_cache_path,
    ingest_aba,
    parse_aba_roster,
    stored_aba_seasons,
)
from nba_similarity.ingestion.client import PageNotFoundError
from nba_similarity.ingestion.parse import ParseError
from nba_similarity.ingestion.schema import init_db


def page(*ids: str, name_stat: str = "name_display") -> str:
    rows = "".join(
        f'<tr><td data-stat="{name_stat}" data-append-csv="{pid}"><a>{pid}</a></td>'
        '<td data-stat="pts_per_g">20.0</td></tr>'
        for pid in ids
    )
    return f'<html><table id="per_game"><tbody>{rows}<tr class="thead"></tr></tbody></table>'


class FakeClient:
    def __init__(self, pages: dict[int, str]) -> None:
        self.pages, self.urls = pages, []

    def get(self, url: str) -> str:
        self.urls.append(url)
        season = int(url.rsplit("ABA_", 1)[1].split("_", 1)[0])
        if season not in self.pages:
            raise PageNotFoundError(url)
        return self.pages[season]


@pytest.fixture
def conn() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    init_db(connection)
    return connection


def test_parse_reads_player_ids_in_both_markups() -> None:
    assert parse_aba_roster(page("ervinju01", "malonmo01")) == {"ervinju01", "malonmo01"}
    assert parse_aba_roster(page("gilmoar01", name_stat="player")) == {"gilmoar01"}
    with pytest.raises(ParseError, match="no player ids"):
        parse_aba_roster("<html><p>Just a moment...</p></html>")


def test_ingest_stores_rosters_caches_pages_and_skips_on_rerun(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    client = FakeClient({1975: page("ervinju01"), 1976: page("ervinju01", "malonmo01")})
    report = ingest_aba(conn, client, tmp_path, seasons=[1975, 1976, 1977])
    assert report.loaded == {1975: 1, 1976: 2} and report.unavailable == [1977]
    assert client.urls[0].endswith("/leagues/ABA_1975_per_game.html")
    assert aba_cache_path(tmp_path, 1976).exists()
    assert stored_aba_seasons(conn) == {1975, 1976}

    again = ingest_aba(conn, FakeClient({}), tmp_path, seasons=[1975, 1976])
    assert again.skipped == [1975, 1976] and again.loaded == {}


def test_cached_page_is_not_refetched(conn: sqlite3.Connection, tmp_path: Path) -> None:
    aba_cache_path(tmp_path, 1974).write_text(page("barryri01"), encoding="utf-8")
    client = FakeClient({})
    assert ingest_aba(conn, client, tmp_path, seasons=[1974]).loaded == {1974: 1}
    assert client.urls == []
