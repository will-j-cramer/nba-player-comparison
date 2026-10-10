"""Parses Basketball-Reference league season pages into raw rows keyed by our column names."""

from __future__ import annotations

from bs4 import BeautifulSoup, SoupStrainer

from nba_similarity.ingestion.cache import Page
from nba_similarity.ingestion.columns import (
    ADVANCED_STATS,
    IDENTITY_STATS,
    NAME_STAT,
    PER100_STATS,
    PLAYER_ID_ATTR,
)
from nba_similarity.ingestion.records import RawRow

_PAGE_STATS: dict[Page, dict[str, str]] = {"per_poss": PER100_STATS, "advanced": ADVANCED_STATS}


class ParseError(Exception):
    """The page does not have the expected table layout (B-Ref may have changed its HTML)."""


def parse_page(html: str, page: Page) -> list[RawRow]:
    """Return one row per player-team line of the page's main stats table.

    Rows without a player id (the "League Average" row) are dropped. Empty cells become None.
    Traded players keep both their combined row and per-team rows; transform.py collapses them.
    """
    soup = BeautifulSoup(html, "lxml", parse_only=SoupStrainer("table", id=page))
    table = soup.find("table", id=page)
    if table is None or table.tbody is None:
        raise ParseError(f"no <table id={page!r}> with a body")

    header = {th.get("data-stat") for th in table.thead.find_all("th")} if table.thead else set()
    missing = IDENTITY_STATS.keys() - header
    if missing:
        raise ParseError(f"table {page!r} is missing identity columns {sorted(missing)}")

    mapping = IDENTITY_STATS | _PAGE_STATS[page]
    rows: list[RawRow] = []
    for tr in table.tbody.find_all("tr"):
        name_cell = tr.find(attrs={"data-stat": NAME_STAT})
        player_id = name_cell.get(PLAYER_ID_ATTR) if name_cell is not None else None
        if not player_id:
            continue
        row: RawRow = {"player_id": str(player_id)}
        for cell in tr.find_all(["th", "td"]):
            column = mapping.get(cell.get("data-stat"))
            if column is not None:
                row[column] = cell.get_text(strip=True) or None
        rows.append(row)

    if not rows:
        raise ParseError(f"table {page!r} has no player rows")
    return rows
