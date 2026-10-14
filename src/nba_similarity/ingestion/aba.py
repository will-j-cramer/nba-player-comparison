"""ABA rosters, 1967-68 to 1975-76: who played in the ABA before the 1976 merger.

Our stats come from NBA league pages only, so a player who started in the ABA (Moses
Malone, Julius Erving) looks like a late NBA starter. Career comps need to know that the
start of such a career is missing. One ABA league page per season gives the player ids;
nothing else from it is stored. Fetched with the same rate-limited client and HTML cache as
the NBA pages (9 pages, once).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

from nba_similarity.ingestion.cache import BASE_URL, Fetcher, fetch_cached
from nba_similarity.ingestion.client import PageNotFoundError
from nba_similarity.ingestion.columns import NAME_STAT, PLAYER_ID_ATTR
from nba_similarity.ingestion.parse import ParseError

logger = logging.getLogger(__name__)

ABA_SEASONS = range(1968, 1977)  # end years: 1967-68 through 1975-76
# Player-name cells: current B-Ref markup uses "name_display"; older pages used "player".
_NAME_STATS = (NAME_STAT, "player")


@dataclass
class AbaReport:
    """What an ABA roster run did, per season."""

    loaded: dict[int, int] = field(default_factory=dict)  # season -> players
    skipped: list[int] = field(default_factory=list)  # already stored
    unavailable: list[int] = field(default_factory=list)


def aba_url(season: int) -> str:
    """URL of an ABA league season page (per-game table; only player ids are read)."""
    return f"{BASE_URL}/ABA_{season}_per_game.html"


def aba_cache_path(raw_dir: Path, season: int) -> Path:
    """Where an ABA season page is cached under `raw_dir`."""
    return raw_dir / f"ABA_{season}_per_game.html"


def parse_aba_roster(html: str) -> set[str]:
    """B-Ref player ids of everyone in the page's player tables."""
    soup = BeautifulSoup(html, "lxml")
    ids = {
        str(cell[PLAYER_ID_ATTR])
        for cell in soup.find_all(["td", "th"], attrs={"data-stat": list(_NAME_STATS)})
        if cell.get(PLAYER_ID_ATTR)
    }
    if not ids:
        raise ParseError("no player ids found on the ABA page (B-Ref's layout may have changed)")
    return ids


def stored_aba_seasons(conn: sqlite3.Connection) -> set[int]:
    """ABA seasons already in aba_rosters."""
    return {s for (s,) in conn.execute("SELECT DISTINCT season FROM aba_rosters")}


def write_aba_season(conn: sqlite3.Connection, season: int, player_ids: Iterable[str]) -> None:
    """Replace one ABA season's roster in a single transaction."""
    with conn:
        conn.execute("DELETE FROM aba_rosters WHERE season = ?", (season,))
        conn.executemany(
            "INSERT INTO aba_rosters (player_id, season) VALUES (?, ?)",
            [(pid, season) for pid in sorted(set(player_ids))],
        )


def ingest_aba(
    conn: sqlite3.Connection,
    client: Fetcher,
    raw_dir: Path,
    seasons: Iterable[int] = ABA_SEASONS,
) -> AbaReport:
    """Store the ABA rosters not stored yet. ABA seasons are final, so the cache is never
    bypassed. A missing page is reported, not fatal; ParseError and FetchError propagate."""
    report = AbaReport()
    done = stored_aba_seasons(conn)
    for season in seasons:
        if season in done:
            report.skipped.append(season)
            continue
        try:
            html = fetch_cached(aba_url(season), aba_cache_path(raw_dir, season), client)
        except PageNotFoundError:
            logger.warning("ABA season %d: page not found, skipping", season)
            report.unavailable.append(season)
            continue
        ids = parse_aba_roster(html)
        write_aba_season(conn, season, ids)
        report.loaded[season] = len(ids)
        logger.info("ABA season %d: %d players", season, len(ids))
    return report
