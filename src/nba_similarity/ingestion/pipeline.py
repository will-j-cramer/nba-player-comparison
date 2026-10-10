"""Orchestrates ingestion: fetch (or read cached) pages, parse, transform, store."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from nba_similarity.ingestion.cache import Fetcher, Page, fetch_page
from nba_similarity.ingestion.client import PageNotFoundError
from nba_similarity.ingestion.records import RawRow
from nba_similarity.ingestion.store import ingested_seasons, write_season
from nba_similarity.ingestion.transform import build_season

logger = logging.getLogger(__name__)

FIRST_SEASON = 1974  # 1973-74: first season with per-100-possession stats on B-Ref

ParsePage = Callable[[str, Page], list[RawRow]]
"""Parses one page's HTML into raw rows keyed by our column names."""


@dataclass
class IngestReport:
    loaded: dict[int, int] = field(default_factory=dict)  # season -> player-season rows
    skipped: list[int] = field(default_factory=list)  # already final in the database
    unavailable: list[int] = field(default_factory=list)  # 404, e.g. not started yet


def current_season(today: date) -> int:
    """End year of the season in progress (or most recently finished) on `today`."""
    return today.year + 1 if today.month >= 10 else today.year


def is_final(season: int, today: date) -> bool:
    """A season is treated as complete from July 1 of its end year (after the Finals)."""
    return today >= date(season, 7, 1)


def ingest(
    seasons: Iterable[int],
    conn: sqlite3.Connection,
    client: Fetcher,
    raw_dir: Path,
    parse_page: ParsePage,
    *,
    force: bool = False,
    today: date | None = None,
) -> IngestReport:
    """Load each season into the database, skipping seasons already loaded as final.

    The cached HTML is bypassed (refetched) when `force` is set, when the season is still in
    progress, or when it was last loaded while in progress. RateLimitedError and other
    FetchErrors propagate; seasons loaded before the failure stay loaded, so a re-run resumes.
    """
    today = today or date.today()
    report = IngestReport()
    loaded = ingested_seasons(conn)

    for season in seasons:
        final = is_final(season, today)
        if loaded.get(season) and not force:
            logger.info("season %d: already loaded, skipping", season)
            report.skipped.append(season)
            continue

        refresh = force or not final or season in loaded
        try:
            pages = {
                page: parse_page(fetch_page(season, page, client, raw_dir, refresh=refresh), page)
                for page in ("per_poss", "advanced")
            }
        except PageNotFoundError:
            logger.warning("season %d: page not found (season not started?), skipping", season)
            report.unavailable.append(season)
            continue

        player_seasons = build_season(season, pages["per_poss"], pages["advanced"])
        write_season(conn, season, player_seasons, is_final=final)
        report.loaded[season] = len(player_seasons)
        logger.info(
            "season %d: loaded %d player-seasons%s",
            season,
            len(player_seasons),
            "" if final else " (in progress)",
        )
    return report
