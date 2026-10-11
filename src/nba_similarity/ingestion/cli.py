"""Command-line entry point: `nba-ingest` / `python -m nba_similarity.ingestion`."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from nba_similarity.ingestion.client import (
    DEFAULT_DELAY_S,
    FetchError,
    PoliteClient,
    RateLimitedError,
)
from nba_similarity.ingestion.parse import ParseError, parse_page
from nba_similarity.ingestion.pipeline import FIRST_SEASON, current_season, ingest
from nba_similarity.ingestion.schema import connect

logger = logging.getLogger("nba_similarity.ingestion")

MIN_DELAY_S = 3.0  # Basketball-Reference allows roughly 20 requests per minute


def _delay(value: str) -> float:
    delay = float(value)
    if delay < MIN_DELAY_S:
        raise argparse.ArgumentTypeError(f"must be at least {MIN_DELAY_S} seconds")
    return delay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nba-ingest",
        description="Load Basketball-Reference per-100-possession and advanced stats into SQLite.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=FIRST_SEASON,
        help="first season end year (default: %(default)s)",
    )
    parser.add_argument(
        "--end", type=int, default=None, help="last season end year (default: current season)"
    )
    parser.add_argument("--db", type=Path, default=Path("data/nba.sqlite"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--force", action="store_true", help="re-download and reload seasons already loaded"
    )
    parser.add_argument(
        "--delay",
        type=_delay,
        default=DEFAULT_DELAY_S,
        help="seconds between requests (default: %(default)s, minimum: 3)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    end = args.end if args.end is not None else current_season(date.today())
    if args.start < FIRST_SEASON or args.start > end:
        logger.error("--start must be between %d and --end (%d)", FIRST_SEASON, end)
        return 2

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(str(args.db))
    try:
        report = ingest(
            range(args.start, end + 1),
            conn,
            PoliteClient(args.delay),
            args.raw_dir,
            parse_page,
            force=args.force,
        )
    except RateLimitedError as exc:
        logger.error("%s. Seasons loaded so far are kept; re-run later to resume.", exc)
        return 3
    except FetchError as exc:
        logger.error("fetch failed: %s. Re-run to resume.", exc)
        return 1
    except ParseError as exc:
        logger.error("could not parse a finished season: %s. B-Ref's layout may have changed.", exc)
        return 1
    finally:
        conn.close()

    rows = sum(report.loaded.values())
    print(
        f"loaded {len(report.loaded)} seasons ({rows} player-seasons), "
        f"skipped {len(report.skipped)} already loaded, "
        f"{len(report.unavailable)} unavailable {report.unavailable or ''}".rstrip()
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
