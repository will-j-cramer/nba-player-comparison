"""Command-line entry point: `nba-features` / `python -m nba_similarity.features`."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from nba_similarity.features.pipeline import build_features
from nba_similarity.features.summary import print_feature_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nba-features",
        description="Build the era-adjusted feature matrix from ingested stats, then summarize it.",
    )
    parser.add_argument("--db", type=Path, default=Path("data/nba.sqlite"))
    parser.add_argument(
        "--summary-only", action="store_true", help="print the summary without rebuilding"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found; run nba-ingest first", file=sys.stderr)
        return 2
    conn = sqlite3.connect(args.db)
    try:
        if not args.summary_only:
            report = build_features(conn)
            print(
                f"built features for {report.n_qualified} of {report.n_player_seasons} "
                f"player-seasons across {report.n_seasons} seasons\n"
            )
        print_feature_summary(conn)
    except (ValueError, sqlite3.OperationalError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
