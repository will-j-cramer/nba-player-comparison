"""Exports the slim database the deployed app needs (for Streamlit Community Cloud).

    python scripts/export_app_db.py [--db data/nba.sqlite] [--out data/processed/nba_app.sqlite]

Copies only the tables the app reads: player names, the feature tables, the map and the
career tables (derived z-scores and per-season rows), plus the list of ingested seasons. The
raw scraped tables (per-100 and advanced stats, team stints, ABA rosters) stay behind. Run
`nba-features` and `nba-map` first; a missing or out-of-date map is refused. Upload the
result somewhere private and point the app's NBA_DB_URL secret at it (see the README). Never
commit it.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from nba_similarity.careers.data import load_career_data
from nba_similarity.mapping.store import is_stale, load_map
from nba_similarity.paths import default_db
from nba_similarity.similarity.data import load_feature_matrix

APP_TABLES = (
    "players",
    "feature_definitions",
    "feature_season_stats",
    "player_season_features",
    "player_season_map",
    "map_metadata",
    "player_season_career_features",
    "career_players",
    "career_season_stats",
    "ingested_seasons",  # which seasons are complete (career outcomes); no scraped stats
)
DEFAULT_OUT = Path("data/processed/nba_app.sqlite")


class ExportError(Exception):
    """The source database is not ready to export."""


def export_app_db(source: sqlite3.Connection, out: Path) -> dict[str, int]:
    """Write APP_TABLES (schema, indexes and rows) from `source` to a new file `out`.

    Returns the row count per table. Raises ExportError if the map is missing or stale.
    """
    stored = load_map(source)
    if stored is None:
        raise ExportError("no map in the database; run nba-map first")
    if is_stale(source, stored):
        raise ExportError("the map is out of date; re-run nba-map first")

    schema = {
        name: sql
        for name, sql in source.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND sql IS NOT NULL"
        )
    }
    missing = [t for t in APP_TABLES if t not in schema]
    if missing:
        raise ExportError(f"missing tables {missing}; run nba-features and nba-map")
    indexes = [
        sql
        for (sql,) in source.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND sql IS NOT NULL "
            f"AND tbl_name IN ({', '.join('?' * len(APP_TABLES))})",
            APP_TABLES,
        )
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    dest = sqlite3.connect(out)
    counts = {}
    try:
        with dest:
            for table in APP_TABLES:
                dest.execute(schema[table])
                rows = source.execute(f"SELECT * FROM {table}").fetchall()
                if rows:
                    marks = ", ".join("?" * len(rows[0]))
                    dest.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
                counts[table] = len(rows)
            for sql in indexes:
                dest.execute(sql)
        dest.execute("VACUUM")
        load_feature_matrix(dest)  # the app can read it
        load_career_data(dest)
        if is_stale(dest, load_map(dest)):  # type: ignore[arg-type]
            raise ExportError("exported map does not match the exported features")
    finally:
        dest.close()
    return counts


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db", type=Path, default=default_db(), help="source (default: $NBA_DB or %(default)s)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Export and print a summary; returns a process exit code."""
    args = build_parser().parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found", file=sys.stderr)
        return 2
    if args.out.resolve() == args.db.resolve():
        print("--out must differ from --db", file=sys.stderr)
        return 2
    source = sqlite3.connect(args.db)
    try:
        counts = export_app_db(source, args.out)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        source.close()
    for table, n in counts.items():
        print(f"  {table:<24} {n:>7,} rows")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB); upload it privately, "
          "never commit it")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
