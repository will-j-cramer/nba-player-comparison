"""Command-line entry point: `nba-map` precomputes the 2D map into the database."""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from nba_similarity.mapping.embed import (
    DEFAULT_MIN_DIST,
    DEFAULT_N_NEIGHBORS,
    DEFAULT_SEED,
    build_embedding,
)
from nba_similarity.mapping.store import MapParams, features_signature, write_map
from nba_similarity.paths import default_db
from nba_similarity.similarity.engine import SimilarityEngine


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(
        prog="nba-map",
        description="Precompute the 2D UMAP map of all player-seasons (re-run after nba-features).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db(),
        help="SQLite database (default: $NBA_DB or %(default)s)",
    )
    parser.add_argument("--n-neighbors", type=int, default=DEFAULT_N_NEIGHBORS)
    parser.add_argument("--min-dist", type=float, default=DEFAULT_MIN_DIST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build and store the map; returns a process exit code."""
    args = build_parser().parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found; run nba-ingest and nba-features first", file=sys.stderr)
        return 2
    conn = sqlite3.connect(args.db)
    try:
        engine = SimilarityEngine.from_db(conn)  # default weights: equal per stat
        x, weights = engine.standardized()
        print(f"building map of {len(x)} player-seasons (first run compiles; ~1-2 min)...")
        start = time.monotonic()
        xy = build_embedding(
            x, weights, n_neighbors=args.n_neighbors, min_dist=args.min_dist, seed=args.seed
        )
        params = MapParams(args.n_neighbors, args.min_dist, args.seed)
        write_map(
            conn, engine.matrix.player_ids, engine.matrix.seasons, xy, params,
            features_signature(conn),
        )  # fmt: skip
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(f"map saved for {len(xy)} player-seasons in {time.monotonic() - start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
