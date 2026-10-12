"""Command-line entry point: `nba-similar "Nikola Jokić" 2023` (season = end year)."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from nba_similarity.similarity.config import SimilarityConfig
from nba_similarity.similarity.engine import Match, SimilarityEngine
from nba_similarity.similarity.explain import format_contribution, season_label
from nba_similarity.similarity.lookup import PlayerNotFoundError, find_player_id


def _weights(items: Sequence[str]) -> dict[str, float]:
    weights = {}
    for item in items:
        category, sep, value = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"expected category=weight, got {item!r}")
        weights[category] = float(value)
    return weights


def _pca(value: str) -> int | float:
    number = float(value)
    return int(number) if number >= 1 else number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nba-similar", description="Find the most similar player-seasons in NBA history."
    )
    parser.add_argument("player", help='player name (e.g. "Nikola Jokic") or B-Ref id')
    parser.add_argument("season", type=int, help="season end year (2023 = 2022-23)")
    parser.add_argument("-n", type=int, default=10, help="number of matches (default 10)")
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="CATEGORY=W",
        help="category weight, repeatable (e.g. --weight scoring=2 --weight defense=0)",
    )
    parser.add_argument(
        "--pca", type=_pca, default=None, help="PCA components (int) or variance share (0-1)"
    )
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument(
        "--include-same-player", action="store_true", help="include the player's other seasons"
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help="list every matching season of a player (default: each comp player once)",
    )
    parser.add_argument("--top-features", type=int, default=3)
    parser.add_argument("--db", type=Path, default=Path("data/nba.sqlite"))
    return parser


def format_match(rank: int, match: Match) -> str:
    season = season_label(match.season)
    coverage = "" if match.shared >= 0.999 else f" [{match.shared:.0%} of stats]"
    line = f"{rank:>2}. {match.name:<26} {season}  {match.score:5.1f}{coverage}"
    alike = "; ".join(format_contribution(c) for c in match.similar_traits) or "-"
    lines = [line, f"      alike:  {alike}"]
    if match.differences:
        lines.append(
            f"      differ: {'; '.join(format_contribution(c) for c in match.differences)}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found; run nba-ingest and nba-features first", file=sys.stderr)
        return 2
    try:
        config = SimilarityConfig(
            metric=args.metric,
            category_weights=_weights(args.weight),
            standardize=not args.no_standardize,
            pca_components=args.pca,
            include_same_player=args.include_same_player,
            distinct_players=not args.all_seasons,
            top_features=args.top_features,
        )
        conn = sqlite3.connect(args.db)
        try:
            engine = SimilarityEngine.from_db(conn, config)
        finally:
            conn.close()
        player_id = find_player_id(engine.matrix, args.player, args.season)
        matches = engine.find_similar(player_id, args.season, n=args.n)
    except (ValueError, argparse.ArgumentTypeError, PlayerNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    query_name = engine.matrix.names[engine.matrix.row(player_id, args.season)]
    method = config.metric + (f", PCA {config.pca_components}" if config.pca_components else "")
    print(f"Most similar to {query_name} {season_label(args.season)} ({method}):")
    for rank, match in enumerate(matches, start=1):
        print(format_match(rank, match))
    return 0


if __name__ == "__main__":
    sys.exit(main())
