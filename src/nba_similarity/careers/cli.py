"""Command-line entry point: `nba-career "Victor Wembanyama" --through-age 21`."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from nba_similarity.careers.data import load_career_data
from nba_similarity.careers.engine import CareerConfig, CareerEngine, CareerResult
from nba_similarity.careers.format import (
    OUTCOME_LEGEND,
    comp_summary,
    format_ages,
    format_outcome,
)
from nba_similarity.careers.lookup import find_career_player
from nba_similarity.paths import default_db
from nba_similarity.seasons import season_label
from nba_similarity.similarity.cli import parse_weights
from nba_similarity.similarity.explain import SHORT_LABELS
from nba_similarity.similarity.lookup import PlayerNotFoundError

# Stats shown in the projection table (with minutes per game in today's terms).
PROJECTION_STATS = ("usg_pct", "ts_pct", "ast_pct", "drb_pct", "blk_pct", "fg3a")


def build_parser() -> argparse.ArgumentParser:
    """Command-line options."""
    parser = argparse.ArgumentParser(
        prog="nba-career",
        description="Players whose careers through the same age were most similar, and what "
        "they did next.",
    )
    parser.add_argument("player", help='player name (e.g. "Victor Wembanyama") or B-Ref id')
    parser.add_argument("--through-age", type=int, required=True, help="last age compared")
    parser.add_argument("--window", type=int, default=3, help="ages compared (default 3)")
    parser.add_argument("--horizon", type=int, default=5, help="seasons projected (default 5)")
    parser.add_argument("-n", type=int, default=10, help="comps listed (default 10)")
    parser.add_argument(
        "--weight", action="append", default=[], metavar="CATEGORY=W",
        help="category weight, repeatable (categories as in nba-similar, plus role=minutes)",
    )  # fmt: skip
    parser.add_argument(
        "--max-experience-gap", type=int, default=None,
        help="only comps within this many NBA seasons of the player's experience",
    )  # fmt: skip
    parser.add_argument(
        "--db", type=Path, default=default_db(),
        help="SQLite database (default: $NBA_DB or %(default)s)",
    )  # fmt: skip
    return parser


def format_result(result: CareerResult, config: CareerConfig) -> str:
    """The comps list and the projection table as text."""
    seasons = ", ".join(
        season_label(s) if s else f"age {a}: -"
        for a, s in zip(result.ages, result.query_seasons, strict=True)
    )
    lines = [
        f"Career comps for {result.name} through age {result.through_age} "
        f"(ages {format_ages(result.ages)}: {seasons}; "
        f"{_seasons(result.query_experience)} in the NBA)",
        f"next {config.horizon} seasons: {OUTCOME_LEGEND}",
        "",
    ]
    first_future = result.through_age + 1
    for rank, comp in enumerate(result.comps, start=1):
        coverage = "" if comp.shared >= 0.999 else f" [{comp.shared:.0%} of window]"
        future = format_outcome(comp.outcome, config.horizon, first_future)
        lines.append(
            f"{rank:>2}. {comp.name:<26} {comp.score:5.1f}{coverage}  "
            f"{_seasons(comp.experience)}  |  {future}"
        )
        lines += [f"      {line}" for line in comp_summary(comp).splitlines()]
    if not result.comps:
        lines.append("no comparable careers found")

    proj = result.projection
    lines.append("")
    if proj.n_comps == 0:
        lines.append("projection: no comps with known outcomes yet")
        return "\n".join(lines)
    lines.append(
        f"Projection from the {proj.n_comps} most similar comps with known outcomes "
        "(score-weighted; percentile within season, median [25-75%]):"
    )
    data_stats = {s: j for j, s in enumerate(result.stats)}
    header = f"{'age':>4} {'full/part/out':>14}  " + "  ".join(
        f"{label:>12}"
        for label in (*(f"{SHORT_LABELS[s]} pct" for s in PROJECTION_STATS), "MPG today")
    )
    lines.append(header)
    for age in proj.ages:
        shares = "/".join(f"{age.shares[c]:.0%}" for c in ("qualified", "limited", "out"))
        cells = []
        for stat in PROJECTION_STATS:
            p25, p50, p75 = age.pct[data_stats[stat]]
            cells.append("-" if p50 != p50 else f"{p50:3.0f} [{p25:.0f}-{p75:.0f}]")
        mpg = age.today[data_stats["mpg"]]
        cells.append("-" if mpg[1] != mpg[1] else f"{mpg[1]:4.1f} [{mpg[0]:.0f}-{mpg[2]:.0f}]")
        lines.append(f"{age.age:>4} {shares:>14}  " + "  ".join(f"{c:>12}" for c in cells))
    lines.append(f"(stats only from the comps who played a full season at that age; n per age: "
                 f"{', '.join(str(a.n_qualified) for a in proj.ages)})")  # fmt: skip
    return "\n".join(lines)


def _seasons(n: int) -> str:
    return f"{n} season" + ("" if n == 1 else "s")


def main(argv: Sequence[str] | None = None) -> int:
    """Print career comps and the projection; returns a process exit code."""
    args = build_parser().parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found; run nba-ingest and nba-features first", file=sys.stderr)
        return 2
    try:
        config = CareerConfig(
            window=args.window,
            horizon=args.horizon,
            n=args.n,
            category_weights=parse_weights(args.weight),
            max_experience_gap=args.max_experience_gap,
        )
        conn = sqlite3.connect(args.db)
        try:
            data = load_career_data(conn)
        finally:
            conn.close()
        player_id = find_career_player(data, args.player)
        result = CareerEngine(data, config).find_comps(player_id, args.through_age)
    except (ValueError, KeyError, PlayerNotFoundError, argparse.ArgumentTypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_result(result, config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
