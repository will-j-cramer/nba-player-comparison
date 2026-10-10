"""Phase 1 sanity report over data/nba.sqlite: per-season counts, gaps, spot checks, NULL checks.

python scripts/check_ingestion.py [--db data/nba.sqlite]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

# (player_id, season, label): well-known seasons to compare against Basketball-Reference.
SPOT_CHECKS = [
    ("abdulka01", 1976, "Kareem Abdul-Jabbar 1975-76"),
    ("jordami01", 1988, "Michael Jordan 1987-88"),
    ("jokicni01", 2023, "Nikola Jokic 2022-23"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path("data/nba.sqlite"))
    args = parser.parse_args()
    conn = sqlite3.connect(args.db)

    print("== Seasons (season, player-seasons, traded, final) ==")
    rows = conn.execute(
        "SELECT i.season, i.n_player_seasons, "
        "  (SELECT COUNT(*) FROM player_seasons p WHERE p.season = i.season AND p.team = 'TOT'), "
        "  i.is_final "
        "FROM ingested_seasons i ORDER BY i.season"
    ).fetchall()
    for season, n, traded, final in rows:
        status = "final" if final else "IN PROGRESS"
        print(f"{season - 1}-{str(season)[2:]}  {n:4d}  {traded:3d}  {status}")
    seasons = [r[0] for r in rows]
    if seasons:
        gaps = sorted(set(range(seasons[0], seasons[-1] + 1)) - set(seasons))
        print(f"{len(seasons)} seasons, {seasons[0]}-{seasons[-1]}; missing: {gaps or 'none'}")

    print("\n== Spot checks (compare with basketball-reference.com) ==")
    for player_id, season, label in SPOT_CHECKS:
        row = conn.execute(
            "SELECT s.team, s.g, s.mp, p.pts, p.trb, p.ast, a.ts_pct, a.usg_pct, a.per, a.ws "
            "FROM player_seasons s "
            "LEFT JOIN per100_stats p USING (player_id, season) "
            "LEFT JOIN advanced_stats a USING (player_id, season) "
            "WHERE s.player_id = ? AND s.season = ?",
            (player_id, season),
        ).fetchone()
        if row is None:
            print(f"{label}: NOT FOUND")
            continue
        team, g, mp, pts, trb, ast, ts, usg, per, ws = row
        print(
            f"{label}: {team} G={g} MP={mp} | per100 PTS={pts} TRB={trb} AST={ast} | "
            f"TS%={ts} USG%={usg} PER={per} WS={ws}"
        )

    print("\n== Era gaps should be NULL, never 0 ==")
    checks = [
        ("per100_stats", "fg3", "season < 1980", "3PT before 1979-80"),
        ("per100_stats", "tov", "season < 1978", "TOV before 1977-78"),
        ("advanced_stats", "usg_pct", "season < 1978", "USG% before 1977-78"),
    ]
    for table, col, where, label in checks:
        total, nulls, zeros = conn.execute(
            f"SELECT COUNT(*), SUM({col} IS NULL), SUM({col} = 0) FROM {table} WHERE {where}"
        ).fetchone()
        print(f"{label}: {total} rows, {nulls or 0} NULL, {zeros or 0} zero")


if __name__ == "__main__":
    main()
