"""Writes ingested seasons to SQLite and reports which seasons are already loaded."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime

from nba_similarity.ingestion.records import ADVANCED_COLUMNS, PER100_COLUMNS, PlayerSeason


def write_season(
    conn: sqlite3.Connection,
    season: int,
    player_seasons: Sequence[PlayerSeason],
    *,
    is_final: bool,
    now: datetime | None = None,
) -> None:
    """Replace everything stored for `season` in a single transaction.

    A failure part-way leaves the previous contents (or nothing) in place, and the season is
    only recorded in `ingested_seasons` once all its rows are written.
    """
    ingested_at = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    with conn:
        for table in ("per100_stats", "advanced_stats", "player_season_teams", "player_seasons"):
            conn.execute(f"DELETE FROM {table} WHERE season = ?", (season,))
        conn.executemany(
            "INSERT INTO players (player_id, name) VALUES (?, ?) "
            "ON CONFLICT (player_id) DO UPDATE SET name = excluded.name",
            [(ps.player_id, ps.name) for ps in player_seasons],
        )
        conn.executemany(
            "INSERT INTO player_seasons "
            "(player_id, season, age, pos, team, n_teams, g, gs, mp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (ps.player_id, season, ps.age, ps.pos, ps.team, ps.n_teams, ps.g, ps.gs, ps.mp)
                for ps in player_seasons
            ],
        )
        conn.executemany(
            "INSERT INTO player_season_teams (player_id, season, stint, team, g, mp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (ps.player_id, season, i, stint.team, stint.g, stint.mp)
                for ps in player_seasons
                for i, stint in enumerate(ps.stints, start=1)
            ],
        )
        _insert_stats(conn, "per100_stats", PER100_COLUMNS, season, player_seasons, "per100")
        _insert_stats(conn, "advanced_stats", ADVANCED_COLUMNS, season, player_seasons, "advanced")
        conn.execute(
            "INSERT INTO ingested_seasons (season, is_final, n_player_seasons, ingested_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT (season) DO UPDATE SET is_final = excluded.is_final, "
            "n_player_seasons = excluded.n_player_seasons, ingested_at = excluded.ingested_at",
            (season, int(is_final), len(player_seasons), ingested_at),
        )


def ingested_seasons(conn: sqlite3.Connection) -> dict[int, bool]:
    """Map each loaded season to whether it was final (complete) when loaded."""
    rows = conn.execute("SELECT season, is_final FROM ingested_seasons").fetchall()
    return {season: bool(is_final) for season, is_final in rows}


def _insert_stats(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    season: int,
    player_seasons: Sequence[PlayerSeason],
    attr: str,
) -> None:
    names = ", ".join(("player_id", "season", *columns))
    placeholders = ", ".join("?" * (len(columns) + 2))
    rows = []
    for ps in player_seasons:
        stats = getattr(ps, attr)
        if stats is not None:
            rows.append((ps.player_id, season, *(stats[col] for col in columns)))
    conn.executemany(f"INSERT INTO {table} ({names}) VALUES ({placeholders})", rows)
