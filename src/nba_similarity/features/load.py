"""Reads ingested player-seasons and the raw values of every feature from SQLite."""

from __future__ import annotations

import sqlite3

import pandas as pd

from nba_similarity.features.definitions import FEATURES

IDENTITY_COLUMNS = ("player_id", "name", "season", "age", "pos", "team", "g", "mp")

_TABLE_ALIASES = {"per100_stats": "p", "advanced_stats": "a"}


def load_player_seasons(conn: sqlite3.Connection) -> pd.DataFrame:
    """One row per player-season: identity columns plus each feature's raw value."""
    feature_cols = ", ".join(f"{_TABLE_ALIASES[f.table]}.{f.column} AS {f.name}" for f in FEATURES)
    query = f"""
        SELECT s.player_id, pl.name, s.season, s.age, s.pos, s.team, s.g, s.mp, {feature_cols}
        FROM player_seasons s
        JOIN players pl USING (player_id)
        LEFT JOIN per100_stats p USING (player_id, season)
        LEFT JOIN advanced_stats a USING (player_id, season)
        ORDER BY s.season, s.player_id
    """
    return pd.read_sql_query(query, conn)


def final_seasons(conn: sqlite3.Connection) -> set[int]:
    """Seasons that were complete when ingested (the rest are still in progress)."""
    return {s for (s,) in conn.execute("SELECT season FROM ingested_seasons WHERE is_final = 1")}
