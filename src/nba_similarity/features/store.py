"""Writes the feature matrix and its per-season normalization parameters to SQLite.

The feature tables are derived data, so each build drops and recreates them in a single
transaction: a failed build leaves the previous tables in place.

- feature_definitions: the features, their categories and labels (read by later phases/UI).
- feature_season_stats: per (season, feature) mean, sd, coverage and whether it is covered.
- player_season_features: one row per qualified player-season with each feature's raw value
  and its clipped z-score (z_<feature>, NULL when missing or not covered that season).
  It deliberately has no foreign key to player_seasons: ingestion rewrites seasons (e.g. the
  one in progress) after features exist, and the features are rebuilt afterwards anyway.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import pandas as pd

from nba_similarity.features.definitions import FEATURE_NAMES, FEATURES, z_column
from nba_similarity.features.normalize import SEASON_STATS_COLUMNS

FEATURE_ROW_COLUMNS: tuple[str, ...] = (
    "player_id",
    "season",
    "mp",
    "min_minutes",
    "n_features",
    *FEATURE_NAMES,
    *(z_column(f) for f in FEATURE_NAMES),
)


def _ddl() -> str:
    raw = ",\n    ".join(f"{f} REAL" for f in FEATURE_NAMES)
    z = ",\n    ".join(f"{z_column(f)} REAL" for f in FEATURE_NAMES)
    return f"""
DROP TABLE IF EXISTS player_season_features;
DROP TABLE IF EXISTS feature_season_stats;
DROP TABLE IF EXISTS feature_definitions;

CREATE TABLE feature_definitions (
    feature  TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    label    TEXT NOT NULL,
    source   TEXT NOT NULL,
    ordinal  INTEGER NOT NULL
);

CREATE TABLE feature_season_stats (
    season   INTEGER NOT NULL,
    feature  TEXT NOT NULL REFERENCES feature_definitions (feature),
    n        INTEGER NOT NULL,
    mean     REAL,
    sd       REAL,
    coverage REAL NOT NULL,
    covered  INTEGER NOT NULL,
    PRIMARY KEY (season, feature)
);

CREATE TABLE player_season_features (
    player_id   TEXT NOT NULL,
    season      INTEGER NOT NULL,
    mp          INTEGER,
    min_minutes INTEGER NOT NULL,
    n_features  INTEGER NOT NULL,
    {raw},
    {z},
    PRIMARY KEY (player_id, season)
);
CREATE INDEX idx_player_season_features_season ON player_season_features (season);
"""


def write_features(conn: sqlite3.Connection, features: pd.DataFrame, stats: pd.DataFrame) -> None:
    """Replace the feature tables with `features` (one row per player-season) and `stats`."""
    statements = [s for s in _ddl().split(";") if s.strip()]
    with conn:
        if not conn.in_transaction:
            conn.execute("BEGIN")  # sqlite3 does not open a transaction for DDL on its own
        for statement in statements:
            conn.execute(statement)
        conn.executemany(
            "INSERT INTO feature_definitions VALUES (?, ?, ?, ?, ?)",
            [
                (f.name, f.category, f.label, f"{f.table}.{f.column}", i)
                for i, f in enumerate(FEATURES)
            ],
        )
        _insert(conn, "feature_season_stats", SEASON_STATS_COLUMNS, stats)
        _insert(conn, "player_season_features", FEATURE_ROW_COLUMNS, features)


def _insert(conn: sqlite3.Connection, table: str, columns: Sequence[str], df: pd.DataFrame) -> None:
    placeholders = ", ".join("?" * len(columns))
    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df[list(columns)].astype(object).itertuples(index=False, name=None)
    ]
    conn.executemany(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})", rows)
