"""Writes the feature matrix and its per-season normalization parameters to SQLite.

The feature tables are derived data, so each build drops and recreates them in a single
transaction: a failed build leaves the previous tables in place.

- feature_definitions: the features, their categories, labels, basis transform and whether
  they count in the similarity distance (read by later phases and the UI).
- feature_season_stats: per (season, feature) mean and sd of the basis value that was
  z-scored, coverage, whether it is covered, and raw_mean (league average as observed).
- player_season_features: one row per qualified player-season with each feature's raw
  (observed) value and its clipped z-score (z_<feature>, NULL when missing or not covered
  that season).
  It deliberately has no foreign key to player_seasons: ingestion rewrites seasons (e.g. the
  one in progress) after features exist, and the features are rebuilt afterwards anyway.
- player_season_career_features: every ingested player-season for career trajectories, with
  a status (qualified / partial / limited), reliability, minutes per game and z-scores
  against the qualified statistics (see career.py).
- career_players: birth offset (season - age), first season and age, and how far back the
  career may be missing (censored_through_age, censor).
- career_season_stats: per-season mean/SD of minutes per game over qualified players.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence

import pandas as pd

from nba_similarity.features.career import (
    CAREER_PLAYER_COLUMNS,
    CAREER_ROW_COLUMNS,
    MPG_STATS_COLUMNS,
)
from nba_similarity.features.definitions import FEATURE_NAMES, FEATURES, MPG_FEATURE, z_column
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
    feature   TEXT PRIMARY KEY,
    category  TEXT NOT NULL,
    label     TEXT NOT NULL,
    source    TEXT NOT NULL,
    transform TEXT NOT NULL,
    in_distance INTEGER NOT NULL,
    ordinal   INTEGER NOT NULL
);

CREATE TABLE feature_season_stats (
    season   INTEGER NOT NULL,
    feature  TEXT NOT NULL REFERENCES feature_definitions (feature),
    n        INTEGER NOT NULL,
    mean     REAL,
    sd       REAL,
    coverage REAL NOT NULL,
    covered  INTEGER NOT NULL,
    raw_mean REAL,
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


def _career_ddl() -> str:
    stats = (*FEATURE_NAMES, MPG_FEATURE)
    raw = ",\n    ".join(f"{f} REAL" for f in stats)
    z = ",\n    ".join(f"{z_column(f)} REAL" for f in stats)
    return f"""
DROP TABLE IF EXISTS player_season_career_features;
DROP TABLE IF EXISTS career_players;
DROP TABLE IF EXISTS career_season_stats;

CREATE TABLE player_season_career_features (
    player_id          TEXT NOT NULL,
    season             INTEGER NOT NULL,
    age                INTEGER,
    g                  INTEGER,
    mp                 INTEGER,
    min_minutes        INTEGER NOT NULL,
    career_min_minutes INTEGER NOT NULL,
    status             TEXT NOT NULL CHECK (status IN ('qualified', 'partial', 'limited')),
    reliability        REAL NOT NULL,
    {raw},
    {z},
    PRIMARY KEY (player_id, season)
);

CREATE TABLE career_players (
    player_id            TEXT PRIMARY KEY,
    birth_offset         INTEGER NOT NULL,
    first_season         INTEGER NOT NULL,
    first_age            INTEGER NOT NULL,
    censored_through_age INTEGER,
    censor               TEXT NOT NULL
);

CREATE TABLE career_season_stats (
    season INTEGER PRIMARY KEY,
    mean   REAL,
    sd     REAL
);
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
            "INSERT INTO feature_definitions VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    f.name,
                    f.category,
                    f.label,
                    f"{f.table}.{f.column}",
                    f.transform,
                    int(f.in_distance),
                    i,
                )
                for i, f in enumerate(FEATURES)
            ],  # fmt: skip
        )
        _insert(conn, "feature_season_stats", SEASON_STATS_COLUMNS, stats)
        _insert(conn, "player_season_features", FEATURE_ROW_COLUMNS, features)


def write_career_features(
    conn: sqlite3.Connection, rows: pd.DataFrame, players: pd.DataFrame, mpg_stats: pd.DataFrame
) -> None:
    """Replace the career tables (see career.py) in a single transaction."""
    statements = [s for s in _career_ddl().split(";") if s.strip()]
    with conn:
        if not conn.in_transaction:
            conn.execute("BEGIN")
        for statement in statements:
            conn.execute(statement)
        _insert(conn, "player_season_career_features", CAREER_ROW_COLUMNS, rows)
        _insert(conn, "career_players", CAREER_PLAYER_COLUMNS, players)
        _insert(conn, "career_season_stats", MPG_STATS_COLUMNS, mpg_stats)


def _insert(conn: sqlite3.Connection, table: str, columns: Sequence[str], df: pd.DataFrame) -> None:
    placeholders = ", ".join("?" * len(columns))
    rows = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df[list(columns)].astype(object).itertuples(index=False, name=None)
    ]
    conn.executemany(f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})", rows)
