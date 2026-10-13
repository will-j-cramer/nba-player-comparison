"""SQLite schema for ingested Basketball-Reference data.

- `season` is the season's end year (2023 = 2022-23).
- `per100_stats` and `advanced_stats` are 1:1 with `player_seasons`, mirroring the two source
  pages. Stats a season did not track (e.g. 3PT before 1979-80, TOV before 1977-78) are NULL,
  never 0. Values are stored as B-Ref reports them (see records.py).
- Traded players have a single `player_seasons` row with team 'TOT'; their per-team stints
  are in `player_season_teams`.
- `ingested_seasons` records which seasons are fully loaded; `is_final = 0` marks a season
  still in progress, which is refetched on the next run.
"""

from __future__ import annotations

import sqlite3

from nba_similarity.ingestion.records import ADVANCED_COLUMNS, PER100_COLUMNS


def _stats_table(name: str, columns: tuple[str, ...]) -> str:
    stat_cols = ",\n    ".join(f"{col} REAL" for col in columns)
    return f"""
CREATE TABLE IF NOT EXISTS {name} (
    player_id TEXT NOT NULL,
    season    INTEGER NOT NULL,
    {stat_cols},
    PRIMARY KEY (player_id, season),
    FOREIGN KEY (player_id, season) REFERENCES player_seasons (player_id, season)
);"""


DDL = f"""
CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    name      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS player_seasons (
    player_id TEXT NOT NULL REFERENCES players (player_id),
    season    INTEGER NOT NULL,
    age       INTEGER,
    pos       TEXT,
    team      TEXT NOT NULL,
    n_teams   INTEGER NOT NULL,
    g         INTEGER,
    gs        INTEGER,
    mp        INTEGER,
    PRIMARY KEY (player_id, season)
);
CREATE INDEX IF NOT EXISTS idx_player_seasons_season ON player_seasons (season);

CREATE TABLE IF NOT EXISTS player_season_teams (
    player_id TEXT NOT NULL,
    season    INTEGER NOT NULL,
    stint     INTEGER NOT NULL,
    team      TEXT NOT NULL,
    g         INTEGER,
    mp        INTEGER,
    PRIMARY KEY (player_id, season, stint),
    FOREIGN KEY (player_id, season) REFERENCES player_seasons (player_id, season)
);
{_stats_table("per100_stats", PER100_COLUMNS)}
{_stats_table("advanced_stats", ADVANCED_COLUMNS)}

CREATE TABLE IF NOT EXISTS ingested_seasons (
    season           INTEGER PRIMARY KEY,
    is_final         INTEGER NOT NULL,
    n_player_seasons INTEGER NOT NULL,
    ingested_at      TEXT NOT NULL
);
"""


def connect(path: str) -> sqlite3.Connection:
    """Open the database with foreign keys enforced and the schema created."""
    conn = sqlite3.connect(path)
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the ingestion tables if they don't exist (and enable foreign keys)."""
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(DDL)
