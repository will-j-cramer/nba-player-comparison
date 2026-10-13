"""Stores the precomputed map in SQLite and detects when it is out of date.

Tables (rebuilt whole by `nba-map`):
- player_season_map (player_id, season, x, y)
- map_metadata (one row: built_at, n_neighbors, min_dist, seed, features_signature)

`features_signature` hashes the feature table the map was built from; when `nba-features` is
re-run the signature changes and the app reports the map as stale.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np


@dataclass(frozen=True)
class MapParams:
    n_neighbors: int
    min_dist: float
    seed: int


@dataclass(frozen=True)
class StoredMap:
    player_ids: np.ndarray
    seasons: np.ndarray
    xy: np.ndarray  # (n, 2)
    params: MapParams
    built_at: str
    features_signature: str


def features_signature(conn: sqlite3.Connection) -> str:
    """Hash of every (player_id, season, z-score) in player_season_features."""
    z_cols = [r[1] for r in conn.execute("PRAGMA table_info(player_season_features)")]
    z_cols = [c for c in z_cols if c.startswith("z_")]
    if not z_cols:
        return ""
    digest = hashlib.sha256()
    query = (
        f"SELECT player_id, season, {', '.join(z_cols)} FROM player_season_features "
        "ORDER BY player_id, season"
    )
    for row in conn.execute(query):
        digest.update(repr(tuple(round(v, 6) if isinstance(v, float) else v for v in row)).encode())
    return digest.hexdigest()


def write_map(
    conn: sqlite3.Connection,
    player_ids: np.ndarray,
    seasons: np.ndarray,
    xy: np.ndarray,
    params: MapParams,
    signature: str,
) -> None:
    with conn:
        if not conn.in_transaction:
            conn.execute("BEGIN")
        conn.execute("DROP TABLE IF EXISTS player_season_map")
        conn.execute("DROP TABLE IF EXISTS map_metadata")
        conn.execute(
            "CREATE TABLE player_season_map (player_id TEXT NOT NULL, season INTEGER NOT NULL, "
            "x REAL NOT NULL, y REAL NOT NULL, PRIMARY KEY (player_id, season))"
        )
        conn.execute(
            "CREATE TABLE map_metadata (built_at TEXT NOT NULL, n_neighbors INTEGER NOT NULL, "
            "min_dist REAL NOT NULL, seed INTEGER NOT NULL, features_signature TEXT NOT NULL)"
        )
        conn.executemany(
            "INSERT INTO player_season_map VALUES (?, ?, ?, ?)",
            [
                (str(p), int(s), float(x), float(y))
                for p, s, (x, y) in zip(player_ids, seasons, xy, strict=True)
            ],
        )
        conn.execute(
            "INSERT INTO map_metadata VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(timespec="seconds"),
                params.n_neighbors,
                params.min_dist,
                params.seed,
                signature,
            ),
        )


def load_map(conn: sqlite3.Connection) -> StoredMap | None:
    """The stored map, or None if `nba-map` has not been run."""
    try:
        meta = conn.execute(
            "SELECT built_at, n_neighbors, min_dist, seed, features_signature FROM map_metadata"
        ).fetchone()
        rows = conn.execute(
            "SELECT player_id, season, x, y FROM player_season_map ORDER BY season, player_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return None
    if meta is None or not rows:
        return None
    return StoredMap(
        player_ids=np.array([r[0] for r in rows]),
        seasons=np.array([r[1] for r in rows], dtype=int),
        xy=np.array([(r[2], r[3]) for r in rows], dtype=float),
        params=MapParams(n_neighbors=meta[1], min_dist=meta[2], seed=meta[3]),
        built_at=meta[0],
        features_signature=meta[4],
    )


def is_stale(conn: sqlite3.Connection, stored: StoredMap) -> bool:
    """True when the feature table changed since the map was built."""
    return features_signature(conn) != stored.features_signature
