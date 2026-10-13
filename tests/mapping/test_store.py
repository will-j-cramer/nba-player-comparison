from __future__ import annotations

import sqlite3

import numpy as np

from nba_similarity.features.pipeline import build_features
from nba_similarity.mapping.store import (
    MapParams,
    features_signature,
    is_stale,
    load_map,
    write_map,
)


def _write(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT player_id, season FROM player_season_features").fetchall()
    ids = np.array([r[0] for r in rows])
    seasons = np.array([r[1] for r in rows])
    xy = np.column_stack([np.arange(len(rows)), -np.arange(len(rows))]).astype(float)
    write_map(conn, ids, seasons, xy, MapParams(15, 0.1, 42), features_signature(conn))


def test_round_trip_and_staleness(ingested_db: sqlite3.Connection) -> None:
    conn = ingested_db
    build_features(conn)
    assert load_map(conn) is None  # nba-map not run yet
    _write(conn)
    stored = load_map(conn)
    assert stored is not None
    n = conn.execute("SELECT COUNT(*) FROM player_season_features").fetchone()[0]
    assert stored.xy.shape == (n, 2) and len(stored.player_ids) == n
    assert stored.params == MapParams(15, 0.1, 42)
    assert not is_stale(conn, stored)

    conn.execute("UPDATE player_season_features SET z_ast_pct = z_ast_pct + 1")
    assert is_stale(conn, stored)


def test_rewrite_replaces_previous_map(ingested_db: sqlite3.Connection) -> None:
    build_features(ingested_db)
    _write(ingested_db)
    _write(ingested_db)
    stored = load_map(ingested_db)
    n = ingested_db.execute("SELECT COUNT(*) FROM player_season_features").fetchone()[0]
    assert stored is not None and len(stored.xy) == n


def test_signature_empty_without_feature_table() -> None:
    assert features_signature(sqlite3.connect(":memory:")) == ""
