"""Builds the feature matrix: load -> qualify -> per-season stats -> z-scores -> store."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from nba_similarity.features.load import final_seasons, load_player_seasons
from nba_similarity.features.normalize import add_zscores, season_stats
from nba_similarity.features.qualify import qualify
from nba_similarity.features.store import write_features


@dataclass(frozen=True)
class FeatureReport:
    n_player_seasons: int
    n_qualified: int
    n_seasons: int


def build_features(conn: sqlite3.Connection) -> FeatureReport:
    """Recompute and store the feature tables from the ingested data."""
    raw = load_player_seasons(conn)
    if raw.empty:
        raise ValueError("no player-seasons in the database; run nba-ingest first")
    qualified = qualify(raw, final_seasons(conn))
    stats = season_stats(qualified)
    features = add_zscores(qualified, stats)
    write_features(conn, features, stats)
    return FeatureReport(
        n_player_seasons=len(raw),
        n_qualified=len(features),
        n_seasons=features["season"].nunique(),
    )
