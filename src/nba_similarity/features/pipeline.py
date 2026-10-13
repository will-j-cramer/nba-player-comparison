"""Builds the feature matrix: load -> qualify -> basis -> per-season stats -> z -> store."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from nba_similarity.features.definitions import FEATURE_NAMES, basis_column
from nba_similarity.features.load import final_seasons, load_player_seasons
from nba_similarity.features.normalize import add_zscores, season_stats
from nba_similarity.features.qualify import qualify
from nba_similarity.features.store import write_features
from nba_similarity.features.transforms import add_basis

_BASIS = {f: basis_column(f) for f in FEATURE_NAMES}


@dataclass(frozen=True)
class FeatureReport:
    """Counts from a feature build."""

    n_player_seasons: int
    n_qualified: int
    n_seasons: int


def build_features(conn: sqlite3.Connection) -> FeatureReport:
    """Recompute and store the feature tables from the ingested data."""
    raw = load_player_seasons(conn)
    if raw.empty:
        raise ValueError("no player-seasons in the database; run nba-ingest first")
    qualified = add_basis(qualify(raw, final_seasons(conn)))
    stats = season_stats(qualified, source=_BASIS)
    features = add_zscores(qualified, stats, source=_BASIS)
    write_features(conn, features, stats)
    return FeatureReport(
        n_player_seasons=len(raw),
        n_qualified=len(features),
        n_seasons=features["season"].nunique(),
    )
