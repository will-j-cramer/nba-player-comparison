"""Builds the feature matrix: load -> qualify -> basis -> per-season stats -> z -> store."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

from nba_similarity.features.career import career_players, career_rows
from nba_similarity.features.definitions import FEATURE_NAMES, basis_column
from nba_similarity.features.load import final_seasons, load_player_seasons
from nba_similarity.features.normalize import add_zscores, season_stats
from nba_similarity.features.qualify import qualify
from nba_similarity.features.store import write_career_features, write_features
from nba_similarity.features.transforms import add_basis

_BASIS = {f: basis_column(f) for f in FEATURE_NAMES}


@dataclass(frozen=True)
class FeatureReport:
    """Counts from a feature build."""

    n_player_seasons: int
    n_qualified: int
    n_seasons: int
    n_career_scored: int = 0  # qualified + partial seasons usable in career trajectories
    aba_rosters_loaded: bool = False  # False: ABA-start careers can't be flagged yet


def build_features(conn: sqlite3.Connection) -> FeatureReport:
    """Recompute and store the feature tables from the ingested data."""
    raw = load_player_seasons(conn)
    if raw.empty:
        raise ValueError("no player-seasons in the database; run nba-ingest first")
    finals = final_seasons(conn)
    qualified = add_basis(qualify(raw, finals))
    stats = season_stats(qualified, source=_BASIS)
    features = add_zscores(qualified, stats, source=_BASIS)
    write_features(conn, features, stats)

    rows, mpg_stats = career_rows(raw, qualified, stats, finals)
    aba = load_aba_rosters(conn)
    write_career_features(conn, rows, career_players(raw, aba), mpg_stats)
    return FeatureReport(
        n_player_seasons=len(raw),
        n_qualified=len(features),
        n_seasons=features["season"].nunique(),
        n_career_scored=int((rows["status"] != "limited").sum()),
        aba_rosters_loaded=aba is not None and len(aba) > 0,
    )


def load_aba_rosters(conn: sqlite3.Connection) -> pd.DataFrame | None:
    """The ABA rosters fetched by nba-ingest, or None before they have been fetched."""
    try:
        return pd.read_sql_query("SELECT player_id, season FROM aba_rosters", conn)
    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        return None
