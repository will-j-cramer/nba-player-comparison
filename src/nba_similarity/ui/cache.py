"""Cached loaders shared by the app's pages (streamlit caching around pure functions).

Every key includes the database file's modification time, so re-running `nba-features` or
`nba-map` refreshes the app without a restart.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import streamlit as st

from nba_similarity.careers.data import CareerData, load_career_data
from nba_similarity.careers.engine import CareerConfig, CareerEngine, CareerResult
from nba_similarity.mapping.cluster import cluster_names, kmeans_labels
from nba_similarity.mapping.store import is_stale, load_map
from nba_similarity.similarity.config import SimilarityConfig
from nba_similarity.similarity.data import FeatureMatrix, load_feature_matrix
from nba_similarity.similarity.engine import Match, SimilarityEngine
from nba_similarity.ui import view

ConfigKey = tuple[str, tuple[tuple[str, float], ...], bool, bool]


@st.cache_resource(show_spinner="Loading feature matrix…")
def load_matrix(db_path: str, mtime: float) -> FeatureMatrix:
    """The feature matrix, loaded once per database version."""
    conn = sqlite3.connect(db_path)
    try:
        return load_feature_matrix(conn)
    finally:
        conn.close()


@st.cache_resource(max_entries=32)
def get_engine(db_path: str, mtime: float, config_key: ConfigKey) -> SimilarityEngine:
    """One SimilarityEngine per settings combination."""
    metric, weights, include_same_player, distinct_players = config_key
    config = SimilarityConfig(
        metric=metric,  # type: ignore[arg-type]
        category_weights=dict(weights),
        include_same_player=include_same_player,
        distinct_players=distinct_players,
    )
    return SimilarityEngine(load_matrix(db_path, mtime), config)


@st.cache_data(max_entries=256, show_spinner=False)
def search(
    db_path: str, mtime: float, config_key: ConfigKey, player_id: str, season: int, n: int
) -> list[Match]:
    """Top-n matches for a player-season under the given settings."""
    return get_engine(db_path, mtime, config_key).find_similar(player_id, season, n=n)


@st.cache_data(show_spinner=False)
def player_options(db_path: str, mtime: float) -> list[view.PlayerOption]:
    """The player dropdown's options."""
    return view.player_options(load_matrix(db_path, mtime))


@st.cache_data(show_spinner=False)
def league_averages(db_path: str, mtime: float) -> view.LeagueAverages:
    """Observed league average per (season, feature)."""
    conn = sqlite3.connect(db_path)
    try:
        return view.load_league_averages(conn)
    finally:
        conn.close()


@st.cache_data(show_spinner="Loading map…")
def map_layout(db_path: str, mtime: float) -> view.MapLayout | None:
    """The precomputed map aligned to the feature matrix, or None before `nba-map` has run."""
    conn = sqlite3.connect(db_path)
    try:
        stored = load_map(conn)
        if stored is None:
            return None
        stale = is_stale(conn, stored)
    finally:
        conn.close()
    xy = view.align_map(load_matrix(db_path, mtime), stored.player_ids, stored.seasons, stored.xy)
    return view.MapLayout(xy=xy, built_at=stored.built_at, stale=stale)


@st.cache_data(max_entries=16, show_spinner=False)
def map_clusters(db_path: str, mtime: float, k: int) -> tuple[np.ndarray, dict[int, str]]:
    """k-means labels on the map (-1 = no position) and a name for each cluster."""
    layout = map_layout(db_path, mtime)
    matrix = load_matrix(db_path, mtime)
    assert layout is not None
    placed = np.isfinite(layout.xy).all(axis=1)
    labels = np.full(len(placed), -1)
    labels[placed] = kmeans_labels(layout.xy[placed], k)
    names = cluster_names(
        matrix.z[placed],
        labels[placed],
        [
            view.SHORT_LABELS.get(f, lbl)
            for f, lbl in zip(matrix.features, matrix.labels, strict=True)
        ],
        include=matrix.in_distance,
    )
    return labels, names


CareerKey = tuple[int, int, tuple[tuple[str, float], ...]]  # window, horizon, weights


@st.cache_resource(show_spinner="Loading careers…")
def load_careers(db_path: str, mtime: float) -> CareerData:
    """The career data (ValueError when nba-features hasn't built the career tables)."""
    conn = sqlite3.connect(db_path)
    try:
        return load_career_data(conn)
    finally:
        conn.close()


@st.cache_resource(max_entries=16)
def get_career_engine(db_path: str, mtime: float, key: CareerKey) -> CareerEngine:
    """One CareerEngine per settings combination."""
    window, horizon, weights = key
    config = CareerConfig(window=window, horizon=horizon, category_weights=dict(weights))
    return CareerEngine(load_careers(db_path, mtime), config)


@st.cache_data(max_entries=128, show_spinner="Comparing careers…")
def career_comps(
    db_path: str, mtime: float, key: CareerKey, player_id: str, through_age: int
) -> CareerResult:
    """Career comps and projection for a player through an age."""
    return get_career_engine(db_path, mtime, key).find_comps(player_id, through_age)
