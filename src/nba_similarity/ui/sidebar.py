"""The query sidebar shared by every page (same widget keys, so state survives page switches)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from nba_similarity.similarity.data import FeatureMatrix
from nba_similarity.similarity.explain import season_label
from nba_similarity.ui import cache, view

DEFAULT_PLAYER = "jokicni01"
_STATE_KEY = "_query"


@dataclass(frozen=True)
class Query:
    db: str
    mtime: float
    matrix: FeatureMatrix
    player: view.PlayerOption
    season: int
    n: int
    config_key: cache.ConfigKey
    has_weights: bool  # False when every category weight is 0

    @property
    def row(self) -> int:
        return self.matrix.row(self.player.player_id, self.season)

    @property
    def label(self) -> str:
        return f"{self.player.name} {season_label(self.season)}"


def render() -> Query:
    """Draw the sidebar and return the query; stops the script if there is no data."""
    db_path = Path(os.environ.get("NBA_DB", "data/nba.sqlite"))
    if not db_path.exists():
        st.error(f"Database not found at `{db_path}`. Run `nba-ingest` and `nba-features` first.")
        st.stop()
    db, mtime = str(db_path), db_path.stat().st_mtime
    try:
        matrix = cache.load_matrix(db, mtime)
    except ValueError as exc:
        st.error(f"Could not load the feature matrix: {exc}")
        st.stop()

    options = cache.player_options(db, mtime)
    categories = list(dict.fromkeys(matrix.categories))
    with st.sidebar:
        st.header("Query")
        default = next((i for i, o in enumerate(options) if o.player_id == DEFAULT_PLAYER), 0)
        player = st.selectbox(
            "Player", options, index=default, format_func=lambda o: o.label, key="player"
        )
        season = st.selectbox("Season", player.seasons, format_func=season_label, key="season")

        st.header("Weights")
        st.caption("How much each category counts. 0 ignores it; 2 counts it double.")
        weights = {
            c: st.slider(c.title(), 0.0, 3.0, 1.0, 0.25, key=f"weight_{c}") for c in categories
        }
        include_own = st.toggle("Include this player's other seasons", key="include_own")
        with st.expander("Advanced"):
            n = st.slider("Number of results", 5, 25, 10, key="n")
            metric = st.radio(
                "Distance",
                ["euclidean", "cosine"],
                key="metric",
                help="Euclidean: close stat lines. Cosine: same profile shape, any magnitude.",
            )
            distinct = st.toggle("One season per comp player", value=True, key="distinct")

    config_key: cache.ConfigKey = (
        metric,
        tuple(sorted((c, w) for c, w in weights.items() if w != 1.0)),
        include_own,
        distinct,
    )
    query = Query(db, mtime, matrix, player, season, n, config_key, any(weights.values()))
    st.session_state[_STATE_KEY] = query
    return query


def current() -> Query:
    """The query drawn by the entry script on this run (for page scripts)."""
    return st.session_state[_STATE_KEY]
