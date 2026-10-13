"""The query sidebar shared by every page (same widget keys, so state survives page switches)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from nba_similarity.paths import default_db
from nba_similarity.similarity.data import FeatureMatrix
from nba_similarity.similarity.explain import season_label
from nba_similarity.ui import cache, view
from nba_similarity.ui.remote_db import DownloadError, download_path, ensure_db

DEFAULT_PLAYER = "jokicni01"
_STATE_KEY = "_query"


@dataclass(frozen=True)
class Query:
    """Everything the pages need about the current selection and settings."""

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
        """The query's row in the feature matrix."""
        return self.matrix.row(self.player.player_id, self.season)

    @property
    def label(self) -> str:
        """Display label, e.g. "Nikola Jokić 2022-23"."""
        return f"{self.player.name} {season_label(self.season)}"


@st.cache_resource(show_spinner="Downloading the database…")
def fetch_db(path: str, url: str | None, token: str | None) -> Path | None:
    """The downloaded database (see remote_db.download_path), fetched once per process; None
    without a URL."""
    if not url:
        return None
    return ensure_db(download_path(Path(path), url), url, token)


def _secrets() -> tuple[str | None, str | None]:
    """NBA_DB_URL and NBA_DB_TOKEN from Streamlit secrets (None without a secrets file)."""
    try:
        return st.secrets.get("NBA_DB_URL"), st.secrets.get("NBA_DB_TOKEN")
    except FileNotFoundError:
        return None, None


def render() -> Query:
    """Draw the sidebar and return the query; stops the script if there is no data."""
    db_path = default_db()
    try:
        found = db_path if db_path.exists() else fetch_db(str(db_path), *_secrets())
    except DownloadError as exc:
        st.error(f"Could not download the database: {exc}")
        st.stop()
    if found is None:
        st.error(
            f"Database not found at `{db_path}`. Run `nba-ingest` and `nba-features` first, or "
            "set `NBA_DB_URL` in the app's secrets to download one (see the README)."
        )
        st.stop()
    db, mtime = str(found), found.stat().st_mtime
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
