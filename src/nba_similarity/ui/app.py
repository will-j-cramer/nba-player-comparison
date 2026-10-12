"""Streamlit app: `streamlit run src/nba_similarity/ui/app.py` (database: $NBA_DB).

Layout and caching only. Comparisons come from the similarity engine, tables and series from
ui/view.py, figures from ui/charts.py. The app reads SQLite and never scrapes or recomputes
features; re-run `nba-features` and the cache refreshes itself (the database's modification
time is part of every cache key).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import streamlit as st

from nba_similarity.similarity.config import SimilarityConfig
from nba_similarity.similarity.data import FeatureMatrix, load_feature_matrix
from nba_similarity.similarity.engine import Match, SimilarityEngine
from nba_similarity.similarity.explain import format_contribution, season_label
from nba_similarity.ui import charts, view

DEFAULT_PLAYER = "jokicni01"
ConfigKey = tuple[str, tuple[tuple[str, float], ...], bool, bool]


@st.cache_resource(show_spinner="Loading feature matrix…")
def load_matrix(db_path: str, mtime: float) -> FeatureMatrix:
    conn = sqlite3.connect(db_path)
    try:
        return load_feature_matrix(conn)
    finally:
        conn.close()


@st.cache_resource(max_entries=32)
def get_engine(db_path: str, mtime: float, config_key: ConfigKey) -> SimilarityEngine:
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
    return get_engine(db_path, mtime, config_key).find_similar(player_id, season, n=n)


@st.cache_data(show_spinner=False)
def player_options(db_path: str, mtime: float) -> list[view.PlayerOption]:
    return view.player_options(load_matrix(db_path, mtime))


@st.cache_data(show_spinner=False)
def league_averages(db_path: str, mtime: float) -> view.LeagueAverages:
    conn = sqlite3.connect(db_path)
    try:
        return view.load_league_averages(conn)
    finally:
        conn.close()


def theme_type() -> str:
    theme = getattr(st.context, "theme", None)
    return getattr(theme, "type", None) or "light"


def main() -> None:
    st.set_page_config(page_title="NBA Player Similarity", page_icon="🏀", layout="wide")
    st.title("NBA player-season similarity")

    db_path = Path(os.environ.get("NBA_DB", "data/nba.sqlite"))
    if not db_path.exists():
        st.error(f"Database not found at `{db_path}`. Run `nba-ingest` and `nba-features` first.")
        st.stop()
    db, mtime = str(db_path), db_path.stat().st_mtime
    try:
        matrix = load_matrix(db, mtime)
    except ValueError as exc:
        st.error(f"Could not load the feature matrix: {exc}")
        st.stop()

    options = player_options(db, mtime)
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

    if not any(weights.values()):
        st.warning("All category weights are 0; raise at least one to compare players.")
        st.stop()

    config_key: ConfigKey = (
        metric,
        tuple(sorted((c, w) for c, w in weights.items() if w != 1.0)),
        include_own,
        distinct,
    )
    query_row = matrix.row(player.player_id, season)
    matches = search(db, mtime, config_key, player.player_id, season, n)

    st.subheader(f"Most similar to {player.name} {season_label(season)}")
    st.caption(f"Standout stats: {view.query_profile(matrix, query_row)}")
    if not matches:
        st.info("No comparable player-seasons found with these settings.")
        st.stop()

    event = st.dataframe(
        view.results_frame(matches),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="results",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Score": st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%.1f", help="100 = identical"
            ),
            "Stats shared": st.column_config.TextColumn(
                help="Share of the stat weight both seasons have (pre-1980 seasons lack some)"
            ),
        },
    )
    selected = event.selection.rows if event and event.selection else []
    match = matches[selected[0]] if selected and selected[0] < len(matches) else matches[0]
    if not selected:
        st.caption("Select a row to compare it; showing the top match.")

    close_z = get_engine(db, mtime, config_key).config.close_z
    render_comparison(matrix, query_row, player, season, match, league_averages(db, mtime), close_z)


def render_comparison(
    matrix: FeatureMatrix,
    query_row: int,
    player: view.PlayerOption,
    season: int,
    match: Match,
    league: view.LeagueAverages,
    close_z: float,
) -> None:
    query_label = f"{player.name} {season_label(season)}"
    match_label = f"{match.name} {season_label(match.season)}"
    st.subheader(f"{query_label} vs {match_label} · score {match.score:.1f}")
    if match.similar_traits:
        st.markdown("**Alike:** " + "; ".join(format_contribution(c) for c in match.similar_traits))
    if match.differences:
        st.markdown("**Differ:** " + "; ".join(format_contribution(c) for c in match.differences))
    if match.shared < 0.999:
        st.caption(
            f"Only {match.shared:.0%} of the stat weight could be compared (some stats were not "
            "tracked that season); missing stats count as typical, never as zero."
        )

    left, right = st.columns([4, 3], gap="medium")
    with left:
        table = view.comparison_frame(matrix, query_row, match, league, close_z)
        st.dataframe(
            table,
            hide_index=True,
            height=35 * (len(table) + 1) + 3,  # every stat visible, no inner scroll
            column_config={
                "Stat": st.column_config.TextColumn(width="medium"),
                "Query": st.column_config.NumberColumn(
                    view.short_label(player.name, season), format="%.3g", help=query_label
                ),
                "Match": st.column_config.NumberColumn(
                    view.short_label(match.name, match.season), format="%.3g", help=match_label
                ),
                "Lg avg (query yr)": st.column_config.NumberColumn(
                    f"Lg '{str(season)[2:]}",
                    format="%.3g",
                    help=f"League average in {season_label(season)}",
                ),
                "Lg avg (match yr)": st.column_config.NumberColumn(
                    f"Lg '{str(match.season)[2:]}",
                    format="%.3g",
                    help=f"League average in {season_label(match.season)}",
                ),
                "Verdict": st.column_config.TextColumn(width="small"),
                "Points": st.column_config.NumberColumn(
                    "Pts", format="%+.1f", help="Contribution to the score; all stats sum to it"
                ),
            },
        )
    with right:
        data = view.radar_data(matrix, query_row, matrix.row(match.player_id, match.season))
        st.plotly_chart(charts.radar_figure(data, query_label, match_label, theme_type()))
        st.caption(
            "Percentile within each player's own season (50 = league average), so eras compare "
            "fairly. Hover a point for the raw value; gaps are stats not tracked that season."
        )


main()
