"""Search page: the most similar player-seasons and a side-by-side comparison."""

from __future__ import annotations

import streamlit as st

from nba_similarity.similarity.data import FeatureMatrix
from nba_similarity.similarity.engine import Match
from nba_similarity.similarity.explain import format_contribution, season_label
from nba_similarity.ui import cache, charts, sidebar, view
from nba_similarity.ui.theme import theme_type


def render(query: sidebar.Query) -> None:
    """Draw the results table and the comparison for the sidebar's query."""
    st.title("NBA player-season similarity")
    if not query.has_weights:
        st.warning("All category weights are 0; raise at least one to compare players.")
        st.stop()

    matrix, player, season = query.matrix, query.player, query.season
    matches = cache.search(
        query.db, query.mtime, query.config_key, player.player_id, season, query.n
    )
    st.subheader(f"Most similar to {query.label}")
    st.caption(f"Standout stats: {view.query_profile(matrix, query.row)}")
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

    close_z = cache.get_engine(query.db, query.mtime, query.config_key).config.close_z
    league = cache.league_averages(query.db, query.mtime)
    render_comparison(matrix, query.row, player, season, match, league, close_z)


def render_comparison(
    matrix: FeatureMatrix,
    query_row: int,
    player: view.PlayerOption,
    season: int,
    match: Match,
    league: view.LeagueAverages,
    close_z: float,
) -> None:
    """Side-by-side table and radar chart for the query and one match."""
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


if __name__ == "__main__":
    render(sidebar.current())
