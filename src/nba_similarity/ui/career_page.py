"""Careers page: whose careers through the same age looked most alike, and what came next."""

from __future__ import annotations

import streamlit as st

from nba_similarity.seasons import season_label
from nba_similarity.similarity.explain import SHORT_LABELS
from nba_similarity.ui import cache, charts, sidebar, view
from nba_similarity.ui.theme import theme_type

DEFAULT_STAT = "usg_pct"
TABLE_STATS = ("usg_pct", "ts_pct", "ast_pct", "drb_pct", "blk_pct", "fg3a", "mpg")


def render(query: sidebar.Query) -> None:
    """Draw the career comps for the sidebar's player."""
    st.title("Career trajectory comps")
    try:
        data = cache.load_careers(query.db, query.mtime)
    except ValueError as exc:
        st.info(f"Career data isn't built yet ({exc}).")
        st.stop()
    player = query.player
    ages = view.career_ages(data, player.player_id)
    if not ages:
        st.info(f"{player.name} has no season of at least 250 minutes.")
        st.stop()

    left, mid, right = st.columns([1, 1, 1])
    with left:
        through_age = st.selectbox("Through age", ages, key="career_age")
    with mid:
        window = st.slider("Ages compared", 1, 5, 3, key="career_window")
    with right:
        horizon = st.slider("Seasons projected", 1, 8, 5, key="career_horizon")
    weights = dict(query.config_key[1])  # the sidebar's category weights
    key: cache.CareerKey = (window, horizon, tuple(sorted(weights.items())))
    result = cache.career_comps(query.db, query.mtime, key, player.player_id, through_age)

    seasons = ", ".join(season_label(s) for s in result.query_seasons if s is not None)
    st.subheader(f"Careers most like {result.name}'s through age {through_age}")
    st.caption(
        f"Ages {result.ages[0]}–{result.ages[-1]} ({seasons}); "
        f"{result.query_experience} NBA season(s) so far. Each age is compared with the same "
        "age, weighted toward the latest; minutes per game count as one stat."
    )
    if not result.comps:
        st.info("No comparable careers at these ages.")
        st.stop()

    event = st.dataframe(
        view.career_comps_frame(result, horizon),
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="career_results",
        column_config={
            "Rank": st.column_config.NumberColumn(width="small"),
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f"),
            "Seasons in": st.column_config.NumberColumn(help="NBA seasons through that age"),
            "Window": st.column_config.TextColumn(
                help="Share of the window both careers have (a missing season counts as typical)"
            ),
        },
    )
    st.caption(
        "● full season  |  ◐ played, under 500 minutes  |  ○ not in the NBA (injury, G League, "
        "overseas or retired: our data can't tell)  |  · not played yet"
    )
    selected = event.selection.rows if event and event.selection else []
    comp = result.comps[selected[0] if selected and selected[0] < len(result.comps) else 0]

    stat = st.selectbox(
        "Stat",
        data.stats,
        index=data.stats.index(DEFAULT_STAT),
        format_func=lambda s: data.labels[data.stats.index(s)],
        key="career_stat",
    )
    future = list(range(through_age + 1, through_age + horizon + 1))
    span = list(result.ages) + future
    figure = charts.trajectory_figure(
        view.trajectory_series(data, player.player_id, result.name, result.ages, stat),
        view.trajectory_series(data, comp.player_id, comp.name, span, stat),
        view.projection_band(result, stat) if result.projection.n_comps else None,
        SHORT_LABELS.get(stat, stat),
        theme_type(),
    )
    st.plotly_chart(figure)

    proj = result.projection
    if proj.n_comps == 0:
        st.info("None of these comps has played all the projected seasons yet, so there is no "
                "projection.")  # fmt: skip
        return
    st.subheader(f"What the {proj.n_comps} most similar comps with known outcomes did next")
    st.dataframe(
        view.outcome_frame(result),
        hide_index=True,
        column_config={
            c: st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d%%")
            for c in ("Full season", "Under the minimum", "Not in the NBA")
        },
    )
    st.markdown(
        f"**Projected stats, in {season_label(data.last_season)} terms** "
        "(score-weighted median, 25–75% range)"
    )
    st.dataframe(view.projection_frame(result, data, TABLE_STATS), hide_index=True)
    st.caption(
        "A rough guide, not a forecast: it summarizes what similar players did, weighted by "
        "similarity, and counts comps who left the league instead of dropping them. Stats come "
        "only from comps who played a full season at that age. Ages are Basketball-Reference's "
        "age on February 1. Careers that began before 1973-74 or in the ABA aren't compared "
        "over the years we don't have."
    )


if __name__ == "__main__":
    render(sidebar.current())
