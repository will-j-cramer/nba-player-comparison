"""Map page: every player-season placed in 2D (UMAP) so similar seasons sit together."""

from __future__ import annotations

import streamlit as st

from nba_similarity.similarity.explain import season_label
from nba_similarity.ui import cache, charts, sidebar, view
from nba_similarity.ui.theme import theme_type

DEFAULT_STAT = "ast_pct"
DEFAULT_K = 6


def render(query: sidebar.Query) -> None:
    """Draw the map page for the sidebar's query."""
    st.title("Map of player-seasons")
    layout = cache.map_layout(query.db, query.mtime)
    if layout is None:
        st.info("No map has been built yet. Run `nba-map` (after `nba-features`), then reload.")
        st.stop()
    if layout.stale:
        st.warning(
            "The features changed since this map was built, so positions may be out of date. "
            "Re-run `nba-map`."
        )
    matrix = query.matrix

    left, right = st.columns([1, 2])
    with left:
        color_by = st.radio(
            "Color by", ["Stat", "Cluster"], horizontal=True, key="map_color_by"
        ).lower()
    color_feature, clusters, names = None, None, None
    with right:
        if color_by == "stat":
            stat_options = [f for f in matrix.features]
            color_feature = st.selectbox(
                "Stat",
                stat_options,
                index=stat_options.index(DEFAULT_STAT) if DEFAULT_STAT in stat_options else 0,
                format_func=lambda f: matrix.labels[matrix.features.index(f)],
                key="map_stat",
            )
        else:
            k = st.slider(
                "Clusters (k-means on the map)", 2, charts.MAX_CLUSTERS, DEFAULT_K, key="map_k"
            )
            clusters, names = cache.map_clusters(query.db, query.mtime, k)

    comp_rows: list[int] = []
    if query.has_weights:
        matches = cache.search(
            query.db, query.mtime, query.config_key, query.player.player_id, query.season, query.n
        )
        comp_rows = [matrix.row(m.player_id, m.season) for m in matches]

    frame = view.map_frame(
        matrix,
        layout.xy,
        query_row=query.row,
        comp_rows=comp_rows,
        color_feature=color_feature,
        clusters=clusters,
        cluster_names=names,
    )
    label = view.SHORT_LABELS.get(color_feature, color_feature) if color_feature is not None else ""
    st.plotly_chart(
        charts.map_figure(frame, color_by=color_by, color_label=label, theme=theme_type()),
        config={"scrollZoom": True},
    )

    if comp_rows:
        st.markdown(
            "**Comps:** "
            + " · ".join(
                f"**{i}** {matrix.names[r]} {season_label(int(matrix.seasons[r]))}"
                for i, r in enumerate(comp_rows, start=1)
            )
        )
    unplaced = len(matrix.player_ids) - len(frame)
    notes = [
        f"{len(frame):,} player-seasons. Nearby points have similar stat profiles under the "
        "default (equal) weights; the axes have no meaning of their own. Scroll to zoom, drag "
        "to pan, hover a point for its stats.",
        f"★ {query.label} with its top {len(comp_rows)} comps from the sidebar settings "
        "(rings, numbered by rank). With non-default weights or cosine distance, comps can "
        "sit away from the star, because the map is precomputed with default weights.",
    ]
    if color_by == "stat":
        notes.append(
            "Color is the percentile within each season (50 = league average), so eras compare "
            "fairly; gray seasons predate the stat."
        )
    else:
        notes.append(
            "Clusters are k-means groups on the map, named by their most distinctive average "
            "stats. They summarize regions of a continuum, not hard positions."
        )
    if unplaced:
        notes.append(f"{unplaced} seasons are not on the map yet; re-run `nba-map`.")
    st.caption("  \n".join(notes))
    st.caption(f"Map built {layout.built_at} · seasons {season_label(int(matrix.seasons.min()))}"
               f" to {season_label(int(matrix.seasons.max()))}")  # fmt: skip


if __name__ == "__main__":
    render(sidebar.current())
