from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from nba_similarity.ui import charts, view
from nba_similarity.ui.charts import MATCH_COLOR, QUERY_COLOR, radar_figure
from nba_similarity.ui.view import RadarData
from tests.similarity.conftest import col, make_matrix, random_population

DATA = RadarData(
    labels=("USG%", "3PA/100", "TS%"),
    query_pct=np.array([90.0, 40.0, 80.0]),
    match_pct=np.array([85.0, np.nan, 70.0]),
    query_raw=np.array([30.0, 5.0, 0.6]),
    match_raw=np.array([28.0, np.nan, 0.58]),
)


def test_two_closed_traces_on_a_percentile_scale() -> None:
    fig = radar_figure(DATA, "Query 2022-23", "Match 1999-00")
    assert [t.name for t in fig.data] == ["Query 2022-23", "Match 1999-00"]
    query, match = fig.data
    assert list(query.theta) == ["USG%", "3PA/100", "TS%", "USG%"]  # polygon closed
    assert list(query.r) == [90.0, 40.0, 80.0, 90.0]
    assert tuple(fig.layout.polar.radialaxis.range) == (0, 100)


def test_missing_stat_is_a_gap_not_zero() -> None:
    match = radar_figure(DATA, "q", "m").data[1]
    assert match.r[1] is None and match.connectgaps is False
    assert "no data" in match.hovertext[1]


def test_profiles_with_gaps_are_not_filled() -> None:
    query, match = radar_figure(DATA, "q", "m").data
    assert query.fill == "toself"  # complete profile
    assert match.fill == "none"  # a fill would bridge the missing 3PA/100


def test_colors_follow_theme() -> None:
    light = radar_figure(DATA, "q", "m", theme="light")
    dark = radar_figure(DATA, "q", "m", theme="dark")
    assert light.data[0].line.color == QUERY_COLOR["light"]
    assert dark.data[1].line.color == MATCH_COLOR["dark"]


def _map_frame(**kwargs: object) -> pd.DataFrame:
    z = random_population(30)
    z[:5, col("fg3a")] = np.nan  # pre-1980 seasons lack 3PA
    xy = np.random.default_rng(1).normal(size=(30, 2))
    return view.map_frame(make_matrix(z), xy, query_row=0, comp_rows=[4, 7, 9], **kwargs)


def test_map_figure_stat_mode_grays_missing_and_highlights() -> None:
    fig = charts.map_figure(_map_frame(color_feature="fg3a"), color_label="3PA/100")
    names = [t.name for t in fig.data]
    assert names == ["No data", "3PA/100", "Top comps (rank)", "Selected season"]
    assert all(isinstance(t, go.Scattergl) for t in fig.data[:2])  # WebGL for ~16k points
    assert len(fig.data[0].x) == 5 and fig.data[0].marker.color == charts.NO_DATA["light"]
    assert list(fig.data[2].text) == ["1", "2", "3"]
    assert len(fig.data[3].x) == 1 and fig.data[3].text[0] == "<b>p0 '20</b>"


def test_map_figure_cluster_mode_one_trace_per_cluster_with_labels() -> None:
    labels = np.arange(30) % 4
    frame = _map_frame(clusters=labels, cluster_names={c: f"group {c}" for c in range(4)})
    fig = charts.map_figure(frame, color_by="cluster", theme="dark")
    clusters = fig.data[:4]
    assert [t.name for t in clusters] == ["1: group 0", "2: group 1", "3: group 2", "4: group 3"]
    # four clusters: colors repeat after three, with a second marker shape
    assert clusters[3].marker.color == clusters[0].marker.color
    assert clusters[3].marker.symbol != clusters[0].marker.symbol
    assert [a.text for a in fig.layout.annotations] == ["<b>1</b>", "<b>2</b>", "<b>3</b>",
                                                        "<b>4</b>"]  # fmt: skip


def test_cluster_styles_unique_up_to_max() -> None:
    styles = {charts.cluster_style(c) for c in range(charts.MAX_CLUSTERS)}
    assert len(styles) == charts.MAX_CLUSTERS


def test_label_positions_avoid_overlaps() -> None:
    # three points on top of each other: every label gets its own side
    sides = charts.label_positions(np.zeros(3), np.zeros(3), [1, 1, 1], px_per_unit=100.0)
    assert len(set(sides)) == 3 and sides[0] == "top center"
    # far apart: all keep the preferred side
    far = charts.label_positions(np.array([0.0, 10.0]), np.zeros(2), [8, 8], px_per_unit=100.0)
    assert far == ["top center", "top center"]
