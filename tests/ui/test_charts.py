from __future__ import annotations

import numpy as np

from nba_similarity.ui.charts import MATCH_COLOR, QUERY_COLOR, radar_figure
from nba_similarity.ui.view import RadarData

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
