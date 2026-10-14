from __future__ import annotations

import sqlite3

import numpy as np
import plotly.graph_objects as go
import pytest

from nba_similarity.careers.data import CareerData, load_career_data
from nba_similarity.careers.engine import CareerConfig, CareerEngine, CareerResult
from nba_similarity.ui import charts, view
from tests.careers.conftest import league  # noqa: F401  (fixture)


@pytest.fixture
def career(league: sqlite3.Connection) -> tuple[CareerData, CareerResult]:  # noqa: F811
    data = load_career_data(league)
    return data, CareerEngine(data, CareerConfig(horizon=3)).find_comps("query", 21)


def test_career_ages_newest_first(career: tuple[CareerData, CareerResult]) -> None:
    data, _ = career
    assert view.career_ages(data, "query") == [21, 20, 19]
    assert view.career_ages(data, "partial")[:3] == [28, 27, 26]


def test_comps_and_outcome_frames(career: tuple[CareerData, CareerResult]) -> None:
    _, result = career
    comps = view.career_comps_frame(result, 3)
    assert list(comps.columns)[:5] == ["Rank", "Player", "Score", "Seasons in", "Window"]
    assert comps.loc[0, "Player"] == "Player twin"
    assert comps.loc[0, "Ages 22-24"] == "22 ●  23 ●  24 ●"
    outcomes = view.outcome_frame(result)
    assert outcomes["Age"].tolist() == [22, 23, 24]
    shares = outcomes[["Full season", "Under the minimum", "Not in the NBA"]].sum(axis=1)
    assert np.allclose(shares, 100, atol=2)  # whole percentages


def test_projection_frame_in_todays_terms(career: tuple[CareerData, CareerResult]) -> None:
    data, result = career
    frame = view.projection_frame(result, data, ("ts_pct", "mpg"))
    assert list(frame.columns) == ["Age", "TS%", "MPG"]
    assert frame["Age"].tolist() == ["22", "23", "24"]
    assert "(" in frame.loc[0, "MPG"]  # median (25%-75%)


def test_trajectory_series_and_figure(career: tuple[CareerData, CareerResult]) -> None:
    data, result = career
    query = view.trajectory_series(data, "query", "Query", result.ages, "ts_pct")
    gap = view.trajectory_series(data, "gap", "Gap", [19, 20, 21, 22], "ts_pct")
    assert np.isnan(gap.pct[1]) and not np.isnan(gap.pct[0])  # no season at 20: a gap
    assert (np.diff(query.pct) > 0).all()  # a rising career
    band = view.projection_band(result, "ts_pct")
    assert band.pct.shape == (3, 3)

    fig = charts.trajectory_figure(query, gap, band, "TS%")
    names = [t.name for t in fig.data]
    assert names == ["Projection, 25–75%", "Projection, median", "Gap", "Query"]
    assert all(isinstance(t, go.Scatter) for t in fig.data)
    assert fig.data[2].y[1] is None  # gaps stay gaps
    assert fig.layout.yaxis.range == (0, 100)
