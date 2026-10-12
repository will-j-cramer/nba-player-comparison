"""Runs the real Streamlit script headlessly (streamlit.testing) against a synthetic database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from nba_similarity.features.pipeline import build_features
from nba_similarity.ingestion.schema import init_db
from nba_similarity.ingestion.store import write_season
from tests.features.conftest import make_player

APP = str(Path(__file__).parents[2] / "src" / "nba_similarity" / "ui" / "app.py")


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "nba.sqlite"
    conn = sqlite3.connect(path)
    init_db(conn)
    for season in (1976, 2020, 2021):
        players = [make_player(f"p{i}", season, 600 + 50 * i, float(i % 8)) for i in range(20)]
        write_season(conn, season, players, is_final=True)
    build_features(conn)
    conn.close()
    monkeypatch.setenv("NBA_DB", str(path))
    return path


def run(at: AppTest) -> AppTest:
    at.run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_renders_results_and_comparison(db: Path) -> None:
    at = run(AppTest.from_file(APP))
    assert not at.error
    assert at.subheader[0].value.startswith("Most similar to Player p0")  # first by name
    results = at.dataframe[0].value
    assert len(results) == 10 and list(results.columns)[:4] == ["Rank", "Player", "Season", "Score"]
    assert len(at.dataframe) == 2  # results + side-by-side comparison
    assert "vs" in at.subheader[1].value


def test_controls_rerun_cleanly(db: Path) -> None:
    at = run(AppTest.from_file(APP))
    at.selectbox(key="season").set_value(2021)
    at.slider(key="weight_scoring").set_value(2.0)
    at.toggle(key="include_own").set_value(True)
    at.slider(key="n").set_value(5)
    at.radio(key="metric").set_value("cosine")
    at = run(at)
    results = at.dataframe[0].value
    assert len(results) == 5
    assert "Player p0" in results["Player"].tolist()  # own other seasons now allowed


def test_all_zero_weights_warns(db: Path) -> None:
    at = run(AppTest.from_file(APP))
    for c in ("scoring", "efficiency", "playmaking", "rebounding", "defense"):
        at.slider(key=f"weight_{c}").set_value(0.0)
    at = run(at)
    assert "All category weights are 0" in at.warning[0].value


def test_missing_database_shows_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NBA_DB", str(tmp_path / "nope.sqlite"))
    at = run(AppTest.from_file(APP))
    assert "Database not found" in at.error[0].value
