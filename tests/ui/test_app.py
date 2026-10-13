"""Runs the real Streamlit script headlessly (streamlit.testing) against a synthetic database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from nba_similarity.features.pipeline import build_features
from nba_similarity.ingestion.schema import init_db
from nba_similarity.ingestion.store import write_season
from nba_similarity.mapping.store import MapParams, features_signature, write_map
from tests.features.conftest import make_player

APP = str(Path(__file__).parents[2] / "src" / "nba_similarity" / "ui" / "app.py")
CLOUD_ENTRY = str(Path(__file__).parents[2] / "streamlit_app.py")


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


def build_map(path: Path) -> None:
    """Stored map with synthetic coordinates (UMAP itself is tested in tests/mapping)."""
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT player_id, season FROM player_season_features").fetchall()
    xy = np.random.default_rng(0).normal(size=(len(rows), 2))
    ids, seasons = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    write_map(conn, ids, seasons, xy, MapParams(15, 0.1, 42), features_signature(conn))
    conn.close()


def map_page(at: AppTest) -> AppTest:
    run(at)
    at.switch_page("map_page.py")
    return run(at)


def test_map_page_without_map_explains_how_to_build_it(db: Path) -> None:
    at = map_page(AppTest.from_file(APP))
    assert "Run `nba-map`" in at.info[0].value


def test_map_page_colors_by_stat_and_cluster(db: Path) -> None:
    build_map(db)
    at = map_page(AppTest.from_file(APP))
    assert at.title[0].value == "Map of player-seasons" and not at.warning
    assert len(at.get("plotly_chart")) == 1
    assert at.selectbox(key="map_stat").value == "ast_pct"
    at.selectbox(key="map_stat").set_value("fg3a")  # missing before 1980: gray points
    at = run(at)
    at.radio(key="map_color_by").set_value("Cluster")
    at = run(at)
    at.slider(key="map_k").set_value(4)
    at.slider(key="weight_defense").set_value(2.0)  # sidebar still drives the highlighted comps
    at = run(at)
    assert not at.error and not at.warning


def test_map_page_warns_when_features_changed(db: Path) -> None:
    build_map(db)
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE player_season_features SET z_ast_pct = z_ast_pct + 0.5")
    conn.close()
    at = map_page(AppTest.from_file(APP))
    assert "Re-run `nba-map`" in at.warning[0].value


def test_cloud_entry_point_renders_search_page(db: Path) -> None:
    at = run(AppTest.from_file(CLOUD_ENTRY))
    assert at.title[0].value == "NBA player-season similarity"
    assert len(at.dataframe[0].value) == 10


def test_missing_database_mentions_download_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NBA_DB", str(tmp_path / "nope.sqlite"))
    at = run(AppTest.from_file(APP))
    assert "NBA_DB_URL" in at.error[0].value


def test_database_downloaded_from_secret_url(
    db: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no file on disk, the app fetches NBA_DB_URL once (the HTTP call is faked)."""
    from nba_similarity.ui import remote_db

    target = tmp_path / "downloaded" / "nba.sqlite"
    monkeypatch.setenv("NBA_DB", str(target))
    calls = []

    class Response:
        status_code = 200

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def iter_content(self, size: int) -> list[bytes]:
            return [db.read_bytes()]

    def fake_get(url: str, **kwargs: object) -> Response:
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr(remote_db.requests, "get", fake_get)
    at = AppTest.from_file(APP)
    at.secrets["NBA_DB_URL"] = "https://example.test/nba.sqlite"
    at.secrets["NBA_DB_TOKEN"] = "secret"
    at = run(at)
    from nba_similarity.ui.remote_db import download_path

    assert download_path(target, "https://example.test/nba.sqlite").exists()
    assert not target.exists() and len(calls) == 1
    assert calls[0][1]["headers"]["Authorization"] == "Bearer secret"  # type: ignore[index]
    assert len(at.dataframe[0].value) == 10
