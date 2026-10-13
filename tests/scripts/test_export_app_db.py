from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from nba_similarity.features.pipeline import build_features
from nba_similarity.mapping.store import (
    MapParams,
    features_signature,
    is_stale,
    load_map,
    write_map,
)
from nba_similarity.similarity.data import load_feature_matrix
from tests.features.conftest import ingested_db  # noqa: F401  (fixture)

_spec = importlib.util.spec_from_file_location(
    "export_app_db", Path(__file__).parents[2] / "scripts" / "export_app_db.py"
)
assert _spec is not None and _spec.loader is not None
export = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = export
_spec.loader.exec_module(export)


def _with_map(conn: sqlite3.Connection) -> sqlite3.Connection:
    build_features(conn)
    rows = conn.execute("SELECT player_id, season FROM player_season_features").fetchall()
    xy = np.random.default_rng(0).normal(size=(len(rows), 2))
    ids, seasons = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    write_map(conn, ids, seasons, xy, MapParams(15, 0.1, 42), features_signature(conn))
    return conn


def tables(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def test_exports_only_app_tables_and_app_can_read_it(
    ingested_db: sqlite3.Connection,  # noqa: F811
    tmp_path: Path,
) -> None:
    source = _with_map(ingested_db)
    out = tmp_path / "app.sqlite"
    counts = export.export_app_db(source, out)

    slim = sqlite3.connect(out)
    assert tables(slim) == set(export.APP_TABLES)
    assert "per100_stats" in tables(source)  # raw tables stay behind
    n = source.execute("SELECT COUNT(*) FROM player_season_features").fetchone()[0]
    assert counts["player_season_features"] == counts["player_season_map"] == n
    assert len(load_feature_matrix(slim).player_ids) == n
    stored = load_map(slim)
    assert stored is not None and not is_stale(slim, stored)


def test_refuses_missing_or_stale_map(
    ingested_db: sqlite3.Connection,  # noqa: F811
    tmp_path: Path,
) -> None:
    build_features(ingested_db)
    with pytest.raises(export.ExportError, match="run nba-map"):
        export.export_app_db(ingested_db, tmp_path / "app.sqlite")
    _with_map(ingested_db)
    ingested_db.execute("UPDATE player_season_features SET z_ast_pct = z_ast_pct + 1")
    with pytest.raises(export.ExportError, match="out of date"):
        export.export_app_db(ingested_db, tmp_path / "app.sqlite")


def test_cli_rejects_overwriting_the_source(tmp_path: Path) -> None:
    db = tmp_path / "nba.sqlite"
    sqlite3.connect(db).close()
    assert export.main(["--db", str(db), "--out", str(db)]) == 2
