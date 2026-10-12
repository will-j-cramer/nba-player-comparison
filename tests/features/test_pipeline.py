from __future__ import annotations

import sqlite3

import pytest

from nba_similarity.features.definitions import FEATURE_NAMES, FEATURES, z_column
from nba_similarity.features.pipeline import build_features
from nba_similarity.ingestion.schema import init_db
from tests.features.conftest import ERA_GAPS


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def test_builds_feature_tables(ingested_db: sqlite3.Connection) -> None:
    report = build_features(ingested_db)
    # 1976: 12 qualify, bench (420 < 500) does not; 2020: threshold 433, so bench (420) out too
    assert (report.n_player_seasons, report.n_qualified, report.n_seasons) == (29, 27, 2)

    assert columns(ingested_db, "player_season_features") == [
        "player_id", "season", "mp", "min_minutes", "n_features",
        *FEATURE_NAMES, *(z_column(f) for f in FEATURE_NAMES),
    ]  # fmt: skip
    defs = ingested_db.execute(
        "SELECT feature, category FROM feature_definitions ORDER BY ordinal"
    ).fetchall()
    assert defs == [(f.name, f.category) for f in FEATURES]
    (n_stats,) = ingested_db.execute("SELECT COUNT(*) FROM feature_season_stats").fetchone()
    assert n_stats == 2 * len(FEATURES)


def test_era_gaps_are_null_and_counted(ingested_db: sqlite3.Connection) -> None:
    build_features(ingested_db)
    for feature in ERA_GAPS:
        nulls, total = ingested_db.execute(
            f"SELECT SUM({z_column(feature)} IS NULL), COUNT(*) "
            "FROM player_season_features WHERE season = 1976"
        ).fetchone()
        assert nulls == total == 12
        (covered,) = ingested_db.execute(
            "SELECT covered FROM feature_season_stats WHERE season = 1976 AND feature = ?",
            (feature,),
        ).fetchone()
        assert covered == 0
    n_features = dict(
        ingested_db.execute(
            "SELECT season, MAX(n_features) FROM player_season_features GROUP BY season"
        )
    )
    assert n_features == {1976: 9, 2020: 12}


def test_zscores_centered_per_season(ingested_db: sqlite3.Connection) -> None:
    build_features(ingested_db)
    for (mean,) in ingested_db.execute(
        "SELECT AVG(z_pts) FROM player_season_features GROUP BY season"
    ):
        assert mean == pytest.approx(0, abs=1e-9)


def test_rebuild_is_idempotent(ingested_db: sqlite3.Connection) -> None:
    build_features(ingested_db)
    first = ingested_db.execute("SELECT * FROM player_season_features ORDER BY 1, 2").fetchall()
    build_features(ingested_db)
    assert (
        ingested_db.execute("SELECT * FROM player_season_features ORDER BY 1, 2").fetchall()
        == first
    )


def test_empty_database_raises() -> None:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    with pytest.raises(ValueError, match="nba-ingest"):
        build_features(conn)


def test_reingesting_a_season_after_building_features(ingested_db: sqlite3.Connection) -> None:
    """Ingestion rewrites seasons (e.g. the one in progress) after features exist."""
    from nba_similarity.ingestion.store import write_season
    from tests.features.conftest import make_player

    build_features(ingested_db)
    players = [make_player(f"p2020_{i}", 2020, 700, float(i)) for i in range(15)]
    write_season(ingested_db, 2020, players, is_final=True)
    report = build_features(ingested_db)
    assert report.n_qualified == 12 + 15


def test_transforms_recorded_and_applied(ingested_db: sqlite3.Connection) -> None:
    import math

    build_features(ingested_db)
    transforms = dict(ingested_db.execute("SELECT feature, transform FROM feature_definitions"))
    assert transforms == {f.name: f.transform for f in FEATURES}

    # z_fg3a_rate is standardized on the sqrt scale; the stored raw value is the observed one
    raw = [
        r
        for (r,) in ingested_db.execute(
            "SELECT fg3a_rate FROM player_season_features WHERE season = 2020 ORDER BY player_id"
        )
    ]
    assert max(raw) == 14.0  # observed value of the top synthetic player, not its sqrt
    mean, sd, raw_mean = ingested_db.execute(
        "SELECT mean, sd, raw_mean FROM feature_season_stats "
        "WHERE season = 2020 AND feature = 'fg3a_rate'"
    ).fetchone()
    roots = [math.sqrt(v) for v in raw]
    assert mean == pytest.approx(sum(roots) / len(roots))
    assert raw_mean == pytest.approx(sum(raw) / len(raw))
    (z,) = ingested_db.execute(
        "SELECT z_fg3a_rate FROM player_season_features WHERE season = 2020 AND fg3a_rate = 14.0"
    ).fetchone()
    assert z == pytest.approx((math.sqrt(14.0) - mean) / sd)
