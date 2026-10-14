from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from nba_similarity.features.career import career_players
from nba_similarity.features.definitions import FEATURE_NAMES
from nba_similarity.features.pipeline import build_features
from nba_similarity.ingestion.store import write_season
from tests.features.conftest import make_player


def rows(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM player_season_career_features", conn)


def test_every_player_season_gets_a_status(ingested_db: sqlite3.Connection) -> None:
    write_season(
        ingested_db,
        2021,
        [make_player(f"q{i}", 2021, 900 + 10 * i, float(i % 5)) for i in range(10)]
        + [make_player("tiny", 2021, 100, 3.0)],
        is_final=True,
    )
    report = build_features(ingested_db)
    df = rows(ingested_db).set_index(["player_id", "season"])
    n = ingested_db.execute("SELECT COUNT(*) FROM player_seasons").fetchone()[0]
    assert len(df) == n

    bench = df.loc[("bench1976", 1976)]  # 420 min: under 500, over 250
    assert bench["status"] == "partial" and np.isclose(bench["reliability"], 420 / 500)
    assert not np.isnan(bench["z_ts_pct"])  # scored against the qualified players
    bench20 = df.loc[("bench2020", 2020)]  # 2019-20's scaled minimum is 433
    assert bench20["status"] == "partial" and np.isclose(
        bench20["reliability"], 420 / 433, atol=1e-4
    )
    assert bench20["career_min_minutes"] == 216  # 250 scaled to the 71-game schedule
    tiny = df.loc[("tiny", 2021)]
    assert tiny["status"] == "limited" and np.isnan(tiny["z_ts_pct"])
    assert report.n_career_scored == int((df["status"] != "limited").sum())
    assert not report.aba_rosters_loaded


def test_qualified_rows_match_the_season_comp_features(ingested_db: sqlite3.Connection) -> None:
    build_features(ingested_db)
    z = [f"z_{f}" for f in FEATURE_NAMES]
    comps = pd.read_sql_query("SELECT * FROM player_season_features", ingested_db)
    career = rows(ingested_db).query("status == 'qualified'")
    merged = comps.merge(career, on=["player_id", "season"], suffixes=("", "_c"))
    assert len(merged) == len(comps) == len(career)
    for col in z:
        assert np.allclose(merged[col], merged[f"{col}_c"], equal_nan=True)


def test_minutes_per_game_is_z_scored_over_qualified_players(
    ingested_db: sqlite3.Connection,
) -> None:
    build_features(ingested_db)
    df = rows(ingested_db)
    q = df[(df["status"] == "qualified") & (df["season"] == 2020)]
    assert np.isclose(q["z_mpg"].mean(), 0, atol=1e-9)
    assert np.isclose((q["mp"] / q["g"]).iloc[0], q["mpg"].iloc[0])


def test_career_players_birth_offset_and_censoring() -> None:
    raw = pd.DataFrame(
        {
            "player_id": ["old", "old", "rook", "rook", "aba", "aba"],
            "season": [1974, 1975, 1990, 1991, 1977, 1978],
            "age": [30, 31, 20, 21, 22, 23],
        }
    )
    aba = pd.DataFrame({"player_id": ["aba", "aba"], "season": [1975, 1976]})
    players = career_players(raw, aba).set_index("player_id")
    assert players.loc["rook", "birth_offset"] == 1970 and players.loc["rook", "censor"] == ""
    assert pd.isna(players.loc["rook", "censored_through_age"])
    assert players.loc["old", "censor"] == "pre-1974"
    assert players.loc["old", "censored_through_age"] == 29  # may have played before 30
    assert players.loc["aba", "censor"] == "ABA"
    assert players.loc["aba", "censored_through_age"] == 21  # ABA through 1975-76, age 21

    no_aba = career_players(raw, None).set_index("player_id")
    assert no_aba.loc["aba", "censor"] == ""
