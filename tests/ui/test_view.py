from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from nba_similarity.features.pipeline import build_features
from nba_similarity.ingestion.schema import init_db
from nba_similarity.ingestion.store import write_season
from nba_similarity.similarity.engine import SimilarityEngine
from nba_similarity.similarity.explain import format_contribution, season_label
from nba_similarity.ui import view
from tests.features.conftest import make_player
from tests.similarity.conftest import col, make_matrix, random_population


def test_season_label_and_format() -> None:
    assert season_label(2023) == "2022-23" and season_label(2000) == "1999-00"


def test_player_options_sorted_with_namesakes_disambiguated() -> None:
    m = make_matrix(np.zeros((4, 12)), ["johnsed01", "jokicni01", "johnsed02", "jokicni01"],
                    [1985, 2022, 1990, 2023])  # fmt: skip
    object.__setattr__(
        m, "names", np.array(["Eddie Johnson", "Nikola Jokić", "Eddie Johnson", "Nikola Jokić"])
    )
    options = view.player_options(m)
    assert [o.label for o in options] == [
        "Eddie Johnson (1985, johnsed01)",
        "Eddie Johnson (1990, johnsed02)",
        "Nikola Jokić (2022–2023)",
    ]
    assert options[2].seasons == (2023, 2022)  # newest first


def test_query_profile_names_standout_stats() -> None:
    z = np.zeros((2, 12))
    z[0, col("ast_pct")] = 3.0
    z[0, col("blk_pct")] = -1.5
    text = view.query_profile(make_matrix(z), 0)
    assert text.startswith("elite Assist %") and "low Block %" in text
    assert view.query_profile(make_matrix(np.zeros((2, 12))), 0).startswith("no stat")


@pytest.fixture
def engine() -> SimilarityEngine:
    z = random_population(80)
    z[1] = z[0] + 0.1
    z[1, col("fg3a")] = np.nan  # a pre-1980-style gap
    return SimilarityEngine(make_matrix(z))


def test_results_frame(engine: SimilarityEngine) -> None:
    matches = engine.find_similar("p0", 2020, n=5)
    df = view.results_frame(matches)
    assert list(df.columns) == [
        "Rank",
        "Player",
        "Season",
        "Score",
        "Stats shared",
        "Alike",
        "Differ",
    ]
    assert df["Rank"].tolist() == [1, 2, 3, 4, 5]
    assert df["Score"].tolist() == sorted(df["Score"], reverse=True)
    assert df.loc[0, "Player"] == "Player p1" and df.loc[0, "Stats shared"] == "91%"
    assert (df.loc[1:, "Stats shared"] == "").all()  # complete seasons show nothing


def test_comparison_frame(engine: SimilarityEngine) -> None:
    match = engine.find_similar("p0", 2020, n=1)[0]
    league = {(2020, f): 1.0 for f in engine.matrix.features}
    df = view.comparison_frame(engine.matrix, 0, match, league)
    assert len(df) == 12 and df.loc[0, "Stat"] == "Points per 100 poss"
    by_stat = df.set_index("Stat")
    assert by_stat.loc["Points per 100 poss", "Verdict"] == "context"  # display-only
    assert pd.isna(by_stat.loc["Points per 100 poss", "Points"])
    assert by_stat.loc["3PA per 100 poss", "Verdict"] == "n/a"  # the match lacks it
    assert pd.isna(by_stat.loc["3PA per 100 poss", "Match"])
    assert set(df["Verdict"]) <= {"Alike", "Differ", "n/a", "context"}
    assert df["Points"].sum() == pytest.approx(match.score, abs=1e-6)
    assert (df["Lg avg (query yr)"] == 1.0).all()
    # every stat the engine listed as a difference is a Differ row
    for c in match.differences:
        assert by_stat.loc[c.label, "Verdict"] in ("Differ", "context")


def test_radar_data_uses_distance_stats_as_percentiles(engine: SimilarityEngine) -> None:
    data = view.radar_data(engine.matrix, 0, 1)
    assert len(data.labels) == 11 and "PTS/100" not in data.labels  # display-only left out
    assert np.isnan(data.match_pct[data.labels.index("3PA/100")])
    finite = data.query_pct[~np.isnan(data.query_pct)]
    assert ((finite >= 0) & (finite <= 100)).all()


def test_percentile() -> None:
    pct = view.percentile(np.array([0.0, 1.0, -1.0, np.nan]))
    assert pct[0] == pytest.approx(50)
    assert pct[1] == pytest.approx(84.13, abs=0.01) and pct[2] == pytest.approx(15.87, abs=0.01)
    assert np.isnan(pct[3])


def test_short_label() -> None:
    assert view.short_label("Nikola Jokić", 2023) == "Jokić '23"


def test_format_contribution_handles_missing(engine: SimilarityEngine) -> None:
    match = engine.find_similar("p0", 2020, n=1)[0]
    c = match.similar_traits[0]
    assert format_contribution(c).startswith(c.label)


def test_load_league_averages() -> None:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    write_season(conn, 2020, [make_player(f"p{i}", 2020, 700, float(i)) for i in range(10)],
                 is_final=True)  # fmt: skip
    build_features(conn)
    league = view.load_league_averages(conn)
    assert league[(2020, "ast_pct")] == pytest.approx(4.5)  # mean of 0..9
