from __future__ import annotations

import importlib.util
import io
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

from nba_similarity.features.definitions import FEATURES
from nba_similarity.features.pipeline import build_features
from nba_similarity.ingestion.schema import init_db
from nba_similarity.ingestion.store import write_season
from nba_similarity.similarity.engine import SimilarityEngine
from tests.features.conftest import make_player

_spec = importlib.util.spec_from_file_location(
    "validate_archetypes", Path(__file__).parents[2] / "scripts" / "validate_archetypes.py"
)
assert _spec is not None and _spec.loader is not None
va = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = va  # dataclasses need the module registered
_spec.loader.exec_module(va)


@pytest.fixture
def engine() -> SimilarityEngine:
    conn = sqlite3.connect(":memory:")
    init_db(conn)
    players = [make_player(f"p{i}", 2020, 600 + 100 * i, float(i)) for i in range(15)]
    write_season(conn, 2020, players, is_final=True)
    build_features(conn)
    return SimilarityEngine.from_db(conn)


def test_has_trait() -> None:
    assert va.has_trait(1.0, +1) is True
    assert va.has_trait(0.5, +1) is False
    assert va.has_trait(-1.0, -1) is True
    assert va.has_trait(1.0, -1) is False
    assert va.has_trait(float("nan"), +1) is None


def test_cases_cover_the_requested_players() -> None:
    assert [(c.player_id, c.season) for c in va.CASES] == [
        ("curryst01", 2016), ("onealsh01", 2000), ("goberru01", 2022),
        ("paulch01", 2008), ("antetgi01", 2020),
    ]  # fmt: skip
    names = {f.name for f in FEATURES}
    for case in va.CASES:
        assert case.signature and all(d in (-1, 1) for _, d in case.signature)
        assert {f for f, _ in case.signature} <= names


def test_review_counts_shared_missing_and_unknown_traits(engine: SimilarityEngine) -> None:
    case = va.Case("p14", 2020, "top player", "high everything", (("pts", 1), ("ast_pct", -1)))
    match = engine.find_similar("p14", 2020, n=1)[0]  # p13: also well above average
    review = va.review_comp(engine, case, match)
    assert review.traits_shared == ("pts",) and review.traits_missing == ("ast_pct",)
    assert review.fit == 0.5 and review.flagged
    m = engine.matrix
    q, r = m.row("p14", 2020), m.row(match.player_id, match.season)
    expected_gap = np.mean([abs(m.z[q, m.features.index(f)] - m.z[r, m.features.index(f)])
                            for f in ("pts", "ast_pct")])  # fmt: skip
    assert review.gap == pytest.approx(expected_gap)


def test_weight_and_distance_shares_sum_to_one(engine: SimilarityEngine) -> None:
    matches = engine.find_similar("p7", 2020, n=5)
    weights = va.weight_shares(engine)
    assert weights["scoring"] == pytest.approx(3 / 11)  # points per 100 is display-only
    distance = va.distance_shares(engine, engine.matrix.row("p7", 2020), matches)
    assert set(distance) == {"scoring", "efficiency", "playmaking", "rebounding", "defense"}
    assert sum(distance.values()) == pytest.approx(1)


def test_run_case_prints_comps_and_summary(engine: SimilarityEngine) -> None:
    case = va.Case("p14", 2020, "Player p14 2019-20", "high everything", (("pts", 1),))
    out = io.StringIO()
    reviews = va.run_case(engine, case, n=3, out=out)
    text = out.getvalue()
    assert len(reviews) == 3
    assert "query z-profile: pts=" in text and "alike:" in text
    assert "summary: mean archetype fit" in text and "distance by category" in text
    assert "weight by category" in text and "gap " in text


def test_missing_query_is_skipped(engine: SimilarityEngine) -> None:
    case = va.Case("nobody", 2020, "Nobody", "none", (("pts", 1),))
    out = io.StringIO()
    assert va.run_case(engine, case, n=3, out=out) == []
    assert "NOT IN FEATURE MATRIX" in out.getvalue()


def test_main_runs_against_a_database(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = tmp_path / "nba.sqlite"
    conn = sqlite3.connect(db)
    init_db(conn)
    write_season(
        conn, 2020, [make_player(f"p{i}", 2020, 700, float(i)) for i in range(12)], is_final=True
    )
    build_features(conn)
    conn.close()
    case = va.Case("p5", 2020, "Player p5", "middle", (("pts", 1),))
    assert va.main(["--db", str(db), "-n", "2", "--metric", "cosine"], cases=[case]) == 0
    out = capsys.readouterr().out
    assert "metric=cosine" in out and "overall: mean archetype fit" in out
    assert np.isfinite(float(out.split("mean archetype fit ")[-1].split("%")[0]))
