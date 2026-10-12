"""The engine over databases built by the real ingestion and feature pipelines."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from nba_similarity.features.pipeline import build_features
from nba_similarity.ingestion.parse import parse_page
from nba_similarity.ingestion.pipeline import ingest
from nba_similarity.ingestion.schema import init_db
from nba_similarity.ingestion.store import write_season
from nba_similarity.similarity import cli
from nba_similarity.similarity.config import SimilarityConfig
from nba_similarity.similarity.engine import SimilarityEngine
from tests.features.conftest import make_player
from tests.ingestion.conftest import read_fixture


@pytest.fixture
def synthetic_db(tmp_path: Path) -> Path:
    path = tmp_path / "nba.sqlite"
    conn = sqlite3.connect(path)
    init_db(conn)
    for season in (1976, 2020, 2021):
        players = [make_player(f"p{i}", season, 600 + 100 * i, float(i)) for i in range(15)]
        write_season(conn, season, players, is_final=True)
    build_features(conn)
    conn.close()
    return path


def test_engine_loads_from_db(synthetic_db: Path) -> None:
    conn = sqlite3.connect(synthetic_db)
    engine = SimilarityEngine.from_db(conn)
    conn.close()
    assert engine.matrix.z.shape == (45, 12)
    matches = engine.find_similar("p7", 2020, n=5)
    assert all(m.player_id != "p7" for m in matches)
    assert matches[0].player_id in {"p6", "p8"}  # neighbours in the synthetic ladder
    # pre-1980 seasons (no 3PA rate, USG%, TOV%) share 80% of the weight and stay comparable
    assert any(m.season == 1976 for m in engine.find_similar("p7", 2020, n=45))


def test_cli_prints_ranked_matches(synthetic_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["Player p7", "2020", "-n", "3", "--db", str(synthetic_db),
                     "--weight", "scoring=2", "--include-same-player"])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0
    assert out.startswith("Most similar to Player p7 2019-20 (euclidean):")
    assert out.count("alike:") == 3 and " 1. Player p7 " in out  # 2020-21 twin first


def test_cli_reports_unknown_player(synthetic_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["Nobody", "2020", "--db", str(synthetic_db)]) == 1
    assert "no qualified player-season" in capsys.readouterr().err


def test_cli_rejects_bad_weight(synthetic_db: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["Player p7", "2020", "--db", str(synthetic_db), "--weight", "shooting=2"]) == 1
    assert "unknown categories" in capsys.readouterr().err


@pytest.fixture(scope="module")
def season_2023() -> SimilarityEngine:
    """The real 2022-23 season (needs fixtures from scripts/make_fixtures.sh)."""
    import tempfile

    raw_dir = Path(tempfile.mkdtemp())
    for page in ("per_poss", "advanced"):
        html = read_fixture(f"NBA_2023_{page}.html.gz")
        (raw_dir / f"NBA_2023_{page}.html").write_text(html, encoding="utf-8")

    class NoNetwork:
        def get(self, url: str) -> str:
            raise AssertionError(url)

    conn = sqlite3.connect(":memory:")
    init_db(conn)
    ingest([2023], conn, NoNetwork(), raw_dir, parse_page, today=date(2026, 10, 10))
    build_features(conn)
    engine = SimilarityEngine.from_db(conn)
    conn.close()
    return engine


def test_real_season_jokic_matches_sabonis(season_2023: SimilarityEngine) -> None:
    match = season_2023.find_similar("jokicni01", 2023, n=1)[0]
    assert match.player_id == "sabondo01"
    assert "drb_pct" in [c.feature for c in match.similar_traits]
    assert "ast_pct" in [c.feature for c in match.differences]  # 46.6 vs 30.4


def test_real_season_curry_and_scorer_weighting(season_2023: SimilarityEngine) -> None:
    assert season_2023.find_similar("curryst01", 2023, n=1)[0].player_id == "lillada01"
    engine = season_2023.with_config(category_weights={"scoring": 3})
    for match in engine.find_similar("curryst01", 2023, n=5):
        assert match.contributions["usg_pct"] > 0  # weighting scoring keeps comps high-usage
        row = engine.matrix.row(match.player_id, match.season)
        assert engine.matrix.raw[row, engine.matrix.features.index("usg_pct")] >= 28


def test_real_season_cosine_and_pca_run(season_2023: SimilarityEngine) -> None:
    for config in (SimilarityConfig(metric="cosine"), SimilarityConfig(pca_components=0.9)):
        matches = SimilarityEngine(season_2023.matrix, config).find_similar("goberru01", 2023, n=5)
        assert len(matches) == 5 and all(0 <= m.score <= 100 for m in matches)
