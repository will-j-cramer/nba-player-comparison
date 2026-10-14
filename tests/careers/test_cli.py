from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from nba_similarity.careers.cli import main
from tests.careers.conftest import build_league


@pytest.fixture(scope="module")
def db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("league") / "nba.sqlite"
    conn = sqlite3.connect(path)
    build_league(conn)
    conn.close()
    return path


def test_prints_comps_outcomes_and_projection(db: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["Player query", "--through-age", "21", "--horizon", "3", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Career comps for Player query through age 21 (ages 19-21")
    assert " 1. Player twin" in out and "22 ●  23 ●  24 ●" in out
    assert "22 ○  23 ○  24 ○" in out  # the bust left the league
    assert "22 ·  23 ·  24 ·" in out  # the young comp hasn't played those seasons yet
    assert "Projection from the" in out and "MPG today" in out


def test_errors_are_reported(db: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["Player query", "--through-age", "30", "--db", str(db)]) == 1
    assert "no season of at least the career minimum at age 30" in capsys.readouterr().err
    assert main(["Nobody Atall", "--through-age", "21", "--db", str(db)]) == 1
