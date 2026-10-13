from __future__ import annotations

from pathlib import Path

import pytest

from nba_similarity.paths import DEFAULT_DB, default_db


def test_default_db_honors_nba_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NBA_DB", raising=False)
    assert default_db() == DEFAULT_DB
    monkeypatch.setenv("NBA_DB", "/tmp/other.sqlite")
    assert default_db() == Path("/tmp/other.sqlite")
    monkeypatch.setenv("NBA_DB", "")
    assert default_db() == DEFAULT_DB
