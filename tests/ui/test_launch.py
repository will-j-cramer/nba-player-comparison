from __future__ import annotations

import sys

import pytest

from nba_similarity.ui import launch


def test_command_uses_this_python_and_the_packaged_app() -> None:
    cmd = launch.command(["--server.port", "8502"])
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"]
    assert cmd[4].endswith("ui/app.py") and launch.APP.exists()
    assert cmd[5:] == ["--server.port", "8502"]


def test_main_passes_arguments_and_returns_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(launch.subprocess, "call", lambda cmd: calls.append(cmd) or 3)
    assert launch.main(["--server.headless", "true"]) == 3
    assert calls == [launch.command(["--server.headless", "true"])]
