from __future__ import annotations

import pytest

from nba_similarity.environment import (
    missing_requirements,
    parse_version,
    streamlit_too_old,
    wrong_environment_message,
)


def test_parse_version() -> None:
    assert parse_version("1.46.0") == (1, 46, 0)
    assert parse_version("1.50.0rc1") == (1, 50, 0)
    assert parse_version("2") == (2,)


@pytest.mark.parametrize(
    ("version", "old"), [("1.45.1", True), ("1.46.0", False), ("1.65.0", False), ("2.0", False)]
)
def test_streamlit_floor(version: str, old: bool) -> None:
    assert streamlit_too_old(version) is old


def test_missing_requirements_reports_modules_and_old_streamlit() -> None:
    installed = {"numpy", "pandas", "sklearn", "plotly", "streamlit", "numba"}  # no umap

    def find_spec(name: str) -> object | None:
        return object() if name in installed else None

    problems = missing_requirements(find_spec, lambda: "1.40.0")
    assert problems == [
        "umap-learn is not installed",
        "streamlit 1.40.0 is older than the required 1.46",
    ]
    assert missing_requirements(lambda name: object(), lambda: "1.65.0") == []


def test_message_names_the_interpreter_and_the_fix() -> None:
    message = wrong_environment_message(["umap-learn is not installed"])
    assert "umap-learn is not installed" in message
    assert "python -m pytest" in message and "nba-app" in message
