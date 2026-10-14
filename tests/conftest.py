"""Stops the run with one clear message when pytest runs outside the project's environment."""

from __future__ import annotations

import pytest

from nba_similarity.environment import missing_requirements, wrong_environment_message


def pytest_configure(config: pytest.Config) -> None:
    """Exit before collection if the interpreter lacks the project's dependencies."""
    problems = missing_requirements()
    if problems:
        pytest.exit(wrong_environment_message(problems), returncode=4)
