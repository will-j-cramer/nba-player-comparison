"""Checks that the running interpreter has what the project needs.

On machines with conda, a bare `pytest` or `streamlit` can resolve to conda's base install
instead of the project's virtualenv, which fails in confusing ways (missing umap, an older
Streamlit). These checks turn that into one clear message.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import sys
from collections.abc import Callable

MIN_STREAMLIT = (1, 46)  # st.navigation(position="top") and st.context.theme
REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "plotly": "plotly",
    "streamlit": "streamlit",
    "numba": "numba",
    "umap": "umap-learn",
}


def parse_version(text: str) -> tuple[int, ...]:
    """Leading numeric parts of a version: "1.46.0" -> (1, 46, 0), "1.50.0rc1" -> (1, 50, 0)."""
    parts = []
    for piece in text.split("."):
        match = re.match(r"\d+", piece)
        if match is None:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def streamlit_too_old(version: str) -> bool:
    """True when `version` is older than MIN_STREAMLIT."""
    return parse_version(version)[: len(MIN_STREAMLIT)] < MIN_STREAMLIT


def missing_requirements(
    find_spec: Callable[[str], object] = importlib.util.find_spec,
    streamlit_version: Callable[[], str] = lambda: importlib.metadata.version("streamlit"),
) -> list[str]:
    """Problems with the current interpreter, as readable lines (empty when all is well)."""
    problems = [
        f"{package} is not installed"
        for module, package in REQUIRED_MODULES.items()
        if find_spec(module) is None
    ]
    if find_spec("streamlit") is not None:
        version = streamlit_version()
        if streamlit_too_old(version):
            minimum = ".".join(map(str, MIN_STREAMLIT))
            problems.append(f"streamlit {version} is older than the required {minimum}")
    return problems


def wrong_environment_message(problems: list[str]) -> str:
    """The message shown when the interpreter is not the project's virtualenv."""
    return (
        f"This interpreter ({sys.executable}) can't run the project: "
        + "; ".join(problems)
        + ". It is probably not the project's virtualenv (conda's base environment can "
        "shadow it). Activate it with `source .venv/bin/activate`, run "
        "`pip install -r requirements.txt`, then use `python -m pytest` and `nba-app`."
    )
