"""Small text helpers shared by every front end (CLI, Streamlit app, scripts)."""

from __future__ import annotations

from nba_similarity.similarity.engine import FeatureContribution


def season_label(season: int) -> str:
    """Season end year to the usual label: 2023 -> "2022-23"."""
    return f"{season - 1}-{str(season)[2:]}"


def format_value(value: float) -> str:
    return "n/a" if value != value else f"{value:.3g}"  # NaN check without numpy


def format_contribution(c: FeatureContribution) -> str:
    """ "Assist % 46.6 vs 44.4" (query first)."""
    return f"{c.label} {format_value(c.query_raw)} vs {format_value(c.match_raw)}"
