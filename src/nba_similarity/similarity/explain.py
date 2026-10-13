"""Small text helpers shared by every front end (CLI, Streamlit app, scripts)."""

from __future__ import annotations

from nba_similarity.seasons import season_label
from nba_similarity.similarity.engine import FeatureContribution

__all__ = ["format_contribution", "format_value", "season_label"]


def format_value(value: float) -> str:
    """A stat for display: 3 significant digits, "n/a" when missing."""
    return "n/a" if value != value else f"{value:.3g}"  # NaN check without numpy


def format_contribution(c: FeatureContribution) -> str:
    """One stat of a match as text, query first: "Assist % 46.6 vs 44.4"."""
    return f"{c.label} {format_value(c.query_raw)} vs {format_value(c.match_raw)}"
