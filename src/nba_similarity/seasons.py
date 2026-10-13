"""Season labels. Seasons are identified by their end year everywhere (2023 = 2022-23)."""

from __future__ import annotations


def season_label(season: int) -> str:
    """Season end year to the usual label: 2023 -> "2022-23"."""
    return f"{season - 1}-{str(season)[2:]}"
