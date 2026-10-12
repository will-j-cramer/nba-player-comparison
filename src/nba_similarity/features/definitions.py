"""The features compared for similarity, and the rules that turn raw stats into them.

Every feature is a rate (per 100 possessions or a share of opportunities), so seasons of
different lengths and paces compare fairly. Composite metrics (PER, BPM, WS, VORP) are left
out because they would double-count these inputs and make explanations vague.

Z-scores are computed on a "basis" value: right-skewed stats are square-rooted first so the
extremes don't all pile up at the clip, and FT% is shrunk toward the league average to damp
small samples. Raw (observed) values are always kept for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Transform = Literal["none", "sqrt", "shrink"]
"""How a raw value becomes the basis that is z-scored (see transforms.py)."""

CATEGORIES: tuple[str, ...] = ("scoring", "efficiency", "playmaking", "rebounding", "defense")


@dataclass(frozen=True)
class FeatureSpec:
    name: str  # column name in player_season_features (z-score column is z_<name>)
    table: str  # ingestion table the raw value comes from
    column: str  # column in that table
    category: str
    label: str  # human-readable, for summaries and explanations
    transform: Transform = "none"


FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec("pts", "per100_stats", "pts", "scoring", "Points per 100 poss"),
    FeatureSpec("usg_pct", "advanced_stats", "usg_pct", "scoring", "Usage %"),
    FeatureSpec(
        "fg3a_rate", "advanced_stats", "fg3a_rate", "scoring", "3PA rate (3PA/FGA)", "sqrt"
    ),
    FeatureSpec("fta_rate", "advanced_stats", "fta_rate", "scoring", "FT rate (FTA/FGA)", "sqrt"),
    FeatureSpec("ts_pct", "advanced_stats", "ts_pct", "efficiency", "True shooting %"),
    FeatureSpec("ft_pct", "per100_stats", "ft_pct", "efficiency", "Free throw %", "shrink"),
    FeatureSpec("ast_pct", "advanced_stats", "ast_pct", "playmaking", "Assist %"),
    FeatureSpec("tov_pct", "advanced_stats", "tov_pct", "playmaking", "Turnover %"),
    FeatureSpec(
        "orb_pct", "advanced_stats", "orb_pct", "rebounding", "Offensive rebound %", "sqrt"
    ),
    FeatureSpec("drb_pct", "advanced_stats", "drb_pct", "rebounding", "Defensive rebound %"),
    FeatureSpec("stl_pct", "advanced_stats", "stl_pct", "defense", "Steal %"),
    FeatureSpec("blk_pct", "advanced_stats", "blk_pct", "defense", "Block %", "sqrt"),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)

# Minimum minutes for a full 82-game season; scaled down for shortened seasons.
MIN_MINUTES = 500
FULL_SCHEDULE = 82
SHORT_SCHEDULES: dict[int, int] = {  # season end year -> games per team
    1999: 50,  # lockout
    2012: 66,  # lockout
    2020: 71,  # COVID suspension (teams played 64-75; 1,059 games / 30 teams)
    2021: 72,  # COVID-shortened schedule
}

# A stat is used for a season only if at least this share of qualified players have it
# (e.g. 1973-74 has turnovers only for Seattle players, so TOV% is not used that season).
COVERAGE_MIN = 0.90

# z-scores are clipped to +/- this so one extreme value cannot dominate a distance.
Z_CLIP = 4.0

# "shrink" (FT%): blend each player's FT% with the season's league FT% as if they had taken
# this many extra league-average attempts, so 10-for-10 doesn't read as an elite shooter.
FT_PRIOR_ATTEMPTS = 50
# Attempts are estimated from per-100 rates: FTA ~= FTA/100 * MP/48 * POSS_PER_48/100.
POSS_PER_48 = 100


def z_column(feature: str) -> str:
    return f"z_{feature}"


def basis_column(feature: str) -> str:
    return f"basis_{feature}"
