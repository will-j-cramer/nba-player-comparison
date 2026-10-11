"""Era adjustment: per-season z-scores over qualified players, with coverage rules.

For each season and feature, the mean and population standard deviation are computed over
qualified players who have a value. A feature is "covered" in a season when at least
COVERAGE_MIN of qualified players have it and its spread is non-zero; otherwise every z-score
for that season and feature is NULL (raw values are kept for display). Missing values are
never filled in.
"""

from __future__ import annotations

import pandas as pd

from nba_similarity.features.definitions import COVERAGE_MIN, FEATURE_NAMES, Z_CLIP, z_column

SEASON_STATS_COLUMNS = ("season", "feature", "n", "mean", "sd", "coverage", "covered")


def season_stats(df: pd.DataFrame, features: tuple[str, ...] = FEATURE_NAMES) -> pd.DataFrame:
    """One row per (season, feature): n, mean, sd (population), coverage share, covered flag."""
    rows = []
    for season, group in df.groupby("season", sort=True):
        for feature in features:
            values = group[feature].dropna()
            n = len(values)
            mean = float(values.mean()) if n else None
            sd = float(values.std(ddof=0)) if n else None
            coverage = n / len(group) if len(group) else 0.0
            covered = coverage >= COVERAGE_MIN and n >= 2 and bool(sd)
            rows.append((int(season), feature, n, mean, sd, coverage, covered))
    return pd.DataFrame(rows, columns=list(SEASON_STATS_COLUMNS))


def add_zscores(
    df: pd.DataFrame, stats: pd.DataFrame, features: tuple[str, ...] = FEATURE_NAMES
) -> pd.DataFrame:
    """Add a clipped z-score column per feature and `n_features` (non-null z-scores per row)."""
    out = df.copy()
    indexed = stats.set_index(["season", "feature"])
    for feature in features:
        params = indexed.xs(feature, level="feature")
        covered = out["season"].map(params["covered"]).fillna(False).astype(bool)
        mean = out["season"].map(params["mean"])
        sd = out["season"].map(params["sd"])
        z = ((out[feature] - mean) / sd).clip(-Z_CLIP, Z_CLIP)
        out[z_column(feature)] = z.where(covered & out[feature].notna())
    out["n_features"] = out[[z_column(f) for f in features]].notna().sum(axis=1)
    return out
