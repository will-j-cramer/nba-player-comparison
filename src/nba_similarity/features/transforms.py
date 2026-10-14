"""Turns raw feature values into the basis values that get z-scored (pure functions).

- "none":   basis = raw value.
- "sqrt":   basis = sqrt(raw), for right-skewed stats (3PA per 100, FT rate, ORB%, BLK%), so the
            most extreme seasons stay distinguishable instead of all hitting the z clip.
- "shrink": FT% blended with the season's league FT% as if the player had taken
            FT_PRIOR_ATTEMPTS extra league-average attempts (empirical-Bayes style), so a
            10-for-10 season doesn't look like elite shooting. Players with no attempts stay
            NULL: nothing is invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_similarity.features.definitions import (
    FEATURES,
    FT_PRIOR_ATTEMPTS,
    POSS_PER_48,
    basis_column,
)


def add_basis(df: pd.DataFrame, league_ft: pd.Series | None = None) -> pd.DataFrame:
    """Add a basis_<feature> column per feature; raw columns are left unchanged.

    The "shrink" transform needs `ft_per100`, `fta_per100` and `mp` columns. The league FT%
    (by season) is computed over the rows passed in (the qualified players) unless
    `league_ft` gives it, e.g. when transforming other rows against the qualified league.
    """
    out = df.copy()
    for f in FEATURES:
        raw = pd.to_numeric(out[f.name], errors="coerce")
        if f.transform == "sqrt":
            out[basis_column(f.name)] = np.sqrt(raw)
        elif f.transform == "shrink":
            out[basis_column(f.name)] = shrunk_ft_pct(out, raw, league_ft)
        else:
            out[basis_column(f.name)] = raw
    return out


def estimated_attempts(per100: pd.Series, mp: pd.Series) -> pd.Series:
    """Season total from a per-100-possession rate, assuming POSS_PER_48 possessions per 48."""
    return per100 * mp / 48 * POSS_PER_48 / 100


def league_ft_pct(df: pd.DataFrame) -> pd.Series:
    """League FT% by season over the rows of `df` (estimated makes / estimated attempts)."""
    mp = pd.to_numeric(df["mp"], errors="coerce")
    fta = estimated_attempts(pd.to_numeric(df["fta_per100"], errors="coerce"), mp)
    ftm = estimated_attempts(pd.to_numeric(df["ft_per100"], errors="coerce"), mp)
    has_attempts = fta.gt(0) & pd.to_numeric(df["ft_pct"], errors="coerce").notna()
    totals = pd.DataFrame(
        {"season": df["season"], "ftm": ftm.where(has_attempts), "fta": fta.where(has_attempts)}
    )
    by_season = totals.groupby("season")[["ftm", "fta"]].sum()
    return by_season["ftm"] / by_season["fta"]


def shrunk_ft_pct(
    df: pd.DataFrame, ft_pct: pd.Series, league_ft: pd.Series | None = None
) -> pd.Series:
    """FT% shrunk toward the season's league FT% by FT_PRIOR_ATTEMPTS league-average attempts
    (attempts estimated from FTA per 100 and minutes); NaN for players with no attempts.
    `league_ft` (by season) defaults to the league FT% over `df`'s own rows.
    """
    mp = pd.to_numeric(df["mp"], errors="coerce")
    fta = estimated_attempts(pd.to_numeric(df["fta_per100"], errors="coerce"), mp)
    has_attempts = fta.gt(0) & ft_pct.notna()
    league_by_season = league_ft_pct(df) if league_ft is None else league_ft
    league = df["season"].map(league_by_season)

    shrunk = (fta * ft_pct + FT_PRIOR_ATTEMPTS * league) / (fta + FT_PRIOR_ATTEMPTS)
    return shrunk.where(has_attempts)
