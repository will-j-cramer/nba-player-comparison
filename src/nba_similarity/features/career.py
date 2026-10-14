"""Per-season rows for career trajectories, and per-player career facts (pure functions).

Every ingested player-season gets a row, so careers/ can tell "not in the NBA" from "played a
little". Each row has a status:
- "qualified": at least the season's MIN_MINUTES (the season comps' population);
- "partial":   at least CAREER_MIN_MINUTES but under MIN_MINUTES, z-scored against the
               qualified players' mean/SD (the qualified statistics are untouched);
- "limited":   under CAREER_MIN_MINUTES; no z-scores (too few minutes to mean anything).
`reliability` = min(1, MP / MIN_MINUTES) down-weights partial seasons in comparisons.

Per player: `birth_offset` = season - age (B-Ref's age on Feb 1, so constant across a
career), and `censored_through_age`: the last age at which the player may have played where
we have no data. That is the case for careers first seen in 1973-74 (they may have started
earlier) and for ABA players (the ABA is not in our NBA pages).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_similarity.features.definitions import (
    CAREER_MIN_MINUTES,
    FEATURE_NAMES,
    FIRST_OBSERVED_SEASON,
    MIN_MINUTES,
    MPG_FEATURE,
    Z_CLIP,
    basis_column,
    z_column,
)
from nba_similarity.features.normalize import add_zscores
from nba_similarity.features.qualify import season_minimums
from nba_similarity.features.transforms import add_basis, league_ft_pct

CAREER_ROW_COLUMNS: tuple[str, ...] = (
    "player_id", "season", "age", "g", "mp", "min_minutes", "career_min_minutes", "status",
    "reliability", *FEATURE_NAMES, MPG_FEATURE,
    *(z_column(f) for f in (*FEATURE_NAMES, MPG_FEATURE)),
)  # fmt: skip
CAREER_PLAYER_COLUMNS = (
    "player_id", "birth_offset", "first_season", "first_age", "censored_through_age", "censor",
)  # fmt: skip
MPG_STATS_COLUMNS = ("season", "mean", "sd")

_BASIS = {f: basis_column(f) for f in FEATURE_NAMES}


def career_rows(
    raw: pd.DataFrame, qualified: pd.DataFrame, stats: pd.DataFrame, final_seasons: set[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rows, mpg_stats): one row per ingested player-season (CAREER_ROW_COLUMNS), and the
    per-season mean/SD of minutes per game over qualified players.

    `qualified` and `stats` are the season comps' qualified rows (with basis columns) and
    per-season statistics, which the partial seasons are standardized against.
    """
    full = season_minimums(raw, final_seasons, MIN_MINUTES)
    low = season_minimums(raw, final_seasons, CAREER_MIN_MINUTES)
    out = raw.assign(min_minutes=raw["season"].map(full), career_min_minutes=raw["season"].map(low))
    mp = pd.to_numeric(out["mp"], errors="coerce").fillna(0)
    games = pd.to_numeric(out["g"], errors="coerce")
    out["status"] = np.select(
        [mp >= out["min_minutes"], mp >= out["career_min_minutes"]],
        ["qualified", "partial"],
        "limited",
    )
    out["reliability"] = np.minimum(1.0, mp / out["min_minutes"]).round(4)
    out[MPG_FEATURE] = (mp / games).where(games > 0)

    scored = out["status"] != "limited"
    based = add_basis(out[scored], league_ft=league_ft_pct(qualified))
    with_z = add_zscores(based, stats, source=_BASIS)

    mpg_q = out.loc[out["status"] == "qualified"].groupby("season")[MPG_FEATURE]
    mpg_stats = pd.DataFrame({"mean": mpg_q.mean(), "sd": mpg_q.std(ddof=0)}).reset_index()
    params = mpg_stats.set_index("season")
    mean = with_z["season"].map(params["mean"])
    sd = with_z["season"].map(params["sd"]).replace(0, np.nan)
    with_z[z_column(MPG_FEATURE)] = ((with_z[MPG_FEATURE] - mean) / sd).clip(-Z_CLIP, Z_CLIP)

    z_cols = [z_column(f) for f in (*FEATURE_NAMES, MPG_FEATURE)]
    out = out.join(with_z[z_cols])
    return out[list(CAREER_ROW_COLUMNS)].reset_index(drop=True), mpg_stats


def career_players(raw: pd.DataFrame, aba_rosters: pd.DataFrame | None) -> pd.DataFrame:
    """One row per player (CAREER_PLAYER_COLUMNS).

    `censor` explains `censored_through_age`: "pre-1974" (first seen in 1973-74, so may have
    played earlier), "ABA", both joined by "+", or "" when the whole career is observed.
    `aba_rosters` (player_id, season) is None when the ABA rosters haven't been fetched.
    """
    ages = raw.dropna(subset=["age"]).assign(offset=lambda d: d["season"] - d["age"])
    grouped = ages.groupby("player_id")
    players = pd.DataFrame(
        {
            "birth_offset": grouped["offset"].agg(lambda s: int(s.mode().iloc[0])),
            "first_season": grouped["season"].min(),
        }
    )
    players["first_age"] = players["first_season"] - players["birth_offset"]
    censored = np.full(len(players), np.nan)
    reasons = [[] for _ in range(len(players))]

    early = (players["first_season"] == FIRST_OBSERVED_SEASON).to_numpy()
    censored = np.where(early, players["first_age"] - 1, censored)
    for i in np.flatnonzero(early):
        reasons[i].append("pre-1974")

    if aba_rosters is not None and len(aba_rosters):
        last_aba = aba_rosters.groupby("player_id")["season"].max()
        aba_last = players.index.to_series().map(last_aba)
        in_aba = aba_last.notna().to_numpy()
        through = np.maximum(
            players["first_age"] - 1, aba_last - players["birth_offset"]
        ).to_numpy()
        censored = np.where(in_aba, np.fmax(censored, through), censored)
        for i in np.flatnonzero(in_aba):
            reasons[i].append("ABA")

    players["censored_through_age"] = pd.array(
        [None if np.isnan(v) else int(v) for v in censored], dtype="Int64"
    )
    players["censor"] = ["+".join(r) for r in reasons]
    return players.reset_index()[list(CAREER_PLAYER_COLUMNS)]
