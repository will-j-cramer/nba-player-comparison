"""Loads the Phase 2 feature tables into arrays the similarity engine works on."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureMatrix:
    """Qualified player-seasons and their features, row-aligned.

    `z` holds the era-adjusted z-scores (NaN where a stat is missing or not covered that
    season); `raw` holds the observed stats, for display.
    """

    player_ids: np.ndarray  # (n,) str
    names: np.ndarray  # (n,) str
    seasons: np.ndarray  # (n,) int
    features: tuple[str, ...]
    categories: tuple[str, ...]
    labels: tuple[str, ...]
    z: np.ndarray  # (n, d) float, NaN = missing
    raw: np.ndarray  # (n, d) float
    # False for display-only features (shown in explanations, zero weight in the distance);
    # empty means every feature counts.
    in_distance: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        n, d = self.z.shape
        if self.raw.shape != (n, d) or len(self.features) != d:
            raise ValueError("feature matrix arrays are not aligned")
        if not self.in_distance:
            object.__setattr__(self, "in_distance", (True,) * d)
        elif len(self.in_distance) != d:
            raise ValueError("in_distance must have one flag per feature")
        if not (len(self.player_ids) == len(self.names) == len(self.seasons) == n):
            raise ValueError("feature matrix rows are not aligned")

    def row(self, player_id: str, season: int) -> int:
        """Index of a player-season, or KeyError if it is not in the matrix (e.g. < min MP)."""
        hits = np.flatnonzero((self.player_ids == player_id) & (self.seasons == season))
        if len(hits) == 0:
            raise KeyError(f"{player_id} {season} is not in the feature matrix")
        return int(hits[0])


def load_feature_matrix(conn: sqlite3.Connection) -> FeatureMatrix:
    """Read feature_definitions and player_season_features (built by `nba-features`)."""
    try:
        defs = conn.execute(
            "SELECT feature, category, label, in_distance FROM feature_definitions ORDER BY ordinal"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        raise ValueError(
            f"feature tables missing or out of date ({exc}); run nba-features"
        ) from exc
    if not defs:
        raise ValueError("no feature definitions; run nba-features first")
    features = tuple(d[0] for d in defs)
    z_cols = ", ".join(f"f.z_{f}" for f in features)
    raw_cols = ", ".join(f"f.{f}" for f in features)
    df = pd.read_sql_query(
        f"SELECT f.player_id, pl.name, f.season, {z_cols}, {raw_cols} "
        "FROM player_season_features f JOIN players pl USING (player_id) "
        "ORDER BY f.season, f.player_id",
        conn,
    )
    d = len(features)
    values = df.iloc[:, 3:].to_numpy(dtype=float)
    return FeatureMatrix(
        player_ids=df["player_id"].to_numpy(dtype=str),
        names=df["name"].to_numpy(dtype=str),
        seasons=df["season"].to_numpy(dtype=int),
        features=features,
        categories=tuple(d_[1] for d_ in defs),
        labels=tuple(d_[2] for d_ in defs),
        z=values[:, :d],
        raw=values[:, d:],
        in_distance=tuple(bool(d_[3]) for d_ in defs),
    )
