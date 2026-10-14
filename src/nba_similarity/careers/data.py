"""Loads the career tables built by `nba-features` into arrays for fast window lookups."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_similarity.features.definitions import MPG_CATEGORY, MPG_FEATURE, MPG_LABEL

SCORED = ("qualified", "partial")  # statuses with z-scores (limited seasons have none)


@dataclass(frozen=True)
class CareerData:
    """Every ingested player-season (row-aligned arrays) plus per-player career facts."""

    stats: tuple[str, ...]  # the 12 season features + minutes per game
    labels: tuple[str, ...]
    categories: tuple[str, ...]
    in_distance: tuple[bool, ...]  # points per 100 stays display-only
    transforms: tuple[str, ...]  # "none" / "sqrt" / "shrink" (for today's-terms values)
    player_ids: np.ndarray  # (n,) str
    seasons: np.ndarray  # (n,) int
    ages: np.ndarray  # (n,) int
    mp: np.ndarray  # (n,) float
    status: np.ndarray  # (n,) "qualified" / "partial" / "limited"
    reliability: np.ndarray  # (n,) min(1, MP / season minimum)
    z: np.ndarray  # (n, d) NaN = missing or not scored
    raw: np.ndarray  # (n, d) observed values, for display
    players: pd.DataFrame  # index player_id: name, birth_offset, first_season, first_age,
    #                        censored_through_age (nullable), censor
    season_mean: pd.DataFrame  # index season, columns stats: basis mean (for today's terms)
    season_sd: pd.DataFrame
    last_season: int  # latest final (complete) season
    final_seasons: frozenset[int] = frozenset()  # seasons fully ingested and complete

    def __post_init__(self) -> None:
        index = {
            (p, int(s)): i
            for i, (p, s) in enumerate(zip(self.player_ids, self.seasons, strict=True))
        }
        object.__setattr__(self, "_index", index)
        by_player: dict[str, list[int]] = {}
        for p, s in zip(self.player_ids, self.seasons, strict=True):
            by_player.setdefault(str(p), []).append(int(s))
        object.__setattr__(self, "_seasons", {p: sorted(v) for p, v in by_player.items()})

    def row(self, player_id: str, season: int) -> int | None:
        """Row of a player-season, or None if the player has no NBA row that season."""
        return self._index.get((player_id, int(season)))  # type: ignore[attr-defined]

    def player_seasons(self, player_id: str) -> list[int]:
        """Every season the player has an NBA row for (any minutes), ascending."""
        return self._seasons.get(player_id, [])  # type: ignore[attr-defined]

    def season_at_age(self, player_id: str, age: int) -> int:
        """The season (end year) in which the player was `age` (B-Ref's Feb 1 age)."""
        return int(self.players.at[player_id, "birth_offset"]) + age

    def name(self, player_id: str) -> str:
        """Display name."""
        return str(self.players.at[player_id, "name"])


def load_career_data(conn: sqlite3.Connection) -> CareerData:
    """Read the career tables; ValueError if `nba-features` hasn't built them yet."""
    try:
        defs = conn.execute(
            "SELECT feature, category, label, in_distance, transform FROM feature_definitions "
            "ORDER BY ordinal"
        ).fetchall()
        rows = pd.read_sql_query(
            "SELECT * FROM player_season_career_features ORDER BY season, player_id", conn
        )
        players = pd.read_sql_query(
            "SELECT c.*, p.name FROM career_players c JOIN players p USING (player_id)", conn
        ).set_index("player_id")
        stats = pd.read_sql_query(
            "SELECT season, feature, mean, sd FROM feature_season_stats", conn
        )
        mpg = pd.read_sql_query("SELECT season, mean, sd FROM career_season_stats", conn)
        finals = {
            s for (s,) in conn.execute("SELECT season FROM ingested_seasons WHERE is_final = 1")
        }
    except (sqlite3.OperationalError, pd.errors.DatabaseError) as exc:
        raise ValueError(f"career tables missing ({exc}); re-run nba-features") from exc
    if not defs or rows.empty:
        raise ValueError("no career data; run nba-ingest and nba-features first")

    names = (*(d[0] for d in defs), MPG_FEATURE)
    stats = pd.concat([stats, mpg.assign(feature=MPG_FEATURE)], ignore_index=True)
    return CareerData(
        stats=names,
        labels=(*(d[2] for d in defs), MPG_LABEL),
        categories=(*(d[1] for d in defs), MPG_CATEGORY),
        in_distance=(*(bool(d[3]) for d in defs), True),
        transforms=(*(d[4] for d in defs), "none"),
        player_ids=rows["player_id"].to_numpy(dtype=str),
        seasons=rows["season"].to_numpy(dtype=int),
        ages=rows["age"].to_numpy(dtype=float).astype(int),
        mp=rows["mp"].to_numpy(dtype=float),
        status=rows["status"].to_numpy(dtype=str),
        reliability=rows["reliability"].to_numpy(dtype=float),
        z=rows[[f"z_{f}" for f in names]].to_numpy(dtype=float),
        raw=rows[list(names)].to_numpy(dtype=float),
        players=players,
        season_mean=stats.pivot(index="season", columns="feature", values="mean")[list(names)],
        season_sd=stats.pivot(index="season", columns="feature", values="sd")[list(names)],
        last_season=max(finals) if finals else int(rows["season"].max()),
        final_seasons=frozenset(finals),
    )
