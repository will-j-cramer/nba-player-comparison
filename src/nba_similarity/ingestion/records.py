"""Typed records produced by ingestion, and the stat columns stored for each source page.

The column tuples are the single source of truth: the SQLite schema, the parser's output keys
and the transform all derive from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Values are stored exactly as Basketball-Reference reports them: shooting percentages and TS%
# as fractions (0.583), rate stats such as usg_pct or ast_pct as percentages (27.2).
PER100_COLUMNS: tuple[str, ...] = (
    "fg", "fga", "fg_pct",
    "fg3", "fg3a", "fg3_pct",
    "fg2", "fg2a", "fg2_pct",
    "ft", "fta", "ft_pct",
    "orb", "drb", "trb",
    "ast", "stl", "blk", "tov", "pf", "pts",
    "ortg", "drtg",
)  # fmt: skip

ADVANCED_COLUMNS: tuple[str, ...] = (
    "per", "ts_pct", "fg3a_rate", "fta_rate",
    "orb_pct", "drb_pct", "trb_pct",
    "ast_pct", "stl_pct", "blk_pct", "tov_pct", "usg_pct",
    "ows", "dws", "ws", "ws_per_48",
    "obpm", "dbpm", "bpm", "vorp",
)  # fmt: skip

# Identity fields every parsed row carries, in addition to its page's stat columns.
IDENTITY_COLUMNS: tuple[str, ...] = ("player_id", "name", "age", "pos", "team", "g", "gs", "mp")

# Team code used for a traded player's combined (whole-season) row.
COMBINED_TEAM = "TOT"

RawRow = dict[str, str | None]
"""One table row as parsed from HTML: column name -> cell text, None for an empty cell."""


@dataclass(frozen=True)
class Stint:
    team: str
    g: int | None
    mp: int | None


@dataclass(frozen=True)
class PlayerSeason:
    """One player's whole season: the combined row if they were traded mid-season."""

    player_id: str
    name: str
    season: int  # end year: 2023 = 2022-23
    age: int | None
    pos: str | None
    team: str  # team code, or COMBINED_TEAM if traded
    g: int | None
    gs: int | None
    mp: int | None
    stints: tuple[Stint, ...]
    per100: dict[str, float | None] | None = field(default=None)
    advanced: dict[str, float | None] | None = field(default=None)

    @property
    def n_teams(self) -> int:
        return len(self.stints)
