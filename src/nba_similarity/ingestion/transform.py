"""Pure transformation of parsed table rows into one PlayerSeason per player.

Traded players appear on Basketball-Reference once per team plus a combined row (labelled
`TOT`, or `2TM`/`3TM`/... since the 2024 site redesign). We keep only the combined row as the
player's season, normalized to team COMBINED_TEAM, and record the per-team rows as stints:
similarity compares whole seasons, partial stints would duplicate players in results, and
B-Ref computes advanced stats on the combined row over the full season.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence

from nba_similarity.ingestion.records import (
    ADVANCED_COLUMNS,
    COMBINED_TEAM,
    PER100_COLUMNS,
    PlayerSeason,
    RawRow,
    Stint,
)

logger = logging.getLogger(__name__)

_COMBINED_TEAM_RE = re.compile(r"^(TOT|\d+TM)$")


def is_combined_team(team: str | None) -> bool:
    """True for B-Ref's combined row of a traded player (TOT, or 2TM/3TM/... since 2024)."""
    return team is not None and bool(_COMBINED_TEAM_RE.match(team))


def build_season(
    season: int, per100_rows: Sequence[RawRow], advanced_rows: Sequence[RawRow]
) -> list[PlayerSeason]:
    """Merge one season's per-100 and advanced rows into one PlayerSeason per player.

    Identity fields and stints come from the advanced page when a player is on it. A player
    missing from one page is kept, with that page's stats left as None.
    """
    per100 = collapse_traded(per100_rows)
    advanced = collapse_traded(advanced_rows)

    only_per100 = per100.keys() - advanced.keys()
    only_advanced = advanced.keys() - per100.keys()
    if only_per100 or only_advanced:
        logger.warning(
            "season %d: %d players only on per-100 page, %d only on advanced page",
            season,
            len(only_per100),
            len(only_advanced),
        )

    result = []
    for player_id in _ordered_union(advanced, per100):
        adv = advanced.get(player_id)
        p100 = per100.get(player_id)
        combined, stint_rows = adv or p100  # type: ignore[misc]
        result.append(
            PlayerSeason(
                player_id=player_id,
                name=_required(combined, "name"),
                season=season,
                age=to_int(combined.get("age")),
                pos=combined.get("pos"),
                team=COMBINED_TEAM if len(stint_rows) > 1 else _required(combined, "team"),
                g=to_int(combined.get("g")),
                gs=to_int(combined.get("gs")),
                mp=to_int(combined.get("mp")),
                stints=tuple(
                    Stint(_required(r, "team"), to_int(r.get("g")), to_int(r.get("mp")))
                    for r in stint_rows
                ),
                per100=_stats(p100[0], PER100_COLUMNS) if p100 else None,
                advanced=_stats(adv[0], ADVANCED_COLUMNS) if adv else None,
            )
        )
    return result


def collapse_traded(rows: Iterable[RawRow]) -> dict[str, tuple[RawRow, list[RawRow]]]:
    """Group rows by player: player_id -> (whole-season row, stint rows in page order).

    A non-traded player's single row is both the season row and its only stint.
    """
    grouped: dict[str, list[RawRow]] = {}
    for row in rows:
        grouped.setdefault(_required(row, "player_id"), []).append(row)

    result = {}
    for player_id, player_rows in grouped.items():
        combined = [r for r in player_rows if is_combined_team(r.get("team"))]
        stints = [r for r in player_rows if not is_combined_team(r.get("team"))]
        if len(combined) > 1 or not stints:
            raise ValueError(f"{player_id}: unexpected rows {[r.get('team') for r in player_rows]}")
        if combined:
            result[player_id] = (combined[0], stints)
        elif len(stints) == 1:
            result[player_id] = (stints[0], stints)
        else:
            raise ValueError(f"{player_id}: {len(stints)} team rows but no combined row")
    return result


def to_int(value: str | None) -> int | None:
    """Cell text to int ("1,234" -> 1234); None for an empty cell."""
    if value is None or value == "":
        return None
    return int(float(value.replace(",", "")))


def to_float(value: str | None) -> float | None:
    """Cell text to float (".512" -> 0.512); None for an empty cell."""
    if value is None or value == "":
        return None
    return float(value.replace(",", ""))


def _stats(row: RawRow, columns: Sequence[str]) -> dict[str, float | None]:
    return {col: to_float(row.get(col)) for col in columns}


def _required(row: RawRow, key: str) -> str:
    value = row.get(key)
    if not value:
        raise ValueError(f"row is missing {key!r}: {row}")
    return value


def _ordered_union(*mappings: dict[str, object]) -> list[str]:
    seen: dict[str, None] = {}
    for mapping in mappings:
        seen.update(dict.fromkeys(mapping))
    return list(seen)
