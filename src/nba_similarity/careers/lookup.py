"""Resolves a typed player name to a B-Ref id in the career data."""

from __future__ import annotations

from nba_similarity.careers.data import CareerData
from nba_similarity.similarity.lookup import PlayerNotFoundError, normalize_name


def find_career_player(data: CareerData, name_or_id: str) -> str:
    """The player id for a name (accent-insensitive) or id; PlayerNotFoundError otherwise."""
    if name_or_id in data.players.index:
        return name_or_id
    target = normalize_name(name_or_id)
    names = data.players["name"].map(normalize_name)
    exact = sorted(names.index[names == target])
    if len(exact) == 1:
        return str(exact[0])
    if len(exact) > 1:
        raise PlayerNotFoundError(
            f"{name_or_id!r} matches several players: {', '.join(exact)}; pass the player id"
        )
    partial = sorted(data.players.loc[names.str.contains(target, regex=False), "name"])
    hint = f"; did you mean: {', '.join(partial[:5])}" if partial else ""
    raise PlayerNotFoundError(f"no player named {name_or_id!r}{hint}")
