"""Resolves a player name (as typed by a person) to a B-Ref player id."""

from __future__ import annotations

import unicodedata

import numpy as np

from nba_similarity.similarity.data import FeatureMatrix


class PlayerNotFoundError(LookupError):
    pass


def normalize_name(name: str) -> str:
    """Lowercase, accents stripped, punctuation-insensitive: "Nikola Jokić" -> "nikola jokic"."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join("".join(c for c in ascii_only.lower() if c.isalnum() or c == " ").split())


def find_player_id(matrix: FeatureMatrix, name_or_id: str, season: int) -> str:
    """The player id for a name (accent-insensitive) or id who has `season` in the matrix."""
    in_season = matrix.seasons == season
    if np.any(in_season & (matrix.player_ids == name_or_id)):
        return name_or_id

    target = normalize_name(name_or_id)
    names = np.array([normalize_name(n) for n in matrix.names])
    exact = np.unique(matrix.player_ids[in_season & (names == target)])
    if len(exact) == 1:
        return str(exact[0])
    if len(exact) > 1:
        raise PlayerNotFoundError(
            f"{name_or_id!r} matches several players in {season}: {', '.join(exact)}; "
            "pass the player id instead"
        )

    partial = sorted({str(n) for n in matrix.names[in_season & (np.char.find(names, target) >= 0)]})
    other_seasons = sorted({int(s) for s in matrix.seasons[names == target]})
    hint = ""
    if partial:
        hint = f"; did you mean: {', '.join(partial[:5])}"
    elif other_seasons:
        hint = f"; qualified seasons for that name: {other_seasons}"
    raise PlayerNotFoundError(
        f"no qualified player-season for {name_or_id!r} in {season} "
        f"(seasons below the minimum minutes are not in the matrix){hint}"
    )
