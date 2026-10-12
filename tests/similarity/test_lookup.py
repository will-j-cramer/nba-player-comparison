from __future__ import annotations

import numpy as np
import pytest

from nba_similarity.similarity.lookup import PlayerNotFoundError, find_player_id, normalize_name
from tests.similarity.conftest import make_matrix


def matrix():
    m = make_matrix(np.zeros((4, 12)), ["jokicni01", "johnsed01", "johnsed02", "jokicni01"],
                    [2023, 1985, 1985, 2022])  # fmt: skip
    names = np.array(["Nikola Jokić", "Eddie Johnson", "Eddie Johnson", "Nikola Jokić"])
    object.__setattr__(m, "names", names)
    return m


def test_normalize_name() -> None:
    assert normalize_name("  Nikola  JOKIĆ ") == "nikola jokic"
    assert normalize_name("Shaquille O'Neal") == "shaquille oneal"


def test_finds_by_accent_insensitive_name_or_id() -> None:
    assert find_player_id(matrix(), "nikola jokic", 2023) == "jokicni01"
    assert find_player_id(matrix(), "jokicni01", 2022) == "jokicni01"


def test_ambiguous_name_asks_for_id() -> None:
    with pytest.raises(PlayerNotFoundError, match="johnsed01, johnsed02"):
        find_player_id(matrix(), "Eddie Johnson", 1985)


def test_not_found_suggests_seasons_or_names() -> None:
    with pytest.raises(
        PlayerNotFoundError, match=r"qualified seasons for that name: \[2022, 2023\]"
    ):
        find_player_id(matrix(), "Nikola Jokic", 2019)
    with pytest.raises(PlayerNotFoundError, match="did you mean: Nikola Jokić"):
        find_player_id(matrix(), "jokic", 2023)
