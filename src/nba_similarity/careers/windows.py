"""Career windows: one slot of stats per age, compared at the same ages (pure functions).

A window "through age A" holds the last k ages up to A (never before the player's NBA
debut). Each slot is weighted geometrically toward the latest age (1, decay, decay², ...)
so a noisy teenage season can't count as much as the latest one. Concatenating the slots
keeps the trajectory: rising and falling careers with the same average stay apart.

A slot is missing when the player had no scored season at that age (injured, G League,
overseas, under CAREER_MIN_MINUTES, or not yet in the league). Missing slots stay NaN and
count as "typical" in the score (scoring.masked_scores), so nothing is filled in.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from nba_similarity.careers.data import SCORED, CareerData


def window_ages(first_age: int, through_age: int, window: int) -> list[int]:
    """The ages a window covers: the last `window` ages through `through_age`, from debut."""
    if window < 1:
        raise ValueError("window must be at least 1")
    return list(range(max(first_age, through_age - window + 1), through_age + 1))


def slot_weights(n_slots: int, decay: float) -> np.ndarray:
    """Weights for ages in ascending order; the latest age gets 1: [.., decay², decay, 1]."""
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1]")
    return decay ** np.arange(n_slots - 1, -1, -1, dtype=float)


def player_window(
    data: CareerData, player_id: str, ages: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, list[int | None]]:
    """(values (len(ages)*d,), reliability (len(ages)*d,), row per age or None).

    Values are z-scores (NaN where the slot is missing or a stat wasn't tracked); reliability
    is the season's min(1, MP / minimum), 0 for a missing slot.
    """
    d = len(data.stats)
    values = np.full(len(ages) * d, np.nan)
    reliability = np.zeros(len(ages) * d)
    rows: list[int | None] = []
    for i, age in enumerate(ages):
        row = data.row(player_id, data.season_at_age(player_id, age))
        rows.append(row)
        if row is not None and data.status[row] in SCORED:
            values[i * d : (i + 1) * d] = data.z[row]
            reliability[i * d : (i + 1) * d] = data.reliability[row]
    return values, reliability, rows


def age_moments(data: CareerData, ages: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per (age, stat) mean and variance of z over every scored season at that age: the
    score's baseline is the expected distance to a random season at the same age."""
    d = len(data.stats)
    mean = np.zeros(len(ages) * d)
    var = np.zeros(len(ages) * d)
    scored = np.isin(data.status, SCORED)
    for i, age in enumerate(ages):
        z = data.z[scored & (data.ages == age)]
        if len(z):
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # a stat with no values
                mean[i * d : (i + 1) * d] = np.nan_to_num(np.nanmean(z, axis=0))
                var[i * d : (i + 1) * d] = np.nan_to_num(np.nanvar(z, axis=0))
    return mean, var


def is_censored(data: CareerData, player_id: str, ages: Sequence[int]) -> bool:
    """True when part of the window may be missing from our data (a career first seen in
    1973-74, or begun in the ABA), so the player can't be compared over these ages."""
    through = data.players.at[player_id, "censored_through_age"]
    return bool(through is not None and not np.isnan(through) and min(ages) <= through)


def experience(data: CareerData, player_id: str, through_season: int) -> int:
    """NBA seasons played (any minutes) up to and including `through_season`."""
    return sum(1 for s in data.player_seasons(player_id) if s <= through_season)
