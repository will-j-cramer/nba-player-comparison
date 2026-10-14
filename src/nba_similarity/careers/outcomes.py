"""What comps did after the window, and a score-weighted projection from it (pure functions).

Each future age of a comp is one of:
- "qualified": a season at or above the season's MIN_MINUTES; its stats are used;
- "limited":   played in the NBA, but under MIN_MINUTES;
- "out":       no NBA season (injury, G League, overseas, retired; we can't tell which).
Leaving the league is an outcome, not missing data: dropping those comps would make every
projection look rosy. Only comps whose whole horizon is in the past ("observable") count.

Stat projections are weighted quantiles over the comps who qualified at that age, as
within-season percentiles (era-fair) and "in today's terms": the z-score mapped through the
latest season's mean/SD and the inverse of the basis transform.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from nba_similarity.careers.data import CareerData

CLASSES = ("qualified", "limited", "out")
QUANTILES = (0.25, 0.5, 0.75)
MIN_QUALIFIED = 3  # fewer qualified comps at an age: no stat projection for it


@dataclass(frozen=True)
class FutureSeason:
    """One comp's season at one future age."""

    age: int
    season: int
    outcome: str  # one of CLASSES
    z: np.ndarray | None  # (d,) when qualified
    raw: np.ndarray | None
    mp: float  # 0 when out of the league


@dataclass(frozen=True)
class Outcome:
    """A comp's next `horizon` seasons; `observable` False when some are still to be played."""

    observable: bool
    seasons: tuple[FutureSeason, ...]


def comp_outcome(data: CareerData, player_id: str, through_age: int, horizon: int) -> Outcome:
    """The comp's seasons at ages through_age+1 .. through_age+horizon, up to the first season
    we don't have (not played yet, or not ingested). Observable only if none is missing."""
    future = []
    end = data.season_at_age(player_id, through_age)
    for step in range(1, horizon + 1):
        season = end + step
        if season not in data.final_seasons:
            break
        row = data.row(player_id, season)
        if row is None:
            future.append(FutureSeason(through_age + step, season, "out", None, None, 0.0))
        elif data.status[row] == "qualified":
            future.append(
                FutureSeason(
                    through_age + step,
                    season,
                    "qualified",
                    data.z[row],
                    data.raw[row],
                    float(data.mp[row]),
                )
            )
        else:
            future.append(
                FutureSeason(through_age + step, season, "limited", None, None, float(data.mp[row]))
            )
    return Outcome(observable=len(future) == horizon, seasons=tuple(future))


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float] = QUANTILES
) -> np.ndarray:
    """Quantiles of `values` with `weights` (interpolated on cumulative weight midpoints)."""
    keep = ~np.isnan(values) & (weights > 0)
    values, weights = values[keep], weights[keep]
    if len(values) == 0:
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cum = (np.cumsum(weights) - weights / 2) / weights.sum()
    return np.interp(quantiles, cum, values)


def percentile(z: float) -> float:
    """Within-season percentile implied by a z-score (normal approximation)."""
    return float("nan") if math.isnan(z) else 50 * (1 + math.erf(z / math.sqrt(2)))


def todays_terms(data: CareerData, stat: int, z: float) -> float:
    """A z-score as a value in the latest season's terms (inverse of the basis transform)."""
    season = data.last_season
    mean = data.season_mean.at[season, data.stats[stat]]
    sd = data.season_sd.at[season, data.stats[stat]]
    if math.isnan(z) or mean is None or sd is None or np.isnan(mean) or np.isnan(sd):
        return float("nan")
    basis = mean + z * sd
    if data.transforms[stat] == "sqrt":
        return max(basis, 0.0) ** 2
    return float(basis)  # "none"; "shrink" (FT%) is read as an FT% directly


@dataclass(frozen=True)
class AgeProjection:
    """The projection for one future age."""

    age: int
    shares: dict[str, float]  # CLASSES -> score-weighted share of comps
    n_qualified: int
    pct: np.ndarray  # (d, 3) percentiles at QUANTILES; NaN without enough comps
    today: np.ndarray  # (d, 3) the same in the latest season's terms


@dataclass(frozen=True)
class Projection:
    """Score-weighted outcomes of the observable comps, per future age."""

    n_comps: int
    ages: tuple[AgeProjection, ...]


def project(
    data: CareerData, outcomes: Sequence[Outcome], scores: Sequence[float], horizon: int
) -> Projection:
    """Combine observable comps' outcomes, weighted by similarity score."""
    pairs = [(o, max(s, 0.0)) for o, s in zip(outcomes, scores, strict=True) if o.observable]
    d = len(data.stats)
    ages = []
    total = sum(w for _, w in pairs)
    for step in range(horizon):
        shares = dict.fromkeys(CLASSES, 0.0)
        zs, ws = [], []
        for outcome, w in pairs:
            season = outcome.seasons[step]
            shares[season.outcome] += w / total if total else 0.0
            if season.outcome == "qualified":
                zs.append(season.z)
                ws.append(w)
        pct = np.full((d, len(QUANTILES)), np.nan)
        today = np.full((d, len(QUANTILES)), np.nan)
        if len(zs) >= MIN_QUALIFIED:
            z_matrix, weights = np.array(zs), np.array(ws)
            for j in range(d):
                q = weighted_quantiles(z_matrix[:, j], weights)
                pct[j] = [percentile(v) for v in q]
                today[j] = [todays_terms(data, j, v) for v in q]
        age = pairs[0][0].seasons[step].age if pairs else step + 1
        ages.append(AgeProjection(age, shares, len(zs), pct, today))
    return Projection(n_comps=len(pairs), ages=tuple(ages))
