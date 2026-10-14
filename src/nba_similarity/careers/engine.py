"""Career comps: the players whose careers through the same age looked most alike.

The query's window (careers/windows.py) is compared with every other player's window over
the same ages, using the season engine's scoring (similarity/scoring.masked_scores): the
weighted euclidean score 0-100 with a query-specific baseline (the expected distance to a
random season at the same age, from every scored season at that age), exact per-feature
contributions, and missing values counted as "typical". Here a "feature" is a (stat, age)
pair, weighted by stat weight × slot weight × the query season's reliability.

A candidate's partial seasons (250-499 minutes) are down-weighted by their reliability: the
slot's points are scaled by min(1, MP / minimum), which blends the observed comparison with
"typical" (0 points). A missing slot is the limit, 0 points but still in the total.

Candidates need a scored season at the window's last age, an uncensored window (careers
first seen in 1973-74 or begun in the ABA may be missing early seasons) and at least
`min_shared` of the window's weight in common with the query.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from nba_similarity.careers.data import SCORED, CareerData
from nba_similarity.careers.outcomes import Outcome, Projection, comp_outcome, project
from nba_similarity.careers.windows import (
    age_moments,
    experience,
    is_censored,
    player_window,
    slot_weights,
    window_ages,
)
from nba_similarity.similarity.config import feature_weights
from nba_similarity.similarity.scoring import masked_scores


@dataclass(frozen=True)
class CareerConfig:
    """How careers are compared and projected.

    - window: number of ages in the window (the last `window` ages through the query age).
    - decay: slot weight per year back (latest age 1, the one before `decay`, ...).
    - horizon: future seasons shown and projected.
    - n: comps returned; n_project: observable comps used for the projection.
    - min_shared: share of the window's weight a candidate must have in common.
    - category_weights: multipliers as in the season comps (the "role" category is minutes).
    - max_experience_gap: optional limit on |NBA seasons played| difference at the window end.
    - close_z: a stat is "alike" when the seasons are within this many SD on average.
    """

    window: int = 3
    decay: float = 0.6
    horizon: int = 5
    n: int = 10
    n_project: int = 20
    min_shared: float = 0.6
    category_weights: Mapping[str, float] = field(default_factory=dict)
    max_experience_gap: int | None = None
    close_z: float = 1.0
    top_traits: int = 3

    def __post_init__(self) -> None:
        if not 1 <= self.window <= 6:
            raise ValueError("window must be between 1 and 6 ages")
        if not 1 <= self.horizon <= 10:
            raise ValueError("horizon must be between 1 and 10 seasons")
        if not 0 < self.min_shared <= 1:
            raise ValueError("min_shared must be in (0, 1]")
        if self.n < 1 or self.n_project < 1:
            raise ValueError("n and n_project must be at least 1")


@dataclass(frozen=True)
class CareerTrait:
    """One stat's part in a career match, summed over the window's ages."""

    stat: str
    label: str
    points: float
    gap: float  # weighted mean |z difference| over the ages both have
    ages: tuple[int, ...]  # ages both seasons have the stat
    query_raw: float  # at the latest shared age
    comp_raw: float


@dataclass(frozen=True)
class CareerComp:
    """One comp: score, window coverage, explanation and what happened next."""

    player_id: str
    name: str
    score: float
    shared: float
    experience: int
    window_seasons: tuple[int | None, ...]  # season played at each window age, None if none
    similar_traits: tuple[CareerTrait, ...]
    differences: tuple[CareerTrait, ...]
    contributions: dict[tuple[int, str], float]  # (age, stat) -> points; sums to score
    outcome: Outcome


@dataclass(frozen=True)
class CareerResult:
    """Comps for one player through one age, and the projection from them."""

    player_id: str
    name: str
    through_age: int
    ages: tuple[int, ...]
    query_seasons: tuple[int | None, ...]
    query_experience: int
    comps: tuple[CareerComp, ...]  # best first; outcome may be pending (not observable)
    projection: Projection
    stats: tuple[str, ...] = ()  # column order of the projection's (d, 3) arrays


class CareerEngine:
    """Build once over the career data, query many times."""

    def __init__(self, data: CareerData, config: CareerConfig | None = None) -> None:
        self.data = data
        self.config = config or CareerConfig()
        self.stat_weights = feature_weights(
            data.categories, self.config.category_weights, data.in_distance
        )
        self._moment_cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}

    def find_comps(self, player_id: str, through_age: int) -> CareerResult:
        """Comps for `player_id`'s career through `through_age`; KeyError/ValueError if the
        player is unknown or has no scored season at that age."""
        data, cfg = self.data, self.config
        if player_id not in data.players.index:
            raise KeyError(f"unknown player {player_id!r}")
        end_season = data.season_at_age(player_id, through_age)
        end_row = data.row(player_id, end_season)
        if end_row is None or data.status[end_row] not in SCORED:
            raise ValueError(
                f"{data.name(player_id)} has no season of at least the career minimum at age "
                f"{through_age}"
            )
        first_age = int(data.players.at[player_id, "first_age"])
        ages = window_ages(first_age, through_age, cfg.window)
        q, q_rel, q_rows = player_window(data, player_id, ages)

        d = len(data.stats)
        slots = slot_weights(len(ages), cfg.decay)
        w = np.repeat(slots, d) * np.tile(self.stat_weights, len(ages)) * q_rel

        q_exp = experience(data, player_id, end_season)
        result = CareerResult(
            player_id=player_id,
            name=data.name(player_id),
            through_age=through_age,
            ages=tuple(ages),
            query_seasons=tuple(int(data.seasons[r]) if r is not None else None for r in q_rows),
            query_experience=q_exp,
            comps=(),
            projection=project(data, [], [], cfg.horizon),
            stats=data.stats,
        )
        candidates = self._candidates(player_id, ages, through_age)
        if not candidates:
            return result
        windows = [player_window(data, c, ages) for c in candidates]
        X = np.array([v for v, _, _ in windows])
        R = np.array([r for _, r, _ in windows])
        mean, var = self._moments(tuple(ages))
        scores = masked_scores(X, q, w, mean, var, "euclidean")
        contributions = scores.contributions * R
        total = np.nansum(contributions, axis=1)
        keep = scores.shared >= cfg.min_shared

        order = [i for i in np.argsort(-np.nan_to_num(total, nan=-np.inf)) if keep[i]]
        if cfg.max_experience_gap is not None:
            order = [
                i
                for i in order
                if abs(self._experience_at(candidates[i], through_age) - q_exp)
                <= cfg.max_experience_gap
            ]
        ranked = [(i, float(max(total[i], 0.0))) for i in order]

        comps = [
            self._comp(
                player_id,
                candidates[i],
                score,
                scores.shared[i],
                ages,
                q,
                X[i],
                contributions[i],
                windows[i][2],
                through_age,
            )  # fmt: skip
            for i, score in ranked[: cfg.n]
        ]
        observable = []
        for i, score in ranked:
            if len(observable) >= cfg.n_project:
                break
            outcome = comp_outcome(data, candidates[i], through_age, cfg.horizon)
            if outcome.observable:
                observable.append((outcome, score))
        projection = project(
            data, [o for o, _ in observable], [s for _, s in observable], cfg.horizon
        )
        return dataclasses.replace(result, comps=tuple(comps), projection=projection)

    def _moments(self, ages: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        if ages not in self._moment_cache:
            self._moment_cache[ages] = age_moments(self.data, ages)
        return self._moment_cache[ages]

    def _experience_at(self, player_id: str, age: int) -> int:
        return experience(self.data, player_id, self.data.season_at_age(player_id, age))

    def _candidates(self, player_id: str, ages: list[int], through_age: int) -> list[str]:
        data = self.data
        out = []
        for pid in data.players.index:
            if pid == player_id or is_censored(data, pid, ages):
                continue
            row = data.row(pid, data.season_at_age(pid, through_age))
            if row is not None and data.status[row] in SCORED:
                out.append(str(pid))
        return out

    def _comp(
        self,
        query_id: str,
        player_id: str,
        score: float,
        shared: float,
        ages: list[int],
        q: np.ndarray,
        x: np.ndarray,
        contributions: np.ndarray,
        rows: list[int | None],
        through_age: int,
    ) -> CareerComp:
        data, cfg = self.data, self.config
        d = len(data.stats)
        slots = slot_weights(len(ages), cfg.decay)
        by_slot = {
            (age, data.stats[j]): float(np.nan_to_num(contributions[i * d + j]))
            for i, age in enumerate(ages)
            for j in range(d)
            if self.stat_weights[j] > 0
        }
        traits = []
        for j, stat in enumerate(data.stats):
            both = [i for i in range(len(ages)) if not np.isnan(q[i * d + j] + x[i * d + j])]
            if not both:
                continue
            gaps = np.array([abs(q[i * d + j] - x[i * d + j]) for i in both])
            gap = float(np.average(gaps, weights=slots[both]))
            latest = both[-1]
            q_row = data.row(query_id, data.season_at_age(query_id, ages[latest]))
            traits.append(
                CareerTrait(
                    stat=stat,
                    label=data.labels[j],
                    points=float(sum(by_slot.get((ages[i], stat), 0.0) for i in both)),
                    gap=gap,
                    ages=tuple(ages[i] for i in both),
                    query_raw=float(data.raw[q_row, j]) if q_row is not None else np.nan,
                    comp_raw=float(data.raw[rows[latest], j]),
                )
            )
        # alike: within close_z on average (display-only stats have 0 points and sort last)
        alike = sorted((t for t in traits if t.gap <= cfg.close_z), key=lambda t: -t.points)
        differ = sorted((t for t in traits if t.gap > cfg.close_z), key=lambda t: -t.gap)
        end = data.season_at_age(player_id, through_age)
        return CareerComp(
            player_id=player_id,
            name=data.name(player_id),
            score=score,
            shared=float(shared),
            experience=experience(data, player_id, end),
            window_seasons=tuple(int(data.seasons[r]) if r is not None else None for r in rows),
            similar_traits=tuple(alike[: cfg.top_traits]),
            differences=tuple(differ[: cfg.top_traits]),
            contributions=by_slot,
            outcome=comp_outcome(data, player_id, through_age, cfg.horizon),
        )
