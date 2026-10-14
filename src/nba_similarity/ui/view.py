"""View models for the Streamlit app: pure functions from engine objects to tables/series.

Nothing here imports streamlit, so all of it is unit-testable; the page scripts only lay
things out.
"""

from __future__ import annotations

import math
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_similarity.careers.data import SCORED, CareerData
from nba_similarity.careers.engine import CareerResult
from nba_similarity.careers.format import format_outcome, format_trait
from nba_similarity.similarity.data import FeatureMatrix
from nba_similarity.similarity.engine import Match
from nba_similarity.similarity.explain import (
    SHORT_LABELS,
    format_contribution,
    format_value,
    season_label,
)

LeagueAverages = dict[tuple[int, str], float]


@dataclass(frozen=True)
class PlayerOption:
    """One entry in the player dropdown."""

    player_id: str
    name: str
    label: str  # shown in the dropdown, e.g. "Nikola Jokić (2016–2026)"
    seasons: tuple[int, ...]  # qualified seasons, newest first


def player_options(matrix: FeatureMatrix) -> list[PlayerOption]:
    """One option per player, sorted by name; namesakes get their B-Ref id in the label."""
    seasons: dict[str, list[int]] = {}
    names: dict[str, str] = {}
    for pid, name, season in zip(matrix.player_ids, matrix.names, matrix.seasons, strict=True):
        seasons.setdefault(str(pid), []).append(int(season))
        names[str(pid)] = str(name)
    name_counts = Counter(names.values())
    options = []
    for pid, name in names.items():
        years = sorted(seasons[pid], reverse=True)
        span = f"{years[-1]}–{years[0]}" if len(years) > 1 else f"{years[0]}"
        suffix = f", {pid}" if name_counts[name] > 1 else ""
        options.append(PlayerOption(pid, name, f"{name} ({span}{suffix})", tuple(years)))
    return sorted(options, key=lambda o: (o.name.lower(), o.seasons[-1], o.player_id))


def short_label(name: str, season: int) -> str:
    """Compact column header: "Nikola Jokić", 2023 -> "Jokić '23"."""
    return f"{name.split()[-1]} '{str(season)[2:]}"


def query_profile(matrix: FeatureMatrix, row: int) -> str:
    """One line describing the query: its standout stats by z-score."""
    parts = []
    order = np.argsort(-np.abs(np.nan_to_num(matrix.z[row])))
    for j in order[:4]:
        z = matrix.z[row, j]
        if np.isnan(z) or abs(z) < 1:
            continue
        level = "elite" if abs(z) >= 2 else "high" if z > 0 else "low"
        if z <= -2:
            level = "very low"
        parts.append(f"{level} {matrix.labels[j]} ({matrix.raw[row, j]:.3g}, {z:+.1f} SD)")
    return "; ".join(parts) if parts else "no stat more than 1 SD from league average"


def results_frame(matches: Sequence[Match]) -> pd.DataFrame:
    """The results table: one row per comp, best first."""
    return pd.DataFrame(
        {
            "Rank": range(1, len(matches) + 1),
            "Player": [m.name for m in matches],
            "Season": [season_label(m.season) for m in matches],
            "Score": [round(m.score, 1) for m in matches],
            "Stats shared": ["" if m.shared >= 0.999 else f"{m.shared:.0%}" for m in matches],
            "Alike": ["; ".join(format_contribution(c) for c in m.similar_traits) for m in matches],
            "Differ": ["; ".join(format_contribution(c) for c in m.differences) for m in matches],
        }
    )


def comparison_frame(
    matrix: FeatureMatrix,
    query_row: int,
    match: Match,
    league: LeagueAverages,
    close_z: float = 1.0,
) -> pd.DataFrame:
    """Side-by-side stat table for the query and one match, with league context.

    Rows follow the feature order (grouped by category). Verdict uses the engine's rule for
    every stat: "Alike" within `close_z` SD, "Differ" beyond it, "n/a" when a season lacks
    the stat, and "context" for display-only stats that don't count in the score. The radar
    shows the matching within-season percentiles.
    """
    match_row = matrix.row(match.player_id, match.season)
    q_season = int(matrix.seasons[query_row])
    rows = []
    for j, feature in enumerate(matrix.features):
        qz, mz = matrix.z[query_row, j], matrix.z[match_row, j]
        if not matrix.in_distance[j]:
            verdict = "context"
        elif np.isnan(qz) or np.isnan(mz):
            verdict = "n/a"
        else:
            verdict = "Alike" if abs(qz - mz) <= close_z else "Differ"
        rows.append(
            {
                "Stat": matrix.labels[j],
                "Query": _num(matrix.raw[query_row, j]),
                "Match": _num(matrix.raw[match_row, j]),
                "Lg avg (query yr)": league.get((q_season, feature)),
                "Lg avg (match yr)": league.get((match.season, feature)),
                "Verdict": verdict,
                "Points": match.contributions.get(feature),
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class RadarData:
    """The radar chart's axes and both profiles."""

    labels: tuple[str, ...]
    query_pct: np.ndarray  # within-season percentile, 0-100, NaN = missing
    match_pct: np.ndarray
    query_raw: np.ndarray
    match_raw: np.ndarray


def percentile(z: np.ndarray) -> np.ndarray:
    """Within-season percentile implied by a z-score (normal approximation), NaN kept."""
    return np.array([np.nan if np.isnan(v) else 50 * (1 + math.erf(v / math.sqrt(2))) for v in z])


def radar_data(matrix: FeatureMatrix, query_row: int, match_row: int) -> RadarData:
    """Radar axes: the stats that drive the distance, as within-season percentiles.

    Percentiles put every stat on the same 0-100 scale and compare across eras the way the
    engine does (relative to each season's league).
    """
    cols = [j for j in range(len(matrix.features)) if matrix.in_distance[j]]
    return RadarData(
        labels=tuple(SHORT_LABELS.get(matrix.features[j], matrix.labels[j]) for j in cols),
        query_pct=percentile(matrix.z[query_row, cols]),
        match_pct=percentile(matrix.z[match_row, cols]),
        query_raw=matrix.raw[query_row, cols],
        match_raw=matrix.raw[match_row, cols],
    )


def load_league_averages(conn: sqlite3.Connection) -> LeagueAverages:
    """Observed league average per (season, feature), from feature_season_stats."""
    rows = conn.execute("SELECT season, feature, raw_mean FROM feature_season_stats").fetchall()
    return {(int(s), str(f)): float(v) for s, f, v in rows if v is not None}


def _num(value: float) -> float | None:
    return None if np.isnan(value) else float(value)


# ---------------------------------------------------------------- map page

# Stats shown when hovering a point on the map (observed values).
MAP_HOVER_FEATURES = ("pts", "usg_pct", "ts_pct", "ast_pct", "drb_pct", "fg3a")


@dataclass(frozen=True)
class MapLayout:
    """The stored map, aligned to the feature matrix."""

    xy: np.ndarray  # (n, 2) aligned to the feature matrix rows; NaN = no position on the map
    built_at: str
    stale: bool  # the features changed since `nba-map` ran


def align_map(
    matrix: FeatureMatrix, player_ids: np.ndarray, seasons: np.ndarray, xy: np.ndarray
) -> np.ndarray:
    """Stored map coordinates in feature-matrix row order (NaN for seasons not on the map)."""
    position = {
        (str(p), int(s)): i for i, (p, s) in enumerate(zip(player_ids, seasons, strict=True))
    }
    out = np.full((len(matrix.player_ids), 2), np.nan)
    for row, key in enumerate(
        zip(matrix.player_ids.astype(str), matrix.seasons.astype(int), strict=True)
    ):
        i = position.get(key)
        if i is not None:
            out[row] = xy[i]
    return out


def map_frame(
    matrix: FeatureMatrix,
    xy: np.ndarray,
    *,
    query_row: int | None = None,
    comp_rows: Sequence[int] = (),
    color_feature: str | None = None,
    clusters: np.ndarray | None = None,
    cluster_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """One row per player-season on the map, with its highlight role and hover text.

    role is "query", "comp" (rank = position in the results) or "other". color_pct is the
    within-season percentile of `color_feature` (NaN when the season lacks it); cluster is
    -1 when no clustering is given. Seasons without a map position are left out.
    """
    n = len(matrix.player_ids)
    role = np.full(n, "other", dtype=object)
    rank = np.zeros(n, dtype=int)
    for i, r in enumerate(comp_rows, start=1):
        role[r], rank[r] = "comp", i
    if query_row is not None:
        role[query_row] = "query"
    clusters = np.full(n, -1) if clusters is None else np.asarray(clusters)
    names = cluster_names or {}

    color_pct = np.full(n, np.nan)
    color_line = [""] * n
    if color_feature is not None:
        j = matrix.features.index(color_feature)
        color_pct = percentile(matrix.z[:, j])
        label = SHORT_LABELS.get(color_feature, matrix.labels[j])
        color_line = [
            f"{label}: no data" if np.isnan(p) else f"{label}: {r:.3g} ({p:.0f}th pct of season)"
            for r, p in zip(matrix.raw[:, j], color_pct, strict=True)
        ]

    hover_cols = [matrix.features.index(f) for f in MAP_HOVER_FEATURES if f in matrix.features]
    hover = []
    for row in range(n):
        stats = [
            f"{SHORT_LABELS.get(matrix.features[j], matrix.labels[j])} "
            f"{format_value(matrix.raw[row, j])}"
            for j in hover_cols
        ]
        lines = [
            f"<b>{matrix.names[row]}</b> {season_label(int(matrix.seasons[row]))}",
            " · ".join(stats[:3]),
            " · ".join(stats[3:]),
        ]
        if color_line[row]:
            lines.append(color_line[row])
        if clusters[row] >= 0:
            lines.append(f"Cluster {clusters[row] + 1}: {names.get(int(clusters[row]), '')}")
        hover.append("<br>".join(line for line in lines if line))

    frame = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "label": [
                short_label(str(m), int(s))
                for m, s in zip(matrix.names, matrix.seasons, strict=True)
            ],
            "role": role,
            "rank": rank,
            "cluster": clusters,
            "cluster_name": [names.get(int(c), "") for c in clusters],
            "color_pct": color_pct,
            "hover": hover,
        }
    )
    return frame[np.isfinite(xy).all(axis=1)].reset_index(drop=True)


# ---------------------------------------------------------------- career page


def career_ages(data: CareerData, player_id: str) -> list[int]:
    """Ages with a season the career comps can use (qualified or partial), newest first."""
    rows = [data.row(player_id, s) for s in data.player_seasons(player_id)]
    return sorted(
        {int(data.ages[r]) for r in rows if r is not None and data.status[r] in SCORED},
        reverse=True,
    )


def career_comps_frame(result: CareerResult, horizon: int) -> pd.DataFrame:
    """The career comps table: one row per comp, best first."""
    first = result.through_age + 1
    return pd.DataFrame(
        {
            "Rank": range(1, len(result.comps) + 1),
            "Player": [c.name for c in result.comps],
            "Score": [round(c.score, 1) for c in result.comps],
            "Seasons in": [c.experience for c in result.comps],
            "Window": ["" if c.shared >= 0.999 else f"{c.shared:.0%}" for c in result.comps],
            f"Ages {first}-{first + horizon - 1}": [
                format_outcome(c.outcome, horizon, first) for c in result.comps
            ],
            "Alike": ["; ".join(format_trait(t) for t in c.similar_traits) for c in result.comps],
            "Differ": ["; ".join(format_trait(t) for t in c.differences) for c in result.comps],
        }
    )


def outcome_frame(result: CareerResult) -> pd.DataFrame:
    """Per future age: score-weighted % of the observable comps in each outcome class."""
    ages = result.projection.ages
    return pd.DataFrame(
        {
            "Age": [a.age for a in ages],
            "Full season": [round(100 * a.shares["qualified"]) for a in ages],
            "Under the minimum": [round(100 * a.shares["limited"]) for a in ages],
            "Not in the NBA": [round(100 * a.shares["out"]) for a in ages],
            "Comps with stats": [a.n_qualified for a in result.projection.ages],
        }
    )


def projection_frame(result: CareerResult, data: CareerData, stats: Sequence[str]) -> pd.DataFrame:
    """Projected stats per future age: median [25-75%] in the latest season's terms."""
    columns: dict[str, list[str]] = {"Age": []}
    for age in result.projection.ages:
        columns["Age"].append(str(age.age))
        for stat in stats:
            j = data.stats.index(stat)
            p25, p50, p75 = age.today[j]
            label = SHORT_LABELS.get(stat, stat)
            text = "–" if np.isnan(p50) else f"{p50:.3g} ({p25:.3g}–{p75:.3g})"
            columns.setdefault(label, []).append(text)
    return pd.DataFrame(columns)


@dataclass(frozen=True)
class TrajectorySeries:
    """One player's within-season percentile of one stat, by age (NaN = no full data)."""

    label: str
    ages: tuple[int, ...]
    pct: np.ndarray
    raw: np.ndarray


def trajectory_series(
    data: CareerData, player_id: str, label: str, ages: Sequence[int], stat: str
) -> TrajectorySeries:
    """Percentiles at `ages` for seasons the comps use (qualified or partial); NaN otherwise."""
    j = data.stats.index(stat)
    pct, raw = [], []
    for age in ages:
        row = data.row(player_id, data.season_at_age(player_id, age))
        ok = row is not None and data.status[row] in SCORED
        pct.append(percentile(np.array([data.z[row, j]]))[0] if ok else np.nan)
        raw.append(data.raw[row, j] if ok else np.nan)
    return TrajectorySeries(label, tuple(ages), np.array(pct), np.array(raw))


def projection_band(result: CareerResult, stat: str) -> TrajectorySeries:
    """The projection by future age: `pct` is (n, 3), the 25th/50th/75th percentile."""
    j = result.stats.index(stat)
    ages = tuple(a.age for a in result.projection.ages)
    return TrajectorySeries(
        "Projection",
        ages,
        np.array([a.pct[j] for a in result.projection.ages]).reshape(len(ages), 3),
        np.full(len(ages), np.nan),
    )
