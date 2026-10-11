"""Sanity report over the stored feature tables (`nba-features` prints it after a build)."""

from __future__ import annotations

import sqlite3
import sys
from typing import TextIO

import pandas as pd

from nba_similarity.features.definitions import FEATURES, MIN_MINUTES, Z_CLIP, z_column

TOP_N = 5


def season_label(season: int) -> str:
    return f"{season - 1}-{str(season)[2:]}"


def print_feature_summary(conn: sqlite3.Connection, out: TextIO = sys.stdout) -> None:
    """Print overview, per-feature coverage, normalization check, era coverage and leaders."""
    rows = pd.read_sql_query(
        "SELECT f.*, pl.name FROM player_season_features f JOIN players pl USING (player_id)",
        conn,
    )
    stats = pd.read_sql_query("SELECT * FROM feature_season_stats", conn)
    (n_all,) = conn.execute("SELECT COUNT(*) FROM player_seasons").fetchone()

    def emit(line: str = "") -> None:
        print(line, file=out)

    seasons = sorted(rows["season"].unique())
    emit("== Overview ==")
    emit(
        f"{len(rows)} qualified player-seasons of {n_all} ({len(rows) / max(n_all, 1):.0%}), "
        f"{len(seasons)} seasons ({season_label(seasons[0])} to {season_label(seasons[-1])})"
        if seasons
        else "no qualified player-seasons"
    )
    if not seasons:
        return
    thresholds = rows.groupby("season")["min_minutes"].first()
    emit(f"minimum minutes: {MIN_MINUTES} (full season)")
    for season, minutes in thresholds[thresholds != MIN_MINUTES].items():
        emit(f"  {season_label(int(season))}: {int(minutes)} (shortened or in-progress season)")

    first, last = seasons[0], seasons[-1]
    emit("\n== Features ==")
    emit(
        f"{'feature':<10} {'category':<11} {'covered seasons':<22} {'z values':>8} "
        f"{'clipped':>7}  median {season_label(first)} / {season_label(last)}"
    )
    for f in FEATURES:
        s = stats[(stats["feature"] == f.name) & (stats["covered"] == 1)]
        covered = f"{len(s)}/{len(seasons)}" + (
            f" from {season_label(int(s['season'].min()))}" if len(s) else ""
        )
        z = rows[z_column(f.name)]
        clipped = int((z.abs() >= Z_CLIP - 1e-9).sum())
        med_first = _median(rows, f.name, first)
        med_last = _median(rows, f.name, last)
        emit(
            f"{f.name:<10} {f.category:<11} {covered:<22} {int(z.notna().sum()):>8} "
            f"{clipped:>7}  {med_first} / {med_last}"
        )

    emit("\n== Normalization check (per season, covered features; expect mean ~0, sd ~1) ==")
    emit(f"{'feature':<10} {'max |mean z|':>12} {'sd range':>14}")
    for f in FEATURES:
        by_season = rows.groupby("season")[z_column(f.name)].agg(["mean", lambda v: v.std(ddof=0)])
        by_season = by_season.dropna()
        if by_season.empty:
            emit(f"{f.name:<10} {'-':>12} {'-':>14}")
            continue
        sd = by_season.iloc[:, 1]
        emit(
            f"{f.name:<10} {by_season['mean'].abs().max():>12.3f} {sd.min():>6.3f} - {sd.max():.3f}"
        )

    emit("\n== Share of rows with each z-score, by decade ==")
    decade = (rows["season"] - 1) // 10 * 10  # 1974 (1973-74) -> 1970s
    coverage = rows[[z_column(f.name) for f in FEATURES]].notna().groupby(decade).mean()
    coverage.columns = [f.name for f in FEATURES]
    coverage.index = [f"{d}s" for d in coverage.index]
    emit(coverage.map(lambda v: f"{v:.0%}").to_string())

    emit(f"\n== Top {TOP_N} player-seasons per feature (by z-score) ==")
    for f in FEATURES:
        col = z_column(f.name)
        top = rows.dropna(subset=[col]).nlargest(TOP_N, col)
        emit(f"{f.label} ({f.name}):")
        for r in top.itertuples():
            emit(
                f"  {r.name:<26} {season_label(int(r.season))}  "
                f"{getattr(r, f.name):>7.3f}  z={getattr(r, col):+.2f}"
            )

    emit("\n== Features available per row (of 12) ==")
    counts = rows["n_features"].value_counts().sort_index(ascending=False)
    for n, count in counts.items():
        emit(f"  {int(n):>2}: {int(count)} rows ({count / len(rows):.1%})")


def _median(rows: pd.DataFrame, feature: str, season: int) -> str:
    values = rows.loc[rows["season"] == season, feature].dropna()
    return f"{values.median():.3f}" if len(values) else "n/a"
