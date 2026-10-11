from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_similarity.features.definitions import Z_CLIP
from nba_similarity.features.normalize import add_zscores, season_stats

FEATS = ("x", "y")


def frame(season: int, x: list[float | None], y: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame({"season": season, "x": x, "y": y})


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = season_stats(df, FEATS)
    return stats, add_zscores(df, stats, FEATS)


def test_zscores_are_standardized_within_each_season() -> None:
    df = pd.concat(
        [
            frame(1990, [1, 2, 3, 4, 5], [10, 20, 30, 40, 50]),
            frame(2020, [10, 30, 50, 70, 90], [1, 1, 2, 2, 3]),
        ]
    )
    _, out = build(df)
    for _, group in out.groupby("season"):
        for col in ("z_x", "z_y"):
            assert group[col].mean() == pytest.approx(0, abs=1e-12)
            assert group[col].std(ddof=0) == pytest.approx(1)
    # same relative position in different eras -> same z-score
    assert out.loc[out["x"] == 5, "z_x"].item() == pytest.approx(
        out.loc[out["x"] == 90, "z_x"].item()
    )


def test_extreme_values_are_clipped() -> None:
    df = frame(2020, [0.0] * 99 + [100.0], [float(i) for i in range(100)])
    _, out = build(df)
    assert out["z_x"].max() == Z_CLIP


def test_low_coverage_season_gets_null_z_but_keeps_raw() -> None:
    # 1 of 10 players has y (like 1973-74 turnovers tracked only for Seattle)
    df = frame(1974, list(range(10)), [5.0] + [None] * 9)
    stats, out = build(df)
    row = stats[(stats["season"] == 1974) & (stats["feature"] == "y")].iloc[0]
    assert row["coverage"] == pytest.approx(0.1) and not row["covered"]
    assert out["z_y"].isna().all()
    assert out["y"].iloc[0] == 5.0


def test_feature_absent_for_whole_season_is_uncovered() -> None:
    _, out = build(frame(1975, [1, 2, 3], [None, None, None]))
    assert out["z_y"].isna().all()
    assert out["n_features"].tolist() == [1, 1, 1]


def test_zero_spread_is_uncovered() -> None:
    stats, out = build(frame(2020, [1, 2, 3], [7, 7, 7]))
    assert not stats.loc[stats["feature"] == "y", "covered"].item()
    assert out["z_y"].isna().all()


def test_individual_missing_value_stays_missing() -> None:
    df = frame(2020, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 6, 7, 8, 9, None])
    stats, out = build(df)
    assert stats.loc[stats["feature"] == "y", "covered"].item()  # 90% coverage is enough
    assert np.isnan(out["z_y"].iloc[-1])
    assert out["n_features"].tolist() == [2] * 9 + [1]
