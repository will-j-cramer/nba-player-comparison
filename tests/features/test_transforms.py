from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_similarity.features.definitions import FEATURE_NAMES, FEATURES, basis_column
from nba_similarity.features.transforms import add_basis, estimated_attempts, shrunk_ft_pct

SQRT_FEATURES = [f.name for f in FEATURES if f.transform == "sqrt"]


def frame(rows: list[dict[str, float | None]]) -> pd.DataFrame:
    """Rows with every feature column (default 1.0) plus helper columns."""
    base = dict.fromkeys(FEATURE_NAMES, 1.0) | {"season": 2023, "mp": 2400.0}
    return pd.DataFrame([base | r for r in rows])


def test_transform_assignments() -> None:
    assert sorted(SQRT_FEATURES) == ["blk_pct", "fg3a", "fta_rate", "orb_pct"]
    assert [f.name for f in FEATURES if f.transform == "shrink"] == ["ft_pct"]


def test_sqrt_basis_keeps_nulls_and_raw_values() -> None:
    df = frame(
        [
            {"blk_pct": 9.0, "ft_per100": 4.0, "fta_per100": 5.0},
            {"blk_pct": None, "ft_per100": 4.0, "fta_per100": 5.0},
        ]
    )
    out = add_basis(df)
    assert out[basis_column("blk_pct")].iloc[0] == pytest.approx(3.0)
    assert np.isnan(out[basis_column("blk_pct")].iloc[1])
    assert out["blk_pct"].iloc[0] == 9.0  # raw value untouched
    assert out[basis_column("pts")].tolist() == [1.0, 1.0]  # "none" copies the raw value


def test_estimated_attempts_from_per100_rate() -> None:
    # 6 FTA per 100 possessions over 2,400 minutes (50 games of 48) ~= 300 attempts
    assert estimated_attempts(pd.Series([6.0]), pd.Series([2400.0])).item() == pytest.approx(300)


def test_small_sample_ft_pct_is_shrunk_toward_league() -> None:
    # Two shooters: a high-volume .760 shooter sets the league average; a 10-attempt 1.000.
    # 10 attempts = 0.2 FTA/100 over 2,400 min; 2,000 attempts = 40 FTA/100.
    df = frame(
        [
            {"ft_pct": 0.76, "ft_per100": 30.4, "fta_per100": 40.0},
            {"ft_pct": 1.0, "ft_per100": 0.2, "fta_per100": 0.2},
        ]
    )
    shrunk = shrunk_ft_pct(df, df["ft_pct"])
    league = (30.4 + 0.2) / (40.0 + 0.2)  # attempt-weighted
    assert shrunk.iloc[1] == pytest.approx((10 * 1.0 + 50 * league) / 60)
    assert shrunk.iloc[1] == pytest.approx(0.80, abs=0.01)
    assert shrunk.iloc[0] == pytest.approx(0.76, abs=0.002)  # high volume barely moves


def test_no_attempts_stays_null() -> None:
    df = frame(
        [
            {"ft_pct": 0.8, "ft_per100": 4.0, "fta_per100": 5.0},
            {"ft_pct": None, "ft_per100": 0.0, "fta_per100": 0.0},
            {"ft_pct": None, "ft_per100": None, "fta_per100": None},
        ]
    )
    out = add_basis(df)
    assert out[basis_column("ft_pct")].isna().tolist() == [False, True, True]


def test_league_ft_pct_is_per_season() -> None:
    df = frame(
        [
            {"season": 1980, "ft_pct": 0.70, "ft_per100": 7.0, "fta_per100": 10.0},
            {"season": 2020, "ft_pct": 0.80, "ft_per100": 8.0, "fta_per100": 10.0},
        ]
    )
    shrunk = shrunk_ft_pct(df, df["ft_pct"])
    assert shrunk.tolist() == pytest.approx([0.70, 0.80])  # each equals its own league average
