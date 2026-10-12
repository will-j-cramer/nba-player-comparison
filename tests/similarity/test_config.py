from __future__ import annotations

import numpy as np
import pytest

from nba_similarity.similarity.config import SimilarityConfig, feature_weights
from tests.similarity.conftest import CATEGORIES


def category_totals(weights: np.ndarray) -> dict[str, float]:
    totals: dict[str, float] = {}
    for category, w in zip(CATEGORIES, weights, strict=True):
        totals[category] = totals.get(category, 0.0) + w
    return totals


def test_equal_weights_give_each_category_the_same_total() -> None:
    w = feature_weights(CATEGORIES, {})
    assert w.sum() == pytest.approx(1)
    for total in category_totals(w).values():
        assert total == pytest.approx(0.2)  # 4 scoring stats count no more than 2 defense stats


def test_category_weight_scales_its_share() -> None:
    totals = category_totals(feature_weights(CATEGORIES, {"scoring": 2}))
    assert totals["scoring"] == pytest.approx(2 / 6)
    assert totals["defense"] == pytest.approx(1 / 6)


def test_zero_weight_drops_a_category() -> None:
    totals = category_totals(feature_weights(CATEGORIES, {"defense": 0}))
    assert totals["defense"] == 0 and totals["scoring"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("weights", "message"),
    [({"shooting": 1}, "unknown categories"), ({"scoring": -1}, ">= 0"),
     (dict.fromkeys(set(CATEGORIES), 0), "positive weight")],
)  # fmt: skip
def test_invalid_weights(weights: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        feature_weights(CATEGORIES, weights)


@pytest.mark.parametrize(
    "kwargs",
    [{"metric": "manhattan"}, {"min_shared": 0}, {"pca_components": 0},
     {"pca_components": 1.5}, {"top_features": -1}, {"close_z": 0}],
)  # fmt: skip
def test_invalid_config(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SimilarityConfig(**kwargs)  # type: ignore[arg-type]


def test_valid_pca_settings() -> None:
    assert SimilarityConfig(pca_components=5).pca_components == 5
    assert SimilarityConfig(pca_components=0.9).pca_components == 0.9
