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


def test_default_weights_are_equal_per_stat() -> None:
    w = feature_weights(CATEGORIES, {})
    np.testing.assert_allclose(w, np.full(12, 1 / 12))
    totals = category_totals(w)
    assert totals["scoring"] == pytest.approx(4 / 12)  # 4 stats
    assert totals["defense"] == pytest.approx(2 / 12)  # 2 stats


def test_category_weight_multiplies_its_stats() -> None:
    w = feature_weights(CATEGORIES, {"defense": 2})
    totals = category_totals(w)
    assert totals["defense"] == pytest.approx(4 / 14)
    assert totals["scoring"] == pytest.approx(4 / 14)
    # every stat in a category gets the multiplier, not a split of it
    assert w[CATEGORIES.index("defense")] == pytest.approx(2 * w[CATEGORIES.index("scoring")])


def test_zero_weight_drops_a_category() -> None:
    totals = category_totals(feature_weights(CATEGORIES, {"defense": 0}))
    assert totals["defense"] == 0 and totals["scoring"] == pytest.approx(0.4)


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
