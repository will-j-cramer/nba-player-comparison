from __future__ import annotations

import numpy as np

from nba_similarity.features.definitions import FEATURES
from nba_similarity.similarity.data import FeatureMatrix

NAMES = tuple(f.name for f in FEATURES)
CATEGORIES = tuple(f.category for f in FEATURES)


def col(feature: str) -> int:
    return NAMES.index(feature)


def make_matrix(
    z: np.ndarray, player_ids: list[str] | None = None, seasons: list[int] | None = None
) -> FeatureMatrix:
    """A FeatureMatrix over the real 12 features; raw values mirror the z-scores."""
    n = len(z)
    player_ids = player_ids or [f"p{i}" for i in range(n)]
    return FeatureMatrix(
        player_ids=np.array(player_ids),
        names=np.array([f"Player {pid}" for pid in player_ids]),
        seasons=np.array(seasons or [2020] * n),
        features=NAMES,
        categories=CATEGORIES,
        labels=tuple(f.label for f in FEATURES),
        z=np.asarray(z, dtype=float),
        raw=np.asarray(z, dtype=float) * 10,
        in_distance=tuple(f.in_distance for f in FEATURES),  # points per 100 is display-only
    )


def random_population(n: int = 300, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, len(NAMES)))
