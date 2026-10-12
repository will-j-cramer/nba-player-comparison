"""Similarity settings and the per-feature weights derived from category weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Metric = Literal["euclidean", "cosine"]


@dataclass(frozen=True)
class SimilarityConfig:
    """How player-seasons are compared.

    - metric: "euclidean" (closeness of the stat lines) or "cosine" (same statistical shape,
      regardless of magnitude: a role player can match a star with the same profile).
    - category_weights: relative weight per category; categories not listed get 1.0, and 0
      drops a category. Each category's weight is split evenly over its features, so a
      category counts the same however many stats it has.
    - standardize: re-standardize each feature over the whole matrix (sklearn StandardScaler)
      before weighting. The stored z-scores are already per-season standardized, so this
      mostly corrects for variance lost to clipping.
    - pca_components: None (no PCA), an int (number of components) or a float in (0, 1)
      (share of variance to keep). With PCA, a query is only compared with seasons that
      have every feature the query has, since PCA cannot handle missing values.
    - min_shared: without PCA, candidates must share at least this share of the total
      feature weight with the query (pre-1980 seasons lack 3PA rate, USG%, TOV%). Stats a
      candidate lacks also count as "typical" in its score (see scoring.py).
    - include_same_player: also return the query player's other seasons.
    - top_features: how many similar traits (and differences) to return per match.
    - close_z: two seasons count as "alike" on a stat when their z-scores are within this
      many SD; further apart, the stat is listed as a difference instead.
    """

    metric: Metric = "euclidean"
    category_weights: Mapping[str, float] = field(default_factory=dict)
    standardize: bool = True
    pca_components: int | float | None = None
    min_shared: float = 0.75
    include_same_player: bool = False
    top_features: int = 3
    close_z: float = 1.0

    def __post_init__(self) -> None:
        if self.metric not in ("euclidean", "cosine"):
            raise ValueError(f"unknown metric {self.metric!r}")
        if not 0 < self.min_shared <= 1:
            raise ValueError("min_shared must be in (0, 1]")
        pca = self.pca_components
        if pca is not None and not (
            (isinstance(pca, int) and pca >= 1) or (isinstance(pca, float) and 0 < pca < 1)
        ):
            raise ValueError("pca_components must be None, an int >= 1, or a float in (0, 1)")
        if self.top_features < 0:
            raise ValueError("top_features must be >= 0")
        if self.close_z <= 0:
            raise ValueError("close_z must be > 0")


def feature_weights(categories: Sequence[str], category_weights: Mapping[str, float]) -> np.ndarray:
    """Per-feature weights summing to 1, with each category's weight split over its features.

    `categories[i]` is the category of feature i.
    """
    known = set(categories)
    unknown = set(category_weights) - known
    if unknown:
        raise ValueError(f"unknown categories {sorted(unknown)}; expected {sorted(known)}")
    if any(w < 0 for w in category_weights.values()):
        raise ValueError("category weights must be >= 0")

    counts = {c: categories.count(c) for c in known}
    raw = np.array([category_weights.get(c, 1.0) / counts[c] for c in categories], dtype=float)
    if raw.sum() == 0:
        raise ValueError("at least one category needs a positive weight")
    return raw / raw.sum()
