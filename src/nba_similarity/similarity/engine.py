"""Finds the most similar player-seasons to a query, with per-feature explanations."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nba_similarity.similarity.config import SimilarityConfig, feature_weights
from nba_similarity.similarity.data import FeatureMatrix, load_feature_matrix
from nba_similarity.similarity.scoring import Scores, masked_scores, projected_scores


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    label: str
    category: str
    points: float  # contribution to the match's score (all features sum to the score)
    query_z: float
    match_z: float
    query_raw: float
    match_raw: float


@dataclass(frozen=True)
class Match:
    player_id: str
    name: str
    season: int
    score: float  # 0-100: 100 = identical, 0 = no closer than a typical player-season
    shared: float  # share of the total feature weight both seasons have
    top_features: tuple[FeatureContribution, ...]  # most-contributing first


@dataclass(frozen=True)
class _PcaSpace:
    rows: np.ndarray  # matrix rows complete on the subspace's features
    features: np.ndarray  # feature indices in the subspace
    sqrt_w: np.ndarray
    projection: np.ndarray
    cov: np.ndarray
    mean: np.ndarray
    n_components: int


class SimilarityEngine:
    """Ranks player-seasons by similarity to a query player-season.

    Build it once (it standardizes the matrix and caches PCA fits) and query it many times.
    """

    def __init__(self, matrix: FeatureMatrix, config: SimilarityConfig | None = None) -> None:
        self.matrix = matrix
        self.config = config or SimilarityConfig()
        self.weights = feature_weights(matrix.categories, self.config.category_weights)
        if self.config.standardize:
            self._x = StandardScaler().fit_transform(matrix.z)  # NaNs are ignored and kept
        else:
            self._x = matrix.z.copy()
        self._mean = np.nanmean(self._x, axis=0)
        self._var = np.nanvar(self._x, axis=0)
        self._pca_cache: dict[tuple[int, ...], _PcaSpace] = {}

    @classmethod
    def from_db(
        cls, conn: sqlite3.Connection, config: SimilarityConfig | None = None
    ) -> SimilarityEngine:
        return cls(load_feature_matrix(conn), config)

    def with_config(self, **changes: object) -> SimilarityEngine:
        """A new engine over the same matrix with some settings changed."""
        return SimilarityEngine(self.matrix, replace(self.config, **changes))  # type: ignore[arg-type]

    def find_similar(self, player_id: str, season: int, n: int = 10) -> list[Match]:
        """Top-`n` most similar player-seasons to `player_id`'s `season`, best first."""
        query = self.matrix.row(player_id, season)
        rows, scores = self._score(query)

        keep = ~np.isnan(scores.score) & (rows != query)
        if not self.config.include_same_player:
            keep &= self.matrix.player_ids[rows] != player_id
        if self.config.pca_components is None:
            keep &= scores.shared >= self.config.min_shared - 1e-9

        candidates = np.flatnonzero(keep)
        order = candidates[np.argsort(-scores.score[candidates], kind="stable")][:n]
        return [self._match(query, rows[i], scores, i) for i in order]

    def _score(self, query: int) -> tuple[np.ndarray, Scores]:
        """Scores for every comparable row; returns (row indices, scores aligned to them)."""
        q = self._x[query]
        if self.config.pca_components is None:
            return np.arange(len(self._x)), masked_scores(
                self._x, q, self.weights, self._mean, self._var, self.config.metric
            )

        available = np.flatnonzero(~np.isnan(q) & (self.weights > 0))
        space = self._pca_space(tuple(available))
        if len(space.rows) == 0 or query not in set(space.rows.tolist()):
            return np.array([], dtype=int), _empty_scores(len(self.weights))
        xw = self._x[np.ix_(space.rows, space.features)] * space.sqrt_w
        qw = q[space.features] * space.sqrt_w
        sub = projected_scores(
            xw,
            qw,
            space.projection,
            space.cov,
            space.mean,
            self.config.metric,
            shared=float(self.weights[space.features].sum() / self.weights.sum()),
        )
        full = np.zeros((len(space.rows), len(self.weights)))
        full[:, space.features] = sub.contributions
        return space.rows, Scores(sub.score, full, sub.shared)

    def _pca_space(self, features: tuple[int, ...]) -> _PcaSpace:
        if features not in self._pca_cache:
            idx = np.array(features, dtype=int)
            rows = np.flatnonzero(~np.isnan(self._x[:, idx]).any(axis=1))
            sqrt_w = np.sqrt(self.weights[idx])
            xw = self._x[np.ix_(rows, idx)] * sqrt_w
            n_components = self.config.pca_components
            if isinstance(n_components, int):
                n_components = min(n_components, len(idx), len(rows))
            pca = PCA(n_components=n_components).fit(xw)
            components = pca.components_
            self._pca_cache[features] = _PcaSpace(
                rows=rows,
                features=idx,
                sqrt_w=sqrt_w,
                projection=components.T @ components,
                cov=np.cov(xw, rowvar=False, ddof=0),
                mean=xw.mean(axis=0),
                n_components=len(components),
            )
        return self._pca_cache[features]

    def _match(self, query: int, row: int, scores: Scores, i: int) -> Match:
        m = self.matrix
        points = scores.contributions[i]
        ranked = [
            j
            for j in np.argsort(-np.nan_to_num(points, nan=-np.inf), kind="stable")
            if self.weights[j] > 0 and not np.isnan(points[j])
            and not np.isnan(m.z[query, j]) and not np.isnan(m.z[row, j])
        ]  # fmt: skip
        top = tuple(
            FeatureContribution(
                feature=m.features[j],
                label=m.labels[j],
                category=m.categories[j],
                points=float(points[j]),
                query_z=float(m.z[query, j]),
                match_z=float(m.z[row, j]),
                query_raw=float(m.raw[query, j]),
                match_raw=float(m.raw[row, j]),
            )
            for j in ranked[: self.config.top_features]
        )
        return Match(
            player_id=str(m.player_ids[row]),
            name=str(m.names[row]),
            season=int(m.seasons[row]),
            score=float(np.clip(scores.score[i], 0.0, 100.0)),
            shared=float(scores.shared[i]),
            top_features=top,
        )


def _empty_scores(d: int) -> Scores:
    return Scores(np.array([]), np.empty((0, d)), np.array([]))
