"""Similarity scores (0-100 scale) that decompose exactly into per-feature contributions.

Euclidean: score = 100 * (1 - D² / E[D²]), where D² is the weighted squared distance between
the query and a candidate, and E[D²] is the expected squared distance between the query and
a random player-season: per feature, (q_j - mean_j)² + var_j. So 100 means identical and 0
means "no closer than a typical player-season" (negative scores are clipped by the engine).
Because the baseline is query-specific, a player who is extreme in some stat is not
penalised for being far from everyone, and matching a query's *distinctive* traits (where
the baseline is large) earns more than matching unremarkable ones. Feature j contributes
100 * (b_j - d_j) / Σb, where d_j is its share of D² and b_j its share of E[D²]; the
contributions sum to the score.

Cosine: score = 100 * cos(angle between the weighted vectors). Feature j contributes
100 * a_j * b_j / (|a| |b|), and the contributions sum to the score. Cosine compares the
shape of a stat profile rather than its size.

`masked_scores` works on the weighted feature space directly and tolerates missing values:
each pair is compared on the features both have, and D² and its baseline cover only those,
so the score is an average over shared features rather than a sum. `projected_scores` works
in a PCA subspace (complete data only), using the projection P = VᵀV so that contributions
still map back onto the original features: D² = Σ_j Δ_j (PΔ)_j.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nba_similarity.similarity.config import Metric


@dataclass(frozen=True)
class Scores:
    score: np.ndarray  # (n,) unclipped, NaN where no comparison is possible
    contributions: np.ndarray  # (n, d) points per feature; rows sum to `score`
    shared: np.ndarray  # (n,) share of total feature weight both seasons have


def masked_scores(
    X: np.ndarray,
    q: np.ndarray,
    w: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    metric: Metric,
) -> Scores:
    """Compare query `q` (d,) with each row of `X` (n, d) on their shared, non-NaN features.

    `w` are feature weights (summing to 1); `mean` and `var` are the per-feature mean and
    variance of `X`, which set the euclidean baseline (expected distance from `q`).
    """
    mask = ~np.isnan(X) & ~np.isnan(q)
    shared = (mask * w).sum(axis=1) / w.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "euclidean":
            diff = np.where(mask, X - q, 0.0)
            distance = w * diff**2
            baseline = np.where(mask, w * ((q - mean) ** 2 + var), 0.0)
            r2 = baseline.sum(axis=1, keepdims=True)
            contributions = 100 * (baseline - distance) / r2
        else:
            a = np.where(mask, q, 0.0)
            b = np.where(mask, X, 0.0)
            norms = np.sqrt((w * a**2).sum(axis=1) * (w * b**2).sum(axis=1))[:, None]
            contributions = 100 * w * a * b / norms
    contributions = np.where(np.isfinite(contributions), contributions, np.nan)
    return Scores(_row_sums(contributions), contributions, shared)


def projected_scores(
    Xw: np.ndarray,
    qw: np.ndarray,
    projection: np.ndarray,
    cov: np.ndarray,
    mean: np.ndarray,
    metric: Metric,
    shared: float = 1.0,
) -> Scores:
    """Compare weighted query `qw` (k,) with weighted rows `Xw` (n, k) inside a PCA subspace.

    `projection` is P = VᵀV for the retained components V; `cov` and `mean` are the
    covariance and column means of the weighted data. They set the euclidean baseline
    (expected projected distance from `qw`), and cosine is measured from the data's centre,
    as PCA does. All inputs are complete (no NaN), so
    every row shares the same features; `shared` is their share of the total weight.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        if metric == "euclidean":
            delta = Xw - qw
            distance = delta * (delta @ projection)
            offset = qw - mean
            baseline = offset * (projection @ offset) + np.diag(projection @ cov)
            contributions = 100 * (baseline - distance) / baseline.sum()
        else:
            a = qw - mean
            b = Xw - mean
            pb = b @ projection
            norms = np.sqrt(a @ projection @ a) * np.sqrt((b * pb).sum(axis=1))
            contributions = 100 * a * pb / norms[:, None]
    contributions = np.where(np.isfinite(contributions), contributions, np.nan)
    return Scores(_row_sums(contributions), contributions, np.full(len(Xw), shared))


def _row_sums(contributions: np.ndarray) -> np.ndarray:
    """Row sums, NaN for rows where every entry is NaN (nothing could be compared)."""
    all_nan = np.isnan(contributions).all(axis=1)
    sums = np.nansum(contributions, axis=1)
    return np.where(all_nan, np.nan, sums)
