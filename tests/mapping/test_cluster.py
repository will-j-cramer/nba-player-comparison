from __future__ import annotations

import numpy as np
import pytest

from nba_similarity.mapping.cluster import cluster_names, kmeans_labels


def test_kmeans_labels_numbered_by_size_and_deterministic() -> None:
    rng = np.random.default_rng(0)
    xy = np.vstack([rng.normal(0, 0.1, (50, 2)), rng.normal(5, 0.1, (20, 2))])
    labels = kmeans_labels(xy, 2)
    assert set(labels) == {0, 1}
    assert (labels[:50] == 0).all() and (labels[50:] == 1).all()  # 0 = largest
    assert (kmeans_labels(xy, 2) == labels).all()


def test_kmeans_rejects_bad_k() -> None:
    with pytest.raises(ValueError):
        kmeans_labels(np.zeros((5, 2)), 1)


def test_cluster_names_from_extreme_mean_z() -> None:
    z = np.zeros((6, 3))
    z[:3, 1] = 2.0  # cluster 0: high assists
    z[:3, 2] = np.nan  # a stat missing for a whole cluster doesn't break naming
    z[3:, 2] = -1.5  # cluster 1: low blocks
    z[3:, 0] = 0.1  # too small to name
    labels = np.array([0, 0, 0, 1, 1, 1])
    names = cluster_names(z, labels, ["Points", "Assist %", "Block %"])
    assert names == {0: "high Assist %", 1: "low Block %"}


def test_cluster_names_respects_include_and_average_fallback() -> None:
    z = np.zeros((2, 2))
    z[:, 0] = 3.0
    names = cluster_names(z, np.array([0, 0]), ["Points", "TS%"], include=[False, True])
    assert names == {0: "average profile"}
