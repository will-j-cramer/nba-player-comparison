from __future__ import annotations

import io
import sqlite3

from nba_similarity.features.pipeline import build_features
from nba_similarity.features.summary import print_feature_summary


def test_summary_prints_every_section(ingested_db: sqlite3.Connection) -> None:
    build_features(ingested_db)
    out = io.StringIO()
    print_feature_summary(ingested_db, out)
    text = out.getvalue()
    for header in (
        "== Overview ==",
        "== Features ==",
        "== Normalization check",
        "== Share of rows with each z-score, by decade ==",
        "== Top 5 player-seasons per feature",
        "== Features available per row",
    ):
        assert header in text
    assert "27 qualified player-seasons of 29" in text
    assert "2019-20: 433" in text  # shortened season threshold is listed
