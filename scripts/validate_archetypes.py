"""Archetype validation: do well-known player-seasons get comps that play the same way?

    python scripts/validate_archetypes.py [--db data/nba.sqlite] [-n 10]
        [--metric cosine] [--weight scoring=2] [--pca 0.9]

For each case it prints:
- the query's z-score profile (what the engine actually sees),
- the top-n comps with their alike/differ stats, era and an archetype-fit check:
  how many of the archetype's signature traits the comp shares (same direction, |z| >= 0.75),
- a summary: average fit, flagged comps, decades of the comps and the share of the score
  each category supplies, which helps tell feature, normalization and weighting problems apart.

The signature traits are a human's idea of the archetype, written down so comps can be checked
against it; they are not used by the engine.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nba_similarity.similarity.config import SimilarityConfig
from nba_similarity.similarity.engine import Match, SimilarityEngine

TRAIT_Z = 0.75  # a comp "has" a trait when its z-score is at least this far in that direction
FLAG_BELOW = 0.6  # comps sharing fewer than this share of the signature traits are flagged


@dataclass(frozen=True)
class Case:
    player_id: str
    season: int
    label: str
    archetype: str
    signature: tuple[tuple[str, int], ...]  # (feature, +1 high / -1 low)


CASES: tuple[Case, ...] = (
    Case(
        "curryst01", 2016, "Stephen Curry 2015-16", "high-volume, hyper-efficient shooter",
        (("fg3a_rate", +1), ("ts_pct", +1), ("ft_pct", +1), ("usg_pct", +1), ("pts", +1)),
    ),
    Case(
        "onealsh01", 2000, "Shaquille O'Neal 1999-00", "dominant low-post center",
        (("usg_pct", +1), ("pts", +1), ("fta_rate", +1), ("orb_pct", +1), ("drb_pct", +1),
         ("blk_pct", +1), ("ft_pct", -1), ("fg3a_rate", -1)),
    ),
    Case(
        "goberru01", 2022, "Rudy Gobert 2021-22", "rim-protecting, rim-running center",
        (("blk_pct", +1), ("drb_pct", +1), ("orb_pct", +1), ("ts_pct", +1),
         ("fg3a_rate", -1), ("usg_pct", -1)),
    ),
    Case(
        "paulch01", 2008, "Chris Paul 2007-08", "pass-first floor general",
        (("ast_pct", +1), ("stl_pct", +1), ("tov_pct", -1), ("ft_pct", +1), ("usg_pct", +1)),
    ),
    Case(
        "antetgi01", 2020, "Giannis Antetokounmpo 2019-20", "downhill, rim-attacking forward",
        (("usg_pct", +1), ("pts", +1), ("fta_rate", +1), ("drb_pct", +1), ("ast_pct", +1),
         ("fg3a_rate", -1), ("ft_pct", -1)),
    ),
)  # fmt: skip


@dataclass(frozen=True)
class CompReview:
    match: Match
    traits_shared: tuple[str, ...]
    traits_missing: tuple[str, ...]  # comp lacks the trait (wrong direction or too mild)
    traits_unknown: tuple[str, ...]  # comp has no value for the stat (e.g. pre-1980 3PA rate)

    @property
    def fit(self) -> float:
        known = len(self.traits_shared) + len(self.traits_missing)
        return len(self.traits_shared) / known if known else 0.0

    @property
    def flagged(self) -> bool:
        return self.fit < FLAG_BELOW


def has_trait(z: float, direction: int) -> bool | None:
    """True/False whether `z` shows the trait; None when the stat is missing."""
    if np.isnan(z):
        return None
    return bool(z * direction >= TRAIT_Z)


def review_comp(engine: SimilarityEngine, case: Case, match: Match) -> CompReview:
    m = engine.matrix
    row = m.row(match.player_id, match.season)
    shared, missing, unknown = [], [], []
    for feature, direction in case.signature:
        verdict = has_trait(float(m.z[row, m.features.index(feature)]), direction)
        bucket = unknown if verdict is None else shared if verdict else missing
        bucket.append(feature)
    return CompReview(match, tuple(shared), tuple(missing), tuple(unknown))


def category_shares(engine: SimilarityEngine, matches: Sequence[Match]) -> dict[str, float]:
    """Share of the comps' total points supplied by each category (positive points only)."""
    totals: Counter[str] = Counter()
    categories = dict(zip(engine.matrix.features, engine.matrix.categories, strict=True))
    for match in matches:
        for feature, points in match.contributions.items():
            totals[categories[feature]] += max(points, 0.0)
    grand = sum(totals.values()) or 1.0
    return {c: totals[c] / grand for c in dict.fromkeys(engine.matrix.categories)}


def season_label(season: int) -> str:
    return f"{season - 1}-{str(season)[2:]}"


def _stat(c: object) -> str:
    return f"{c.label} {c.query_raw:.3g} vs {c.match_raw:.3g}"  # type: ignore[attr-defined]


def run_case(engine: SimilarityEngine, case: Case, n: int, out=sys.stdout) -> list[CompReview]:
    def emit(line: str = "") -> None:
        print(line, file=out)

    m = engine.matrix
    emit("=" * 100)
    emit(f"{case.label} — expected archetype: {case.archetype}")
    try:
        row = m.row(case.player_id, case.season)
    except KeyError:
        emit(f"  NOT IN FEATURE MATRIX ({case.player_id} {case.season}); skipped")
        return []

    profile = "  ".join(
        f"{f}={m.z[row, j]:+.1f}" if not np.isnan(m.z[row, j]) else f"{f}=n/a"
        for j, f in enumerate(m.features)
    )
    emit(f"query z-profile: {profile}")
    weak = [f for f, d in case.signature if not has_trait(float(m.z[row, m.features.index(f)]), d)]
    if weak:
        emit(f"  note: the query itself is not >= {TRAIT_Z} SD on {weak}; signature may be off")
    signature = ", ".join(f"{'high' if d > 0 else 'low'} {f}" for f, d in case.signature)
    emit(f"signature traits: {signature}")
    emit("")

    matches = engine.find_similar(case.player_id, case.season, n=n)
    reviews = [review_comp(engine, case, match) for match in matches]
    for rank, r in enumerate(reviews, start=1):
        match = r.match
        coverage = "" if match.shared >= 0.999 else f" [{match.shared:.0%} of stats]"
        flag = "  <-- FLAG" if r.flagged else ""
        emit(
            f"{rank:>2}. {match.name:<26} {season_label(match.season)}  {match.score:5.1f}"
            f"{coverage}  fit {len(r.traits_shared)}/{len(r.traits_shared) + len(r.traits_missing)}"
            f"{flag}"
        )
        emit(f"      alike:   {'; '.join(_stat(c) for c in match.similar_traits) or '-'}")
        if match.differences:
            emit(f"      differ:  {'; '.join(_stat(c) for c in match.differences)}")
        if r.traits_missing or r.traits_unknown:
            parts = []
            if r.traits_missing:
                parts.append(f"lacks {', '.join(r.traits_missing)}")
            if r.traits_unknown:
                parts.append(f"no data for {', '.join(r.traits_unknown)}")
            emit(f"      archetype: {'; '.join(parts)}")

    if not reviews:
        emit("  no comps found")
        return reviews
    fits = [r.fit for r in reviews]
    decades = Counter(f"{(r.match.season - 1) // 10 * 10}s" for r in reviews)
    shares = category_shares(engine, [r.match for r in reviews])
    emit("")
    flagged = sum(r.flagged for r in reviews)
    mean_score = np.mean([r.match.score for r in reviews])
    emit(f"summary: mean archetype fit {np.mean(fits):.0%}, flagged {flagged}/{len(reviews)}, "
         f"mean score {mean_score:.1f}")  # fmt: skip
    emit(f"  comps by decade: {', '.join(f'{d} {c}' for d, c in sorted(decades.items()))}")
    emit("  share of points by category: "
         + ", ".join(f"{c} {s:.0%}" for c, s in shares.items()))  # fmt: skip
    return reviews


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=Path("data/nba.sqlite"))
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--weight", action="append", default=[], metavar="CATEGORY=W")
    parser.add_argument("--pca", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None, cases: Sequence[Case] = CASES) -> int:
    args = build_parser().parse_args(argv)
    if not args.db.exists():
        print(f"{args.db} not found; run nba-ingest and nba-features first", file=sys.stderr)
        return 2
    weights = {}
    for item in args.weight:
        category, _, value = item.partition("=")
        weights[category] = float(value)
    pca = None if args.pca is None else (int(args.pca) if args.pca >= 1 else args.pca)
    config = SimilarityConfig(metric=args.metric, category_weights=weights, pca_components=pca)
    conn = sqlite3.connect(args.db)
    try:
        engine = SimilarityEngine.from_db(conn, config)
    finally:
        conn.close()

    print(f"config: metric={config.metric}, weights={weights or 'equal'}, pca={pca}, "
          f"own seasons excluded, min_shared={config.min_shared}")  # fmt: skip
    all_reviews = [run_case(engine, case, args.n) for case in cases]
    reviewed = [r for reviews in all_reviews for r in reviews]
    if reviewed:
        print("=" * 100)
        print(f"overall: mean archetype fit {np.mean([r.fit for r in reviewed]):.0%}, "
              f"flagged {sum(r.flagged for r in reviewed)}/{len(reviewed)} comps")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
