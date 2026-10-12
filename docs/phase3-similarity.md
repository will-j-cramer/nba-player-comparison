# Phase 3 — Similarity engine

## Status
- **Implemented:** `similarity/` (config, data, scoring, engine, lookup, cli), `nba-similar`,
  and 62 new tests (138 total).
- **Verified here on 2022-23 only:**
  - Jokić → Sabonis (AST%, DRB%, TS%).
  - Curry → Lillard/Booker/Irving.
  - Gobert → Zubac/Duren/Mark Williams.
  - Embiid with PCA → Davis/Porziņģis.
- **Pending:** the phase gate on the owner's full database. Jokić 2022-23 ≈ Jokić 2021-22
  with `--include-same-player`, and the explanations should name the stats that drove the
  match.

## Requirements → implementation
| Requirement | Where |
|---|---|
| Standardize, optional PCA | `SimilarityConfig.standardize` (StandardScaler), `pca_components` (PCA, int or variance share) |
| Cosine and euclidean | `SimilarityConfig.metric`; `scoring.masked_scores` / `projected_scores` |
| Per-category weights | `SimilarityConfig.category_weights` → `config.feature_weights` (split evenly within a category) |
| Top-N with a 0–100 score | `SimilarityEngine.find_similar(player_id, season, n)` → `Match.score` |
| Exclude own seasons by default | `SimilarityConfig.include_same_player` (default False) |
| Features that drove the match | `Match.top_features` (`FeatureContribution.points` sum to the score) |

## Design decisions
- **Query-specific euclidean baseline.** The first version compared against the distance
  between two random seasons. That made outliers score badly (Jokić's best match: 55; ninth:
  0). It also explained matches with average stats ("BLK% 1.8 vs 1.2"). Comparing against
  the expected distance *from the query* fixed both: Jokić → Sabonis 80, explained by AST%,
  DRB% and TS%.
- **Exact decomposition everywhere.** Contributions sum to the score with and without PCA.
  Under PCA they use D² = Σ_j Δ_j (PΔ)_j, so the explanations stay in terms of real stats.
- **PCA and missing values.** PCA needs complete data, so with PCA a query is compared only
  with seasons that have all of its stats: modern queries won't match pre-1980 seasons.
  The default (no PCA) compares on shared stats instead.
