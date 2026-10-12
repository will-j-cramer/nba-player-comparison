# Phase 3 — Similarity engine

## Status
- **Implemented:** `similarity/` (config, data, scoring, engine, lookup, cli), `nba-similar`,
  and 62 new tests (138 total).
- **Verified here on 2022-23 only:**
  - Jokić → Sabonis (AST%, DRB%, TS%).
  - Curry → Lillard/Booker/Irving.
  - Gobert → Zubac/Duren/Mark Williams.
  - Embiid with PCA → Davis/Porziņģis.
- **Phase gate run (owner's machine):**
  - With `--include-same-player`, Jokić 2022-23's top 10 holds 8 of his own seasons;
    2021-22 is #2 at 94.8. This is the CLAUDE.md check, and it passed.
  - Comps for Jordan 1988, Curry 2016 (scoring ×3) and Rodman 1995 (cosine) were mostly
    sensible.
- **Two problems found, both fixed with the owner's choices:**
  1. *Incomplete seasons gained by skipping stats.* Walter Davis 1978-79 (no 3-point data)
     was Curry 2016's #2. Stats a candidate lacks now count as typical (0 points, still in
     the denominator) instead of being dropped from the comparison.
  2. *Explanations were the same for every match and could mislead.* Harden got "+11 points"
     for DRB% at 20.6 vs Jokić's 31.3. Matches now carry `similar_traits` (within 1 SD) and
     `differences`, plus the full `contributions`.
- **Pending:** a re-run of the same five queries on the full database. 149 tests pass.

## Requirements → implementation
| Requirement | Where |
|---|---|
| Standardize, optional PCA | `SimilarityConfig.standardize` (StandardScaler), `pca_components` (PCA, int or variance share) |
| Cosine and euclidean | `SimilarityConfig.metric`; `scoring.masked_scores` / `projected_scores` |
| Per-category weights | `SimilarityConfig.category_weights` → `config.feature_weights` (split evenly within a category) |
| Top-N with a 0–100 score | `SimilarityEngine.find_similar(player_id, season, n)` → `Match.score` |
| Exclude own seasons by default | `SimilarityConfig.include_same_player` (default False) |
| Features that drove the match | `Match.similar_traits` / `Match.differences`; `Match.contributions` (sum to the score) |

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
  The default (no PCA) compares directly on shared stats and counts the candidate's missing
  ones as typical.
