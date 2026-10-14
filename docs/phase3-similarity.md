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
- **Full-database validation:** three rounds, below; the gate passed in round 3.

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

## Archetype validation (`scripts/validate_archetypes.py`)
First full-database run (equal category weights, every season listed). Mean archetype fit
was 85%, with 2/50 comps flagged:
- **Gobert 2022:** excellent (Mutombo, Chandler, Dwight, Whiteside).
- **Shaq 2000:** the right type, but six Duncan seasons, and FT%-driven picks (Mobley,
  Stanley Roberts).
- **Chris Paul 2008:** Stockton, Kidd and Payton, but Eric Murdock was #1 and #2 on STL%.
  Defense supplied 40% of all points.
- **Curry 2016:** Jordan 1989 was #1. Only 2/10 comps were high-volume 3-point shooters.
- **Giannis 2020:** three Duncans with assist rates of 15–19 against his 34.

Diagnosis:
1. **Weighting (main cause).** Splitting each category's weight gave 2-stat-category stats
   10% each against 5% for scoring stats. With the query-specific baseline, one extreme 10%
   stat (STL%, FT%) dominated.
2. **Stat choice.** 3PA rate measures shot mix, not volume. Curry 2015-16 is only +1.1 SD.
3. **Presentation.** Repeat seasons of one comp player crowded the list.
4. **Normalization:** no problem found. Cross-era comps were sensible.

Applied, as the owner chose:
- Equal weight per stat (category multipliers kept).
- One season per comp player by default.

Proposed but not applied:
- B: 3PA per 100 possessions instead of 3PA rate.
- D: cap extreme-stat amplification, only if still needed.
- E: correct the script's signatures for Chris Paul's TOV%, Giannis's 3PA rate and
  Gobert's usage.

### Round 2 (after equal per-stat weights + one season per comp player)
- **Better:** Shaq (Mourning, Robinson, Ewing replaced Mobley and Stanley Roberts) and Giannis
  (LeBron, Luka, Zion, Malone added).
- **Unchanged:** Curry (Walter Davis rose to #3). Chris Paul lost Stockton entirely.
- **Correction:** for candidates with every stat, the ranking depends only on the weighted
  distance. The query-specific baseline only rescales scores, sets the missing-stat penalty
  and shapes explanations. The "share of points" figure the first diagnosis relied on came
  from that baseline, so it overstated the query's extreme stats. Fix D (capping
  amplification) was withdrawn because it can't change rankings.

Applied, as the owner chose:
- **B: 3PA per 100 possessions replaces 3PA rate.**
- **F: Points per 100 is display-only.** Zero distance weight, still shown in alike/differ;
  it double-counted USG% and TS%, which pushed Stockton out for Chris Paul.
- **G: validation diagnostics.**
  - Each category's share of the *distance* (and of the weight) replaced "share of points".
  - A mean signature gap (in SD) per comp.
  - My wrong expected traits were corrected (Chris Paul TOV%, Giannis 3PA, Gobert usage).
- **Side effect handled:** with 11 distance stats, 1973-77 seasons (no 3PA/USG%/TOV%) share
  only 8/11 = 73% of the weight. `min_shared` went from 75% to 70% so they stay comparable.

### Round 3 (after 3PA per 100, display-only points and min_shared 70%), owner's full database
Overall: mean archetype fit **92%** (85% in round 1), mean signature gap 0.90 SD,
**0/50 comps flagged**.

| Case | Top comps | Fit | Gap |
|---|---|---|---|
| Curry 2016 | Ginóbili, Jordan 1990, Arenas, Ray Allen, Chris Paul 2018, Kyrie, Mark Price, Bird | 100% | 1.00 SD |
| Shaq 2000 | Duncan 2004, Griffin, Stanley Roberts, Giannis 2025, Mobley, McDyess, Mourning, McGee | 90% | 0.81 SD |
| Gobert 2022 | Mutombo, Przybilla, Chandler, Dwight Howard, Jarrett Allen, DeAndre Jordan | 100% | 0.78 SD |
| Chris Paul 2008 | Murdock, Haliburton, Kidd, Brandon, Stockton, Payton, Isiah Thomas | 90% | 0.84 SD |
| Giannis 2020 | Cousins, Embiid, Duncan, McGinnis, LeBron, Luka, Griffin, Malone | 82% | 1.09 SD |

- **Better than round 2:**
  - Curry's list now holds real high-volume shooters (Ray Allen, Mark Price, Bird, Kyrie).
    Walter Davis, who has no 3-point data, fell from #3 to #9.
  - Stockton is back in Chris Paul's top 5.
- **Residuals, accepted without a code change:** each is a single comp, and chasing it would
  mean another weighting round with its own side effects.
  - Eric Murdock 1993-94 is still Chris Paul's #1 (90.8). Both seasons sit at the top of
    the steal-% scale (3.9 vs 4.0), and defense supplies 42% of the query-to-comp distance.
    The explanation shows the assist-% gap (52.2 vs 36.9) as a difference, so the
    output isn't misleading.
  - Stanley Roberts (#3) and Evan Mobley (#5, fit 5/8) for Shaq are driven by free-throw %
    and blocks. Their usage gap is listed as a difference.
- **Phase gate: passed.** The obvious-comp check passed earlier (Jokić 2021-22 ranked #2 for
  2022-23 with own seasons included), all five archetypes fit 82–100%, and the explanations
  name the stats that drove each match.
