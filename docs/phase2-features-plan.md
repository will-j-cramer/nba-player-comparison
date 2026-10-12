# Plan: Phase 2 — Feature engineering

## Status
- **Implemented:** every module below, `nba-features`, and 27 new tests (76 total).
  Verified here on 1973-74 and 2022-23: 367 qualified rows in 2022-23, per-season z
  mean ≈ 0 and SD ≈ 1, the 1973-74 gaps (USG%, 3PA rate, TOV%) are NULL, and the top-5
  lists are plausible (Haliburton/Jokić on AST%, Drummond on DRB%).
- **Change from the plan:** `player_season_features` has no foreign key to
  `player_seasons`. Re-ingesting a season after features exist hit that constraint (a
  regression test covers it), and the table is rebuilt whole anyway.
- **First full run (owner's machine, 2026-10-11):** 15,990 qualified player-seasons, means
  ≈ 0 per season, gaps only in the 1970s. Two problems:
  - FT% leaders were 1.000 shooters on a handful of attempts.
  - 3PA rate (SD down to 0.79, 1979-81 role players at +4) and BLK% (95 values at the
    clip) piled up at ±4.
- **Follow-up (the owner chose both fixes):**
  - FT% is shrunk toward the league average with a 50-attempt prior.
  - 3PA rate, FT rate, ORB% and BLK% are square-rooted before z-scoring.
  - Raw columns stay observed; `feature_definitions.transform` and
    `feature_season_stats.raw_mean` record the basis.
  - Locally (1974 + 2023): FT% top-5 became Lillard, Herro, Barry, Durant and SGA; clipped
    BLK%/ORB%/FT rate values went from 4/2/1 to 0/0/0. 85 tests pass.
- **Pending:** a re-run of `nba-features` on the full database to close the phase gate.

## Context
Phase 1 is done and verified: `data/nba.sqlite` holds 22,105 player-seasons (1973-74 → 2025-26) in `player_seasons`, `per100_stats` and `advanced_stats`. Phase 2 turns those into an era-adjusted feature matrix that the Phase 3 similarity engine compares. The user approved all of the recommendations:
- **12 rate stats in 5 categories:**
  - scoring: pts/100, usg_pct, fg3a_rate, fta_rate
  - efficiency: ts_pct, ft_pct
  - playmaking: ast_pct, tov_pct
  - rebounding: orb_pct, drb_pct
  - defense: stl_pct, blk_pct
- **Excluded:** composites (PER/BPM/WS/VORP), 3P%/FG% and ORtg/DRtg.
- **Era adjustment:** z-scores within each season. Mean and SD come from qualified players only (population SD). Values are capped at ±4, and raw values are kept alongside.
- **Minimum minutes:** 500, scaled by schedule length:
  - 1999 → 50 games (305 min)
  - 2012 → 66 games (402 min)
  - 2020 → 71 games (433 min)
  - 2021 → 72 games (439 min)
  - a season in progress → its max games played so far
- **Missing values:** stay NULL.
  - A stat counts for a season only if ≥ 90% of qualified players have it; otherwise its z-scores are NULL for that season (raw values kept).
  - Coverage is recorded per season and per row.
  - Category-balanced weighting and comparing on shared stats belong to Phase 3.

## Package layout (`src/nba_similarity/features/`)
Pure transforms are kept separate from I/O, matching the ingestion package.

| File | Responsibility |
|---|---|
| `definitions.py` | `FeatureSpec(name, table, column, category, label)`, the 12 `FEATURES` and the `CATEGORIES` order. Constants: `MIN_MINUTES = 500`, `FULL_SCHEDULE = 82`, `SHORT_SCHEDULES = {1999: 50, 2012: 66, 2020: 71, 2021: 72}`, `COVERAGE_MIN = 0.90`, `Z_CLIP = 4.0`. |
| `load.py` | `load_player_seasons(conn) -> pd.DataFrame`: one row per player-season, joining identity columns (name, season, age, pos, team, g, mp) with the 12 raw columns. Also returns the `is_final` flag per season from `ingested_seasons`. |
| `qualify.py` | Pure. `min_minutes(season, games) -> int`; `schedule_games(season, df, is_final) -> int` (short-season table, else 82; in-progress uses max games); `qualify(df, ...) -> df` keeps rows with mp ≥ threshold. |
| `normalize.py` | Pure. `season_stats(df) -> DataFrame[season, feature, n, mean, sd, coverage, covered]`, where `covered` means coverage ≥ 0.90, n ≥ 2 and sd > 0. `zscores(df, stats) -> df` adds `z_<feature>`, NULL where the season isn't covered or the value is missing, clipped to ±4. It also adds `n_features` (count of non-null z values). |
| `store.py` | `write_features(conn, features, stats)`: one transaction that replaces three tables. The features are derived, so the whole matrix is rebuilt each run. |
| `pipeline.py` | `build_features(conn) -> FeatureReport` (load → qualify → stats → z → store); the report holds counts. |
| `summary.py` | `print_feature_summary(conn, out=sys.stdout)`: the sanity report described below. |
| `cli.py`, `__main__.py` | `nba-features [--db data/nba.sqlite] [--summary-only]`: builds the features, then prints the summary. Registered in `[project.scripts]`. |

## SQLite output tables
```sql
feature_definitions (feature TEXT PK, category TEXT, label TEXT, source TEXT, ordinal INTEGER)
feature_season_stats (season, feature, n, mean, sd, coverage REAL, covered INTEGER,
                      PRIMARY KEY (season, feature))
player_season_features (player_id, season, mp, min_minutes, n_features,
                        <feature> REAL ×12   -- raw value
                        z_<feature> REAL ×12 -- capped z-score, NULL if missing/uncovered
                        PRIMARY KEY (player_id, season),
                        FOREIGN KEY → player_seasons)
```
- **`feature_definitions`:** lets Phase 3 and the UI read categories and labels from the database. CLAUDE.md says the UI reads only SQLite.
- **`feature_season_stats`:** lets explanations turn a z-score back into context ("league avg 22.1").

## Summary report (`print_feature_summary`)
1. **Overview:** player-seasons in and qualified (count and %), number of seasons, and the threshold used for the short seasons.
2. **Per-feature table:** category, first covered season, non-null z count, number of clipped values, and the raw value's median in 1980 vs 2025 for an era sense check.
3. **Normalization check:** across covered seasons, the largest |mean z| (expect ≈ 0) and the SD range (expect ≈ 1, slightly below 1 where clipping bites).
4. **Coverage by decade:** share of rows with each z non-null, which should show the 3PA rate/USG%/TOV% gaps before 1980.
5. **Top 5 player-seasons per feature by z** (name, season, raw, z), as an eyeball test that e.g. Jokić/Westbrook top AST% and Rodman tops ORB%.
6. **Distribution of `n_features`** (how many of 12 stats each row has).

## Tests (`tests/features/`)
- **`test_qualify.py`:** thresholds 500/305/402/433/439; in-progress season uses max games; filtering.
- **`test_normalize.py`:** builds synthetic frames. Checks per-season mean 0 and SD 1; clipping at ±4; a season under 90% coverage gets NULL z but keeps raw values; an SD of 0 counts as uncovered; individual NULLs stay NULL; `n_features`.
- **`test_pipeline.py`:** writes two synthetic seasons through `ingestion.store.write_season`, one pre-1980 with NULL fg3a_rate/usg_pct/tov_pct. Runs `build_features` and checks table columns, row counts, NULL z for the era gaps, the `feature_definitions` rows, and that re-running gives the same result.
- **`test_real_season.py`:** loads the 2023 fixture through ingestion, then the features. Expects 367 qualified rows (matching the earlier MP ≥ 500 count), all 12 stats covered, and Jokić's z_ast_pct among the top. It skips when fixtures are missing, using the existing `tests/ingestion/conftest.read_fixture`.
- **`test_summary.py`:** the summary runs on the synthetic database into a StringIO, and every section header appears.

## Other edits
- **`pyproject.toml`:** add `nba-features = "nba_similarity.features.cli:main"`.
- **`CLAUDE.md`:** in the features section, record the chosen stats, the z-score/clip/threshold/coverage rules and the output tables; add `nba-features` to Commands.
- **`docs/phase2-features-plan.md`:** a copy of this plan with a status section, as for Phase 1.

## Verification
- `ruff check .`, `ruff format --check .` and `pytest` all pass here. The local sandbox database has 1974 and 2023 loaded, so I can run `nba-features` on it and read the summary (z mean ≈ 0, SD ≈ 1; 1974 lacks 3PA rate/USG%/TOV%).
- Commit and push to `claude/zen-heisenberg-rye7m6`.
- **User's phase gate:** on their machine, `git pull && pip install -e . && nba-features`, then paste the summary. Before Phase 3 I review it: normalization ≈ 0/1, coverage gaps only where expected, top-5 lists plausible, about 15–16k qualified rows.
