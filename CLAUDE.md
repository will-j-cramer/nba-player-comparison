# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Project goal

An NBA player similarity search. A user picks a **player-season** (e.g. Nikola Jokić, 2022-23)
and gets back the **most statistically similar player-seasons in NBA history**, each with a
plain-language **explanation of why it matches** (e.g. "both averaged ~25 PPG on 58% TS with
high usage and elite assist rate").

Explanations are a first-class requirement, not an afterthought: every similarity score must be
decomposable into per-feature contributions that the UI can surface.

## Data source

[Basketball-Reference](https://www.basketball-reference.com/) season pages, scraped: the
per-100-possession (`NBA_{year}_per_poss.html`) and advanced (`NBA_{year}_advanced.html`)
league pages. Coverage starts with **1973-74** (end year 1974), the first season with
per-100-possession stats; seasons are identified by their end year (2023 = 2022-23).

Scraping etiquette (non-negotiable):
- Rate limit to **≤ 1 request every 3–4 seconds** (B-Ref blocks clients above ~20 req/min).
- Send a descriptive `User-Agent`.
- **Cache raw HTML** under `data/raw/` and never re-fetch a page that is already cached.
  The one exception: a season still in progress (or last loaded while in progress) is
  refetched so its stats are current.
- Follow Basketball-Reference's terms of use; do not redistribute scraped data.

## Architecture

Four phases, each its own package under `src/nba_similarity/`:

```
Basketball-Reference
        │  (rate-limited HTTP, cached HTML in data/raw/)
        ▼
 ingestion/   ──►  SQLite raw tables (per-game, per-100, advanced, by season)
        ▼
 features/    ──►  SQLite feature tables (cleaned, era-adjusted, normalized)
        ▼
 similarity/  ──►  top-N matches + per-feature contribution breakdowns
        ▼
 ui/          ──►  Streamlit app (read-only over SQLite)
```

### 1. `ingestion/`
- `client.py`: rate-limited HTTP (≥ 4 s between requests), retries with exponential backoff on
  connection errors/5xx, honors short `Retry-After` on 429 and aborts on long ones.
- `cache.py`: raw HTML cache in `data/raw/`; parsing reads from the cache.
- `parse.py` / `columns.py`: B-Ref HTML → raw rows keyed by our column names (cells are
  identified by their `data-stat` attribute). `records.py` holds the column lists that the
  schema and parser share.
- `transform.py`: rows → one `PlayerSeason` per player per season.
- `schema.py` / `store.py`: SQLite schema; each season is written in one transaction and
  recorded in `ingested_seasons`, which makes re-runs skip completed seasons.
- **Traded players:** keep only B-Ref's combined row (`TOT`, or `2TM`/`3TM` since 2024),
  stored with team `TOT`; per-team stints go to `player_season_teams`. Similarity compares
  whole seasons, per-team rows would duplicate players in results, and B-Ref computes the
  advanced stats on the combined row over the full season.

### 2. `features/`
Run with `nba-features`; definitions live in `definitions.py`.
- **12 rate stats in 5 categories:** scoring (pts/100, USG%, 3PA/100, FT rate), efficiency
  (TS%, FT%), playmaking (AST%, TOV%), rebounding (ORB%, DRB%), defense (STL%, BLK%).
  Composites (PER/BPM/WS/VORP), 3P%/FG% and ORtg/DRtg are deliberately excluded.
- **3-point volume, not shot mix:** 3PA per 100 possessions replaced 3PA rate (3PA/FGA),
  which put Curry 2015-16 only +1.1 SD above his league (spot-up role players take a bigger
  *share* of threes).
- **Points per 100 is display-only** (`in_distance = False`): kept in the tables and in
  explanations, but zero weight in the similarity distance, because it is roughly usage ×
  efficiency and would double-count USG% and TS%. So 11 stats drive the distance.
- **Minimum minutes:** 500, scaled by schedule length (1998-99: 305, 2011-12: 402,
  2019-20: 433, 2020-21: 439; a season in progress uses games played so far).
- **Era adjustment:** z-scores within each season, using mean and population SD over
  qualified players only, clipped to ±4. Raw values are stored next to the z-scores.
- **Transforms before z-scoring** (`transforms.py`; recorded in `feature_definitions`):
  square root for the right-skewed 3PA/100, FT rate, ORB% and BLK%, so extreme seasons stay
  distinguishable instead of piling up at the clip. FT% is shrunk toward the season's league
  FT% as if the player took 50 extra league-average attempts (attempts estimated as
  FTA/100 × MP/48), so tiny samples can't look elite. Z-scores use this basis value, but the
  stored raw columns are always the observed stats, which explanations should display.
- **Missing values stay NULL, never filled.** A stat counts for a season only if ≥ 90% of
  qualified players have it (1973-74 TOV is Seattle-only, so it is dropped that season).
  Era gaps: 3PT stats before 1979-80, USG%/TOV% before 1977-78.
- **Output tables:** `feature_definitions` (category/label/transform per feature),
  `feature_season_stats` (per-season basis mean/sd, coverage, observed `raw_mean`), `player_season_features`
  (raw + `z_<feature>` + `n_features` per qualified player-season). They are rebuilt whole
  on every run and have no foreign keys into ingestion tables, so re-ingesting is never
  blocked; re-run `nba-features` after `nba-ingest`.

### 3. `similarity/`
Run with `nba-similar "Nikola Jokic" 2023`. The engine is
`SimilarityEngine(FeatureMatrix, SimilarityConfig)`: build it once, query it many times.
- **Pipeline:** z-scores from `player_season_features` → optional sklearn `StandardScaler` →
  per-feature weights → optional sklearn `PCA` (int components or variance share) →
  euclidean or cosine scores.
- **Weights:** every distance stat weighs the same by default (1/11; points per 100 is
  display-only); `category_weights` multiply a
  category's stats (`--weight scoring=2`, 0 drops it). Do **not** split a category's weight
  over its stats: that gave each stat in a 2-stat category double the weight of a scoring
  stat. Single extreme stats then took over matches in the archetype validation (Chris
  Paul → Eric Murdock on STL%; Shaq → poor FT shooters).
- **Score (0–100), `scoring.py`:**
  - **Euclidean:** `100 × (1 − D² / E[D²])`, where E[D²] is the expected distance from
    *this query* to a random player-season. 0 means "no closer than a typical season".
    Because the baseline is per query, outliers aren't penalised, and matching a query's
    distinctive traits earns more than matching average ones.
  - **Cosine:** `100 × cos` of the weighted vectors.
  - Negative scores are clipped to 0.
- **Exact explanations:** every score is a sum of per-feature contributions, also under PCA
  (via the projection P = VᵀV); `Match.contributions` has them all. For display, each match
  has `similar_traits` (highest-scoring stats where the seasons are within `close_z` = 1 SD:
  genuinely alike) and `differences` (shared stats more than 1 SD apart). A stat can earn
  points while still being a difference (e.g. Harden's DRB% vs Jokić's), so never present
  raw contribution rank as "alike".
- **Missing values:** a stat the query has but a candidate lacks counts as *typical*: 0
  points, but it stays in the score's denominator. That stops pre-1980 seasons from gaining
  by skipping the query's distinctive stats (e.g. Curry's 3-point volume). No value is filled in.
  Candidates must also share `min_shared` (default 70%: 1973-77 seasons share 8/11) of the
  weight. With PCA, a query is
  compared only with seasons that have all of the query's stats.
- **Own seasons:** excluded by default (`include_same_player=True` to include them); the
  query row itself is never returned.
- **One season per comp player** by default (`distinct_players`, CLI `--all-seasons` to
  turn off): each comp player appears once, with their best-matching season.

### 4. `ui/`
Run with `nba-app` (`ui/launch.py`: `python -m streamlit run` with this package's Python),
database from `$NBA_DB`, default `data/nba.sqlite`.
- **Layering:** `view.py` (tables and series built from engine objects) and `charts.py`
  (Plotly figures) are pure and never import streamlit; `app.py` is only layout and caching.
  Engine logic stays in `similarity/`. Shared text helpers (`season_label`,
  `format_contribution`) are in `similarity/explain.py`.
- **Caching:** `st.cache_resource` holds the feature matrix and one `SimilarityEngine` per
  settings combination; `st.cache_data` holds search results, player options and league
  averages. Every key includes the database file's mtime, so re-running `nba-features`
  refreshes the app without a restart.
- **Reads only SQLite;** never scrapes or recomputes features.
- **Comparison view:** verdicts use the engine's `close_z` rule for every stat; points per
  100 is shown as "context". The radar plots the 11 distance stats as within-season
  percentiles (Φ(z)), so eras compare as the engine sees them. Missing stats are gaps, and
  a profile with gaps is drawn without fill so nothing is implied. The colors are slots 1–2
  of the validated palette, with light/dark steps.

## Conventions

- **Python 3.11+.**
- **Type hints** on all public functions and methods.
- **pytest** for tests, under `tests/`, mirroring the `src/nba_similarity/` layout.
- **SQLite** (stdlib `sqlite3`) is the single source of truth. Database lives at
  `data/nba.sqlite` and is gitignored.
- **Small, focused modules**: one responsibility per module. Keep pure transformation logic
  separate from I/O (network, disk, database) so it is easy to test.
- **No network calls in tests.** Parser tests use trimmed B-Ref pages in `tests/fixtures/`,
  which are gitignored (scraped data is not redistributed) and built locally with
  `scripts/make_fixtures.sh`; tests needing them skip when they are missing.
- **ruff** for linting and formatting.

## Phase gating

Build the phases **in order**: ingestion → features → similarity → UI.

**Each phase must be verified before the next begins.** Verification means passing tests *and*
a manual sanity check of the output, for example:
- Ingestion: row counts per season look right; a handful of known players' stats match B-Ref.
- Features: no unexpected nulls/zeros; normalized distributions are centered per season.
- Similarity: obvious comps rank highly (e.g. Jokić 2022-23 ≈ Jokić 2021-22); explanations
  reference the features that actually drove the score.

Do not start a downstream phase on top of unverified upstream data.

## Commands

```bash
pip install -r requirements.txt
pip install -e .                 # provides the nba-ingest command
nba-ingest                       # load 1973-74 → present into data/nba.sqlite (resumable)
scripts/make_fixtures.sh         # build parser test fixtures from data/raw
python scripts/check_ingestion.py  # per-season counts, spot checks, NULL checks
nba-features                     # build the feature matrix and print its sanity summary
nba-similar "Nikola Jokic" 2023  # top matches; --metric cosine --pca 0.9 --weight scoring=2
python scripts/validate_archetypes.py  # archetype check: Curry, Shaq, Gobert, CP3, Giannis
pytest
ruff check .
ruff format .
nba-app                          # the app (NBA_DB=path for another database; extra args go to streamlit)
```

Run tools through the project's Python. On machines with conda, its base environment has
its own `streamlit` and `pytest`, which can shadow the venv's. Use `nba-app`,
`python -m streamlit run src/nba_similarity/ui/app.py` and `python -m pytest`, never a bare
`streamlit`.

## Layout

```
src/nba_similarity/
  ingestion/     # scraping, caching, parsing, raw SQLite writes
  features/      # cleaning, era adjustment, normalization
  similarity/    # distance computation, top-N, explanations
  ui/            # Streamlit app
tests/           # mirrors src/nba_similarity/; fixtures/ holds saved HTML
data/raw/        # cached HTML (gitignored)
data/processed/  # derived artifacts (gitignored)
```
