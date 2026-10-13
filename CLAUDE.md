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

Four phases, each its own package under `src/nba_similarity/`, plus `mapping/` for the app's
map page:

```
Basketball-Reference
        │  (rate-limited HTTP, cached HTML in data/raw/)
        ▼
 ingestion/   ──►  SQLite raw tables (per-game, per-100, advanced, by season)
        ▼
 features/    ──►  SQLite feature tables (cleaned, era-adjusted, normalized)
        ▼
 similarity/  ──►  top-N matches + per-feature contribution breakdowns
        │
        ├──► mapping/  ──►  SQLite map table (precomputed 2D UMAP positions)
        ▼
 ui/          ──►  Streamlit app, Search + Map pages (read-only over SQLite)
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
database from `paths.default_db()` (`$NBA_DB`, default `data/nba.sqlite`; every CLI's `--db`
uses the same default).
- **Pages:** `app.py` is the entry point. It draws the shared query sidebar (`sidebar.py`;
  the same widget keys on every page, so the selection survives page switches) and runs
  `st.navigation` over `search_page.py` and `map_page.py`.
- **Layering:** `view.py` (tables and series built from engine objects) and `charts.py`
  (Plotly figures) are pure and never import streamlit; the page scripts are only layout,
  and the cached loaders live in `cache.py`. Engine logic stays in `similarity/` and map
  computation in `mapping/`. Shared text helpers (`season_label`,
  `format_contribution`) are in `similarity/explain.py`.
- **Caching:** `st.cache_resource` holds the feature matrix and one `SimilarityEngine` per
  settings combination; `st.cache_data` holds search results, player options, league
  averages, the map and k-means labels per k. Every key includes the database file's mtime,
  so re-running `nba-features` or `nba-map` refreshes the app without a restart.
- **Reads only SQLite;** never scrapes, recomputes features or builds the map. The one
  exception to "local file": when the database is missing and the `NBA_DB_URL` secret is
  set, `remote_db.py` downloads it once (see Deployment).
- **Comparison view:** verdicts use the engine's `close_z` rule for every stat; points per
  100 is shown as "context". The radar plots the 11 distance stats as within-season
  percentiles (Φ(z)), so eras compare as the engine sees them. Missing stats are gaps, and
  a profile with gaps is drawn without fill so nothing is implied. The colors are slots 1–2
  of the validated palette, with light/dark steps.
- **Map page:** points are colored either by a stat's within-season percentile (single-hue
  blue ramp; seasons lacking the stat are gray "no data") or by k-means cluster (k 2–9).
  Only the first 3 palette slots pass the all-pairs check a scatter needs, so clusters 4–9
  reuse them with a second and third marker shape, and every cluster has a numbered label at
  its centroid and a name in the legend. The query is an ink star with its label; comps are
  ink rings numbered by rank, with a names key under the map, and label sides are chosen
  greedily so they don't overlap. All other points are dimmed. Highlights follow the
  sidebar settings, but positions are fixed (default weights), and the page says so.

### 5. `mapping/`
Run with `nba-map` after `nba-features` (about 1–2 minutes for ~16k seasons, including the
numba compile on first run).
- **Same distance as the engine:** UMAP runs on the engine's standardized z-matrix
  (`SimilarityEngine.standardized()`) with a custom numba metric (`metric.py`): the weighted
  euclidean distance with default weights (points per 100 has weight 0), where a stat only
  one season has counts as *typical*, `w·(z² + 1)` (`w·2` if neither has it). Nothing is
  filled in. UMAP rejects NaN, so missing values are passed as a sentinel (−1e30) that the
  metric recognises.
- **Stored in SQLite** (`store.py`): `player_season_map (player_id, season, x, y)` and
  `map_metadata (built_at, n_neighbors, min_dist, seed, features_signature)`, rebuilt whole.
  `features_signature` hashes the feature table's z-scores; when it no longer matches, the
  map page warns that the map is out of date.
- **Deterministic:** fixed seed (42) with n_neighbors 15 and min_dist 0.1 (CLI flags).
- **Clusters** (`cluster.py`): k-means on the 2D positions at view time (cheap, so k is a
  slider and clusters match the visual groups), numbered by size and named by their two most
  extreme mean z-scores among the distance stats ("high AST%, high USG%").

## Deployment (Streamlit Community Cloud)
- The database is never committed: the repo is public and scraped data must not be
  redistributed. `scripts/export_app_db.py` writes a slim copy holding only the tables the
  app reads (`players`, the three feature tables, the two map tables) to
  `data/processed/nba_app.sqlite`; it refuses a missing or stale map. The owner uploads it
  somewhere private (README: a private GitHub release asset).
- `ui/remote_db.py` (pure, no streamlit): `ensure_db` streams `NBA_DB_URL` with an optional
  `Authorization: Bearer NBA_DB_TOKEN` and `Accept: application/octet-stream` (GitHub's
  asset API needs it), checks the SQLite header, and renames into place atomically. The
  file is named after a hash of the URL (`download_path`), so a new URL forces a fresh
  download. `ui/sidebar.py` reads the secrets and caches the download per process.
- `streamlit_app.py` (repo root, Cloud's default main file) puts `src/` on `sys.path` and
  calls `ui.app.main()`; `app.py` runs `main()` only under `__main__`.
- `.streamlit/secrets.example.toml` documents the keys; `.streamlit/secrets.toml` is
  gitignored.

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
- **Docstrings** on every public module, class and function (one line is fine).
- Season labels come from `nba_similarity/seasons.py` (re-exported by `similarity/explain.py`);
  don't redefine `season_label`.

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
pip install -e .                 # provides the nba-* commands
nba-ingest                       # load 1973-74 → present into data/nba.sqlite (resumable)
scripts/make_fixtures.sh         # build parser test fixtures from data/raw
python scripts/check_ingestion.py  # per-season counts, spot checks, NULL checks
nba-features                     # build the feature matrix and print its sanity summary
nba-similar "Nikola Jokic" 2023  # top matches; --metric cosine --pca 0.9 --weight scoring=2
python scripts/validate_archetypes.py  # archetype check: Curry, Shaq, Gobert, CP3, Giannis
pytest
ruff check .
ruff format .
nba-map                          # precompute the 2D map for the app (re-run after nba-features)
python scripts/export_app_db.py  # slim app DB for deployment → data/processed/ (never commit)
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
  mapping/       # 2D UMAP map (precomputed), k-means clusters
  ui/            # Streamlit app (Search and Map pages)
  paths.py       # default database location ($NBA_DB)
  seasons.py     # season_label
streamlit_app.py # Streamlit Community Cloud entry point
tests/           # mirrors src/nba_similarity/; fixtures/ holds saved HTML
data/raw/        # cached HTML (gitignored)
data/processed/  # derived artifacts, e.g. the exported app DB (gitignored)
```
