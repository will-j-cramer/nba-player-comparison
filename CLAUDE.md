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
- **12 rate stats in 5 categories:** scoring (pts/100, USG%, 3PA rate, FT rate), efficiency
  (TS%, FT%), playmaking (AST%, TOV%), rebounding (ORB%, DRB%), defense (STL%, BLK%).
  Composites (PER/BPM/WS/VORP), 3P%/FG% and ORtg/DRtg are deliberately excluded.
- **Minimum minutes:** 500, scaled by schedule length (1998-99: 305, 2011-12: 402,
  2019-20: 433, 2020-21: 439; a season in progress uses games played so far).
- **Era adjustment:** z-scores within each season, using mean and population SD over
  qualified players only, clipped to ±4. Raw values are stored next to the z-scores.
- **Transforms before z-scoring** (`transforms.py`; recorded in `feature_definitions`):
  square root for the right-skewed 3PA rate, FT rate, ORB% and BLK%, so extreme seasons stay
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
- Weighted distance (or cosine similarity) over feature vectors.
- Handle missing features explicitly: compare only on the shared feature set, average
  (don't sum) over it, require a minimum shared share, and report coverage — never zero-fill.
- Weight the five feature categories equally, regardless of how many stats each has.
- Return top-N matches **plus per-feature contributions**, which drive the explanations.

### 4. `ui/`
- Streamlit app at `src/nba_similarity/ui/app.py`: player + season pickers, results table,
  explanation text per match.
- Reads only from SQLite. The UI never scrapes or recomputes features.

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
pytest
ruff check .
ruff format .
streamlit run src/nba_similarity/ui/app.py
```

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
