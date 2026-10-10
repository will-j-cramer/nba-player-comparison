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
- Traded players are already collapsed to one row per season by ingestion.
- Era gaps: 3PT stats don't exist before 1979-80 and TOV (so USG%/TOV%) before 1977-78.
  These arrive as NULL from ingestion; keep them missing, never zero.
- Minimum-minutes / games filters to exclude tiny samples.
- Pace / era adjustment and per-season z-score normalization so a 1965 season and a 2023
  season are comparable.
- Persist the feature matrix to SQLite.

### 3. `similarity/`
- Weighted distance (or cosine similarity) over feature vectors.
- Handle missing features explicitly (e.g. compare only on the shared feature set and
  report coverage) — never silently zero-fill.
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
