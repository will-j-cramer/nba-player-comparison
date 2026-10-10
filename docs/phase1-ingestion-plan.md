# Plan: Phase 1 — Basketball-Reference ingestion

## Status
- **Done:** `client.py` (named `http.py` below; renamed to avoid shadowing the stdlib `http`),
  `cache.py`, `records.py` (shared column lists + record types), `transform.py`
  (traded-player collapse, page merge), `schema.py`, `store.py`, `pipeline.py`, with unit tests.
- **Blocked on network access to www.basketball-reference.com:** step 0 (inspect real HTML),
  `columns.py`/`parse.py`, the trimmed 2022-23 fixtures and the fixture-based season test,
  `cli.py`/`__main__.py` + `[project.scripts]`, and the full ingestion run / phase gate.

## Context
The scaffold is committed (`77d8bf7`), but `src/nba_similarity/ingestion/` is still an empty package. Phase 1 fills the SQLite raw tables that every later phase reads. The user wants per-100-possession and advanced stats for every player-season. Decisions from the Q&A:
- **Seasons: 1973-74 → present** (season end years 1974 → current). Per-100 and most advanced stats (USG%, ORB%/DRB%, STL%/BLK%, BPM) don't exist on B-Ref before 1973-74, so the user chose to drop the 1960s rather than ingest half-empty rows.
- **Own thin scraper** (requests + BeautifulSoup) rather than a PyPI package, so we control rate limiting, retries and caching.
- **Network:** this environment's proxy currently blocks `www.basketball-reference.com`. The user is adding it to Allowed domains. Step 0 re-checks access. If it's still blocked (network changes may only apply to a new session), stop and tell the user instead of writing a parser against guessed HTML.

## Traded players: one combined row per player-season
B-Ref lists a traded player once per team, plus a combined row. It used to be labeled `TOT`; since the 2024 redesign it's `2TM`/`3TM`/…. **We keep only the combined row** in the stats tables and normalize its team to `TOT`. Reasons:
- The product compares *seasons*. A 20-game stint is a partial sample, not a season.
- Per-team rows would put the same player in the results 2–3 times with near-identical lines, crowding out real matches.
- B-Ref computes advanced stats (USG%, TS%, BPM, …) on the combined row over the whole season. Recombining team rows ourselves would be lossy.
- What's lost is stint context, such as a role change after a trade. We keep the stint list (team, games, minutes, in order) in `player_season_teams`, so the UI can still show "BRK → PHO" and per-team rows can be added later if needed.

## Schema (`data/nba.sqlite`)
```sql
players (
  player_id TEXT PRIMARY KEY,      -- B-Ref slug, e.g. 'jokicni01'
  name      TEXT NOT NULL
);
player_seasons (                   -- one row per player per season (combined row if traded)
  player_id TEXT NOT NULL REFERENCES players,
  season    INTEGER NOT NULL,      -- end year: 2023 = 2022-23
  age INTEGER, pos TEXT,
  team      TEXT NOT NULL,         -- team code, or 'TOT' if traded
  n_teams   INTEGER NOT NULL,
  g INTEGER, gs INTEGER, mp INTEGER,   -- season totals
  PRIMARY KEY (player_id, season)
);
player_season_teams (              -- stints, in B-Ref order; 1 row for non-traded players
  player_id TEXT, season INTEGER, stint INTEGER, team TEXT, g INTEGER, mp INTEGER,
  PRIMARY KEY (player_id, season, stint),
  FOREIGN KEY (player_id, season) REFERENCES player_seasons
);
per100_stats (                     -- 1:1 with player_seasons, all REAL
  player_id, season, PRIMARY KEY/FK as above,
  fg, fga, fg_pct, fg3, fg3a, fg3_pct, fg2, fg2a, fg2_pct, ft, fta, ft_pct,
  orb, drb, trb, ast, stl, blk, tov, pf, pts, ortg, drtg
);
advanced_stats (                   -- 1:1 with player_seasons, all REAL
  player_id, season, PRIMARY KEY/FK as above,
  per, ts_pct, fg3a_rate, fta_rate, orb_pct, drb_pct, trb_pct, ast_pct, stl_pct,
  blk_pct, tov_pct, usg_pct, ows, dws, ws, ws_per_48, obpm, dbpm, bpm, vorp
);
ingested_seasons (                 -- drives skip-on-rerun
  season INTEGER PRIMARY KEY,
  is_final INTEGER NOT NULL,       -- 0 = season in progress → refetched on rerun
  n_player_seasons INTEGER NOT NULL,
  ingested_at TEXT NOT NULL        -- ISO-8601 UTC
);
```
- Values are stored exactly as B-Ref reports them: fractions for `*_pct` shooting stats and TS% (0.701), percentages for rate stats like `usg_pct` (27.2). This is documented in `schema.py`.
- Stats a season didn't track become **NULL, never 0**. Examples: 3PT before 1979-80, TOV and therefore USG% and TOV% before 1977-78.
- Per-100 and advanced stats get separate tables because they mirror the two source pages. Both join 1:1 on `(player_id, season)`.

## Modules (`src/nba_similarity/ingestion/`)
| File | Responsibility |
|---|---|
| `client.py` (planned as `http.py`) | `PoliteClient`: descriptive User-Agent; at least 4.0 s between requests (`time.monotonic`); timeout; retries with exponential backoff and jitter (5 s × 2ⁿ, max 5 attempts) on connection errors, timeouts and 5xx. On 429 it honors `Retry-After` up to 300 s; beyond that it raises `RateLimitedError`, because B-Ref can block a client for up to an hour, and since the run resumes from cache, aborting beats hammering. 404 → `PageNotFoundError`. `sleep`/`clock`/`session` are injectable for tests. |
| `cache.py` | `fetch_page(season, page, client, raw_dir, refresh=False) -> str`. Reads `data/raw/NBA_{season}_{page}.html` if present, otherwise fetches and writes it. |
| `columns.py` | Maps B-Ref `data-stat` attribute names to our column names for both tables, checked against real HTML in step 0. |
| `parse.py` | Pure functions. `parse_table(html, table_id) -> list[dict]` reads cells by `data-stat`, also finds tables hidden in HTML comments, drops repeated header rows and the "League Average" row, takes `player_id` from `data-append-csv`, and turns `''` into `None`. `collapse_traded(rows)` returns combined rows plus stints. |
| `schema.py` | DDL and `init_db(conn)`; turns on `PRAGMA foreign_keys`. |
| `store.py` | `write_season(conn, season, ..., is_final)` runs as **one transaction**: delete the season's rows, insert players (upsert name) and the four tables, then upsert `ingested_seasons`. A crash leaves no half-written season. `final_seasons(conn) -> set[int]`. |
| `pipeline.py` | `ingest(seasons, conn, client, raw_dir, force=False) -> IngestReport`. Skips seasons already final in `ingested_seasons` (unless `--force`). Fetches `per_poss` and `advanced`, parses, cross-checks that both pages have the same player set (logs a warning on mismatch), and writes. A season is final once July 1 of its end year has passed. The in-progress season is refetched, bypassing the HTML cache. A 404 on the newest season ("not started yet") is logged and skipped. |
| `cli.py` + `__main__.py` | `python -m nba_similarity.ingestion [--start 1974] [--end <current>] [--db data/nba.sqlite] [--raw-dir data/raw] [--force] [--delay 4.0]`. It logs progress per season and prints a summary (fetched / skipped / rows). The same CLI is exposed as `nba-ingest` via `[project.scripts]`. |

Other edits:
- **`pyproject.toml`:** add `[build-system]` (setuptools) and `[project.scripts]`, so `pip install -e .` provides `nba-ingest`.
- **`CLAUDE.md`:** data starts in 1973-74; record the traded-player decision; allow the in-progress season as the one refetch exception; add ingestion commands.
- **`requirements.txt`:** unchanged; requests, bs4 and lxml are already listed.

## Tests (`tests/ingestion/`, no network)
- **Fixtures:** the real 2022-23 pages (`NBA_2023_per_poss.html`, `NBA_2023_advanced.html`), trimmed to just the stats `<table>` to keep them small, in `tests/fixtures/`. A tiny script `scripts/trim_fixture.py` does the trimming so it can be repeated.
- **`test_pipeline.py`** (the requested test): a fake client serves the fixtures and `ingest([2023])` runs into a tmp SQLite DB. It asserts that:
  - `per100_stats` and `advanced_stats` have exactly the expected columns;
  - the season loads a reasonable number of rows (400–700 unique player-seasons), with no duplicate keys;
  - a traded player (Kevin Durant, `duranke01`) has `team='TOT'`, `n_teams=2` and 2 stints;
  - 2–3 values match numbers copied from B-Ref (e.g. Jokić's TS% and USG%).
- **`test_pipeline.py` skip/resume:** a second run makes 0 fetches; `--force` refetches; an in-progress season gets refetched.
- **`test_http.py`:** a 503→200 retry waits with backoff; 429 honors `Retry-After`; it gives up after max attempts; consecutive requests are spaced ≥ delay apart (fake clock/sleep, so tests run instantly).
- **`test_parse.py`:** header and League Average rows are dropped; empty cells become None; `2TM`/`TOT` both collapse.

## Execution order
0. Re-check access to B-Ref. Fetch the 2023 pages plus one 1970s page (1974) to confirm the table ids, `data-stat` names, the combined-row label and which columns are empty in early seasons. Save the trimmed fixtures.
1. Write `http.py`, `cache.py`, `parse.py`/`columns.py` and `schema.py`/`store.py` with their unit tests.
2. Write `pipeline.py` and `cli.py`, plus the pipeline tests.
3. Run `ruff check .`, `ruff format --check .` and `pytest`.
4. Run the full ingestion (54 seasons × 2 pages ≈ 108 requests ≈ 8 min).
5. Commit and push to `claude/zen-heisenberg-rye7m6`. Raw HTML and the DB stay gitignored; only the trimmed fixtures are committed.

## Verification (phase gate per CLAUDE.md)
- `pytest` and `ruff` are green.
- After the full run: query rows per season (expect ~250 in the mid-70s to ~550–600 recently, with no gaps from 1974 to the present). Spot-check 3 players across eras (e.g. Kareem 1975-76, Jordan 1987-88, Jokić 2022-23) against B-Ref. Confirm that 1970s rows have NULL 3PT and TOV, not 0.
- Re-running the CLI immediately reports every final season as skipped and makes 0 requests.
- Share the per-season count table and spot-checks with the user before Phase 2 starts.
