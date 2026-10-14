# Phase 7 — Portfolio readiness

## Status
- **README.md:**
  - what the project does and the methodology, with the reasoning behind the feature and
    normalization choices;
  - local setup;
  - step-by-step deployment to Streamlit Community Cloud;
  - limitations, and the data/licensing position (no license; the data belongs to Sports
    Reference and isn't redistributed);
  - screenshots are placeholders, with the image lines commented out and
    `docs/images/{search,comparison,map}.png` as their targets.
- **Deployment support:**
  - `scripts/export_app_db.py` makes a slim app DB;
  - `ui/remote_db.py` downloads it at startup from a private URL in the app's secrets;
  - `streamlit_app.py` is the Cloud entry point;
  - `.streamlit/secrets.example.toml` documents the secrets;
  - `paths.default_db()` gives the app and every CLI the same `$NBA_DB` default.
- **Cleanup:**
  - Removed `IDENTITY_COLUMNS` (×2) and `CATEGORIES` (unused).
  - Removed the one-off `scripts/repair_cache_encoding.py` and its test.
  - Moved the test-only numpy metric into the tests.
  - One `season_label`, in `nba_similarity/seasons.py`; `features` no longer has its own, and
    it can't import from `similarity` because that would invert the layering.
  - The validation script reuses the CLI's weight/PCA parsers (`parse_weights`, `parse_pca`)
    and the NaN-safe formatter.
  - Fixed stale "3PA rate" and "app.py" wording in code and docs.
  - Added `numba` to the requirements.
- **Docstrings:** an AST scan finds no public module, class or function without one in
  `src/`, `scripts/` or `streamlit_app.py`.
- **Kept on purpose:**
  - `fg3a_rate` and the other stored-but-unused advanced columns: ingestion mirrors
    Basketball-Reference, and dropping them would mean a schema change and a re-ingest.
  - `SimilarityEngine.with_config`: public API.

## Verification
- `ruff check .`, `ruff format --check .` and `python -m pytest` pass: 224 tests (+15 new,
  −2 removed with the repair script).
- End to end on the local database (1973-74 and 2022-23):
  1. Exported the slim DB (6 tables, 0.3 MB).
  2. Served it from a local HTTP server that requires a bearer token.
  3. Ran `streamlit run streamlit_app.py` with no database on disk and the URL and token in
     `.streamlit/secrets.toml`.
  - The server logged exactly one request, with `Authorization: Bearer …` and
    `Accept: application/octet-stream`, across three page loads.
  - Search and Map rendered from the downloaded file, and the map was not reported stale.
  - That test caught a bug, since fixed: the sidebar kept using the original path after
    downloading.
- **Not verifiable here:** the actual Streamlit Community Cloud setup (private data repo,
  token, app creation), which the owner does by following the README.
- **Phase 3 gate:** closed with the round-3 validation (92% archetype fit, 0/50 comps
  flagged).

## Follow-up: the owner's local run
The owner's bare `pytest` and `streamlit` resolved to conda's base install, not the project
venv: pytest 8.4.2 with conda's anyio plugin, and a traceback under `/opt/anaconda3`. That
produced three failures:
- missing `umap`;
- `ModuleNotFoundError: nba_similarity`;
- a map-page warning. Older Streamlit treats `st.plotly_chart(height=...)` as a deprecated
  keyword argument.

Fixes:
- The map's height is set in the Plotly figure instead.
- `streamlit>=1.46` is the real floor: the first version with `st.navigation(position="top")`
  and `st.context.theme`, verified by running the full suite in a throwaway venv. 1.43 fails
  outright, and 1.44–1.45 pass only by accident.
- `nba_similarity/environment.py` adds a check. `tests/conftest.py` uses it to stop pytest
  with one message naming the interpreter and the fix, and the app shows an error on a
  Streamlit older than 1.46.
