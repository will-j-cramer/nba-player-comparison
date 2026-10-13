# Phase 5 — Streamlit app

## Status
- **Implemented:** `ui/view.py`, `ui/charts.py`, `ui/app.py`, and `similarity/explain.py`
  (shared by CLI and UI). 22 new tests, including headless runs of the real script via
  `streamlit.testing.v1.AppTest`; 183 tests total.
- **Verified here** against the local database (1973-74 and 2022-23 only) with a real
  `streamlit run` and Playwright screenshots:
  - Jokić 2022-23 → Sabonis / Harden / Luka…;
  - clicking a row switches the side-by-side table and radar;
  - a cross-era row (Boerwinkle 1973-74) shows the 73% caveat, "n/a" verdicts and radar gaps.
- **Pending:** the owner's check on the full database. The similarity phase gate (round-3
  `validate_archetypes.py` output) is also still to be reviewed.

## Requirements → implementation
| Requirement | Where |
|---|---|
| Searchable player dropdown, then season | sidebar `st.selectbox` (type to search) with `view.player_options`; season selectbox of that player's qualified seasons |
| Category weight sliders, own-seasons toggle | 5 sliders (0–3) → `category_weights`; toggle → `include_same_player`; Advanced: results count, metric, one-season-per-player |
| Results table with scores | `st.dataframe` of `view.results_frame`, with Score as a 0–100 progress column |
| Click a result → side-by-side + radar | row selection (`on_select="rerun"`) → `view.comparison_frame` + `charts.radar_figure` |
| Fast | cached matrix, engines, searches, options and league averages, keyed by DB mtime |
| UI separate from engine | `view.py` and `charts.py` never import streamlit (since Phase 6 the layout lives in the page scripts and `cache.py`); engine untouched apart from a NaN-safety fix |

## Found while building
- **Engine bug:** a stat missing for *every* row (NaN mean/variance) made every cosine score
  NaN, because NaN × 0 = NaN. Real data has no such stat, but the app test's synthetic
  database did. Fixed with NaN-safe arithmetic in `scoring.py` and `engine.py`, plus
  regression tests.
- **Layout fixes from screenshots:**
  - short radar labels (labels were clipped);
  - compact comparison headers (the table scrolled sideways);
  - verdicts for every stat, not only the top-3 lists;
  - no radar fill for profiles with gaps (the fill bridged missing stats).

## Troubleshooting: `No module named 'nba_similarity'`
- **Seen on the owner's Mac** with both `(.venv)` and conda `(base)` active: `streamlit`
  resolved to conda base's own Streamlit, whose Python doesn't have the package installed.
  Their earlier `pytest` header showed 8.4.2 instead of the venv's 9.x, so pytest came from
  conda too; tests still passed because `pyproject.toml` puts `src/` on the test path.
- **Fix:** the `nba-app` command (`ui/launch.py`) runs `<venv python> -m streamlit run app.py`.
  Equivalently: `python -m streamlit run src/nba_similarity/ui/app.py`.
- **Check:** `which streamlit pytest nba-app` shows which environment each comes from.
