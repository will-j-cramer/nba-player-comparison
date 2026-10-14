# Phase 6 — Map page (UMAP)

## Status
- **Implemented:**
  - the `mapping/` package (`metric.py`, `embed.py`, `cluster.py`, `store.py`, `cli.py`) and
    the `nba-map` command;
  - `SimilarityEngine.standardized()`;
  - the app split into an entry point with a shared sidebar (`app.py`, `sidebar.py`,
    `cache.py`) and two pages (`search_page.py`, unchanged in behaviour, and `map_page.py`);
  - `view.map_frame` / `view.align_map` and `charts.map_figure` / `charts.label_positions`.
  - 26 new tests (211 total), including headless runs of the Map page (no map, stat and
    cluster coloring, stale map).
- **Verified here** on the local database (1973-74 and 2022-23, 540 seasons) with a real
  `nba-map` (26 s, mostly the numba compile) and `nba-app` plus Playwright screenshots:
  - Jokić 2022-23 sits among the bigs, next to his top comps Sabonis, Williamson and
    Şengün; Harden, Dončić, LeBron and Giannis sit together in the high-usage creator
    region (cluster "high USG%, high AST%").
  - Bigs, creators and low-usage role players form separate regions.
  - Eras mix only partly: of a 1973-74 season's 10 nearest neighbours on the map, 50% are
    also 1973-74, against a 32% base rate. That is expected: those seasons lack 3PA, USG%
    and TOV%, and a missing stat counts as typical, so they share that part of their
    distance. Worth rechecking on the full database.
  - Labels: comp names overlapped around the star in the first version, so comps are now
    numbered rings with a names key under the map, and label sides are chosen to avoid
    overlaps.
- **Full database:** the owner's `nba-map` placed all 15,990 qualified seasons in 54 s. The
  era-mixing check above is still worth repeating on that map.

## Requirements → implementation
| Requirement | Where |
|---|---|
| 2D map of all player-seasons via UMAP | `nba-map` → `mapping.embed.build_embedding` over the engine's standardized z-matrix, with the engine's own distance (`mapping.metric.typical_distance`) |
| Similar players cluster visually | same distance as the search, so neighbours on the map are near-matches under default weights |
| Hover: player, season, key stats | `view.map_frame` hover text: name, season, PTS/100, USG%, TS%, AST%, DRB%, 3PA/100 (observed; "n/a" when missing), plus the color value and cluster |
| Color by a stat or k-means cluster | page radio; stat → within-season percentile on a blue ramp (gray = no data); cluster → `mapping.cluster.kmeans_labels` on the 2D map, k slider 2–9, names from `cluster_names` |
| Highlight the selected player and top comps | shared sidebar → same cached search as the Search page; star + label for the query, numbered rings for comps, other points dimmed |
| Precompute and cache to disk | `player_season_map` + `map_metadata` tables in the SQLite database, with a features signature for staleness |

## Decisions
- **No imputation for UMAP.** UMAP rejects NaN, and zero-filling would pull pre-1980
  seasons toward the league average on the stats they lack. The custom metric applies the
  engine's rule instead (a stat only one season has costs `w·(z² + 1)`, the expected
  distance to an unknown typical value), with missing values passed as a sentinel.
- **Fixed default weights.** The embedding takes about a minute, so it is precomputed and
  can't follow the sliders. The highlighted comps do follow them, and the page says comps
  can sit away from the star under non-default settings.
- **Stored in SQLite, not a file.** SQLite is the project's single source of truth; the map
  tables are rebuilt whole like the feature tables, and the app's cache keys include the
  database mtime.
- **k-means on the 2D map, not the 11-D stats.** The clusters are meant to name the regions
  the viewer sees. It is fast enough to run per k at view time.
- **Colors.** Clusters use only the 3 palette slots that pass the all-pairs check
  (`validate_palette.js --pairs all`: worst CVD ΔE 9.2 light / 9.4 dark), times 3 marker
  shapes for k ≤ 9, with numbered centroid labels so identity never rests on color.
  Highlights use neutral ink so they never collide with a data color.

## Re-running
`nba-ingest` → `nba-features` → `nba-map`. If the features change and the map isn't
rebuilt, the Map page shows "Re-run `nba-map`" and seasons missing from the map are counted.
