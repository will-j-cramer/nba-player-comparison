# NBA Player-Season Similarity

**Pick any NBA player-season since 1973-74 and find the most statistically similar seasons in
league history, with a plain-language explanation of why each one matches.**

For example, Nikola Jokić 2022-23 matches Domantas Sabonis 2022-23 because both seasons are
alike on defensive rebound %, true shooting % and turnover %. They differ mainly on assist %:
46.6 against 30.4.

> 📸 **Screenshot placeholder: Search page.** Jokić 2022-23 selected, the results table with
> scores, and the "Alike / Differ" line for the top match.
<!-- ![Search page](docs/images/search.png) -->

> 📸 **Screenshot placeholder: comparison view.** The side-by-side stat table with league
> averages and verdicts, next to the percentile radar chart.
<!-- ![Comparison view](docs/images/comparison.png) -->

> 📸 **Screenshot placeholder: Careers page.** Wembanyama through age 21: the comps table
> with next-season marks, the trajectory chart with the projection band, and the outcome
> shares.
<!-- ![Careers page](docs/images/careers.png) -->

> 📸 **Screenshot placeholder: Map page.** All ~16,000 player-seasons on the UMAP map,
> colored by k-means cluster, with Jokić's star and his numbered top comps.
<!-- ![Map page](docs/images/map.png) -->

---

## What it does

- **Search** any qualified player-season from 1973-74 to the present: about 16,000 seasons
  with at least 500 minutes.
- **Top comps** come with a 0–100 similarity score. One season per comp player is shown by
  default, and the query player's own seasons are excluded unless you include them.
- **Explanations:** each match lists the stats where the two seasons are genuinely alike
  (within 1 SD) and the ones where they differ. Behind that, every score breaks down exactly
  into per-stat contributions.
- **Your own weights:** sliders for scoring, efficiency, playmaking, rebounding and defense.
  For example, set defense to 0 to compare offensive profiles only.
- **Comparison view:** raw stats side by side with each season's league average, plus a radar
  chart of within-season percentiles, so a 1980s season and a 2020s season sit on the same
  scale.
- **Map:** every season placed in 2D with UMAP, so similar players cluster together. Hover a
  point for its stats, color by any stat or by k-means cluster, and see the selected season
  and its comps highlighted.
- **Career trajectories:** "Victor Wembanyama through age 21" finds the players whose careers
  through the same age looked most alike, season by season, and shows what they did next:
  how many played full seasons, played less, or left the league, plus a score-weighted range
  for their stats.
- **Command line:** `nba-similar "Nikola Jokic" 2023` and
  `nba-career "Victor Wembanyama" --through-age 21` print the same results in a terminal.

## Methodology

The pipeline has four stages, each a package under `src/nba_similarity/`, and each one was
verified before the next was built:

```
Basketball-Reference ─► ingestion/ ─► features/ ─► similarity/ ─► ui/ (Streamlit)
  (rate-limited,         SQLite raw    era-adjusted  scores and      Search + Map pages
   cached HTML)          tables        z-scores      explanations    (mapping/: UMAP layout)
```

### Data
- **Source:** Basketball-Reference's league-wide *per-100-possession* and *advanced* stat
  pages, one pair per season.
- **Why per 100 possessions:** per-game stats mostly measure pace and minutes. A 1980s team
  played about 10 more possessions a game than a 2000s team. Per-100 and rate stats (AST%,
  USG%, ...) compare what a player did with the possessions he was on the floor for.
- **Why it starts in 1973-74:** that's the first season with per-100 data, and steals,
  blocks and turnovers start around then too.
- **Traded players:** only Basketball-Reference's combined full-season row is used.
  - Similarity compares whole seasons.
  - Per-team rows would list the same player twice in results.
  - The advanced stats are computed over the full season on that row.
- **Scraping etiquette:**
  - at most one request every 4 seconds;
  - a descriptive User-Agent;
  - every page is cached, so a season is fetched once (a season in progress is refreshed).

### Features: 12 rate stats in 5 categories

| Category | Stats |
|---|---|
| Scoring | points per 100*, usage %, 3PA per 100, free-throw rate |
| Efficiency | true shooting %, free-throw % |
| Playmaking | assist %, turnover % |
| Rebounding | offensive rebound %, defensive rebound % |
| Defense | steal %, block % |

\* Shown in explanations, but not used in the distance (see below).

Choices worth explaining:
- **Rates, not totals:** similarity should describe *how* someone played, not how long.
  Minutes and games only decide who qualifies.
- **Composites excluded** (PER, BPM, Win Shares, VORP): they are weighted blends of the stats
  already here. Including them would count the same things twice and hide *why* two seasons
  match.
- **3P% and FG% excluded:** 3P% is noisy for low-volume shooters, and FG% overlaps true
  shooting %, which also credits threes and free throws.
- **Offensive and defensive rating excluded:** they depend heavily on the team.
- **3-point volume, not 3-point share.** The first version used 3PA rate (threes per field
  goal attempt). That put Stephen Curry's 2015-16 only +1.1 SD above his league, because
  spot-up role players take a bigger *share* of threes. 3PA per 100 possessions measures
  volume, and the archetype comps improved.
- **Points per 100 is display-only.** It is roughly usage × efficiency, so including it
  double-counted USG% and TS%. That double-counting was what pushed John Stockton out of
  Chris Paul's comps. It stays in the explanations. 11 stats drive the distance.

### Normalization: making eras comparable
- **Z-scores within each season.** Each stat is standardized against that season's qualified
  players: mean 0, SD 1. A 3PA rate that was extreme in 1990 is ordinary today, so raw values
  can't be compared across eras. Percentiles within the season can.
- **Qualified players only:** at least 500 minutes, scaled down for shortened seasons
  (1998-99: 305; 2011-12: 402; 2019-20: 433; 2020-21: 439). Without that, garbage-time
  players would distort every season's mean and SD.
- **Square-root transform** for right-skewed stats (3PA per 100, FT rate, ORB%, BLK%) before
  z-scoring. Without it, the most extreme seasons pile up at the clip and stop being
  distinguishable.
- **FT% shrinkage:** each player's FT% is blended with the league FT% as if he had taken 50
  extra league-average attempts. A center who went 9-for-10 shouldn't look like an elite
  shooter.
- **Clipped at ±4 SD** so one freak value can't dominate a comparison.
- **Missing values are never filled in.**
  - The 3-point line arrived in 1979-80; usage and turnover % start in 1977-78.
  - A stat only counts for a season if at least 90% of qualified players have it. That drops
    1973-74 turnovers, which only Seattle tracked.
  - Raw values are stored next to the z-scores, and explanations always show the observed
    stats.

### Similarity
- **Distance:** a weighted euclidean (or optionally cosine) distance on the z-scores. scikit-learn
  provides optional standardization and PCA.
- **Equal weight per stat by default.** The category sliders multiply their stats' weights.
  An earlier version split each category's weight over its stats. That gave each stat in a
  2-stat category twice the weight of a scoring stat, so one extreme stat took over: Chris
  Paul's top comp became Eric Murdock, purely on steal %.
- **Score (0–100):** `100 × (1 − D² / E[D²])`.
  - E[D²] is the expected distance from *this query* to a random season.
  - 0 means "no closer than a typical season".
  - Because the baseline depends on the query, an unusual player isn't penalised for being
    unusual, and matching his distinctive traits earns the most.
- **Missing stats count as "typical", not as a match.**
  - If the query has a stat and the candidate doesn't, it scores 0 points but stays in the
    total.
  - Otherwise pre-1980 seasons gained by *skipping* the query's distinctive stats: Walter
    Davis 1978-79, who has no 3-point data, was briefly Curry's #2 comp.
  - Candidates must also share at least 70% of the stat weight.
- **Exact explanations:** every score is a sum of per-stat contributions, also under PCA.
  The UI shows two lists:
  - **alike:** high-contributing stats within 1 SD;
  - **differ:** shared stats more than 1 SD apart.

  A stat can contribute points while still being a difference, so raw contribution rank
  alone would mislead.
- **One season per comp player,** so the list isn't five Tim Duncan seasons.

### The map
- **Same distance as the engine:** UMAP runs with a custom numba distance that is exactly the
  engine's (equal weights, missing stats counted as typical), not on imputed vectors. Seasons
  that are close on the map are close in the search too.
- **Precomputed** by `nba-map` and stored in SQLite. The app warns when the features have
  changed since the map was built.
- **Clusters:** k-means on the 2D layout, so they match the groups you see. Each is named by
  its most distinctive average stats, such as "high USG%, high AST%".
- **Fixed weights:** positions use the default weights. The highlighted comps follow your
  sliders, and the page says so.

### Career trajectories
- **Aligned by age, not years in the league.** Development tracks age: a 21-year-old in his
  second season and a 21-year-old rookie out of college are both compared with other
  21-year-olds. Ages are Basketball-Reference's age on February 1, so no birthdates are
  needed. Years in the league are shown, and can be capped.
- **The window keeps the trajectory.** Each age is its own slot of stats, compared with the
  same age, with recent ages weighted more (1, 0.6, 0.36). Averaging would make a rising and
  a falling career with the same mean look identical.
- **Playing time counts.** Minutes per game is a trajectory-only stat: a 30-minute starter
  at 20 and a 12-minute reserve are on different paths.
- **Young players' short seasons count, at a discount.** Seasons of 250–499 minutes are
  scored against the qualified players' statistics, but their points are scaled by
  MP / 500. A season the player didn't play counts as "typical" (no credit, still in the
  total). Our data can't tell an injury from a G League or overseas year, and says so.
- **Truncated careers are excluded, not guessed.** Careers first seen in 1973-74 may have
  started earlier, and ABA players' early years aren't on NBA pages. Both are flagged (the
  ABA rosters are fetched once, for ids only) and not compared over the missing ages.
  Earlier seasons can't be backfilled: possessions, steals and blocks weren't tracked
  before 1973-74.
- **No survivor bias.** The projection uses only comps whose next seasons have all been
  played, and a comp who left the league counts as such. Stats are summarized only over
  comps who played a full season, as within-season percentiles and in the latest season's
  terms. Comps whose future is still unfolding are shown as peers.

### Validation
`scripts/validate_archetypes.py` checks five well-known archetypes against "signature traits"
written down beforehand: Curry 2016, Shaq 2000, Gobert 2022, Chris Paul 2008 and Giannis
2020. For each, it reports how well the comps fit and which categories keep the comps apart.
- It caught the weighting problem (the Murdock example).
- It caught the 3PA-rate problem (the Curry example).
- It caught the points double-count (the missing Stockton).
- The obvious check also passed at the Phase 3 gate: with own seasons included, Jokić
  2021-22 ranked #2 for Jokić 2022-23 (score 94.8 then).
- **The final round:** a mean archetype fit of 92%, with no comps flagged. Curry's list holds
  Ray Allen, Mark Price and Larry Bird, and Gobert's holds Mutombo, Przybilla and Chandler.
  One known residual: Eric Murdock still tops Chris Paul's list, on near-identical steal
  rates.

Design notes and findings for each phase are in [`docs/`](docs/).

## Running it locally

Requires Python 3.11+.

```bash
git clone https://github.com/will-j-cramer/nba-player-comparison.git
cd nba-player-comparison
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # installs the nba-* commands
```

Build the data. The database is not included; see [Data and license](#data-and-license).

```bash
nba-ingest      # scrape 1973-74 → present into data/nba.sqlite
nba-features    # era-adjusted feature tables and a sanity summary
nba-map         # precompute the 2D map (1–2 min; the first run compiles numba code)
```

`nba-ingest` fetches 2 pages per season plus 9 ABA roster pages (used only to flag careers that
began in the ABA), one request every 4 seconds: about 115 pages, roughly 8 minutes. It caches every page under `data/raw/` and is resumable: stop it any time and
re-run it.

Use it:

```bash
nba-app                                   # the Streamlit app at http://localhost:8501
nba-similar "Nikola Jokic" 2023           # top 10 comps in the terminal
nba-similar "Stephen Curry" 2016 --weight scoring=2 --metric cosine
nba-career "Victor Wembanyama" --through-age 21    # career comps and what came next
python scripts/validate_archetypes.py     # the archetype check
```

Seasons are written as their end year (2023 = 2022-23). Every command accepts `--db`, or
reads `$NBA_DB`.

Development:

```bash
python -m pytest      # 261 tests; no network access needed
ruff check . && ruff format --check .
```

Parser tests use trimmed copies of real pages. They aren't committed, so build them with
`scripts/make_fixtures.sh` after an ingest; those tests are skipped until then.

> **Conda users:** conda's base environment has its own `streamlit` and `pytest`, which can
> shadow the virtualenv's. The symptoms are `ModuleNotFoundError: nba_similarity`, a missing
> `umap`, or pytest stopping with "This interpreter … can't run the project". Check with
> `which -a pytest streamlit python`. The project's `.venv/bin` should come first. If it
> doesn't, run `conda deactivate` before `source .venv/bin/activate`, or turn off conda's
> auto-activation with `conda config --set auto_activate_base false`. Then use `nba-app` and
> `python -m pytest` rather than bare `streamlit` or `pytest`.

## Deploying to Streamlit Community Cloud

The scraped data must not be published, so the database stays out of this public repository.
Instead, the deployed app downloads a slim copy from a private location the first time it
starts. The copy holds only player names, the feature tables and the map, with no raw scraped
tables.

**1. Export the app database** (after `nba-features` and `nba-map`):

```bash
python scripts/export_app_db.py           # → data/processed/nba_app.sqlite (gitignored)
```

**2. Upload it privately.** One option is a release asset in a private GitHub repository,
here with the [GitHub CLI](https://cli.github.com/):

```bash
gh repo create nba-similarity-data --private
gh release create data-v1 data/processed/nba_app.sqlite \
  --repo <you>/nba-similarity-data --title "App database" --notes "Private; do not redistribute."
gh api repos/<you>/nba-similarity-data/releases/tags/data-v1 --jq '.assets[0].url'
```

The last command prints the asset's API URL, like
`https://api.github.com/repos/<you>/nba-similarity-data/releases/assets/123456789`.

**3. Create a read-only token.** On GitHub, go to Settings → Developer settings →
Fine-grained tokens:
- Repository access: only `nba-similarity-data`.
- Permissions: Contents, read-only.

Note the expiry date, because the app stops downloading after it.

**4. Create the app** at [share.streamlit.io](https://share.streamlit.io): Create app →
Deploy a public app from GitHub, then:
- **Repository and branch:** your copy of this repo.
- **Main file path:** `streamlit_app.py`.
- **Advanced settings → Python version:** 3.12.
- **Advanced settings → Secrets:** paste the following (see
  [`.streamlit/secrets.example.toml`](.streamlit/secrets.example.toml)):

  ```toml
  NBA_DB_URL = "https://api.github.com/repos/<you>/nba-similarity-data/releases/assets/123456789"
  NBA_DB_TOKEN = "github_pat_..."
  ```

Deploy. The first visit downloads the database (several MB) once per container; later visits
use the copy on disk.

**Updating the data:**
1. Re-run the pipeline and the export.
2. Upload the file as a new release (or a new asset), so it gets a new URL.
3. Change `NBA_DB_URL` in the app's Secrets.

The downloaded file is named after the URL, so the new URL makes the app download again.
Reboot the app from its menu if it doesn't pick up the change.

**Notes:**
- Any direct download link works as `NBA_DB_URL`. Leave out `NBA_DB_TOKEN` if the link needs
  no authentication.
- `streamlit_app.py` makes the `src/` package importable without installing it. Cloud installs
  `requirements.txt`, which includes the pipeline and development packages; they cost a little
  install time but aren't loaded by the app.
- Troubleshooting:
  - **"not a SQLite database":** the URL served an error page. Check the asset URL, and use
    the `api.github.com/.../releases/assets/<id>` form, not the browser download link.
  - **HTTP 404:** usually an expired token, or one without access to the data repository.
  - **"map … out of date":** run `nba-map` before exporting; the export refuses a stale map.

## Project structure

```
src/nba_similarity/
  ingestion/   rate-limited client, HTML cache, parser, SQLite schema/store, nba-ingest
  features/    feature definitions, qualification, transforms, per-season z-scores, nba-features
  similarity/  feature matrix, scoring, engine with exact explanations, nba-similar
  mapping/     UMAP with the engine's distance, map storage, k-means clusters, nba-map
  careers/     career windows by age, career comps, outcomes and projection, nba-career
  ui/          Streamlit app: entry point, shared sidebar, Search/Map/Careers pages, pure view +
               chart builders, cached loaders, database download
  paths.py     default database location ($NBA_DB)
scripts/       validate_archetypes.py, export_app_db.py, check_ingestion.py, fixture tools
tests/         pytest suite mirroring src/
docs/          per-phase design notes and verification
streamlit_app.py   Streamlit Community Cloud entry point
```

The app only reads SQLite. It never scrapes, recomputes features or builds the map. The
modules that build tables and figures (`ui/view.py`, `ui/charts.py`) are pure functions with
no Streamlit imports, so they are unit-tested directly. The pages are tested headlessly with
Streamlit's `AppTest`.

## Limitations

- **Box-score stats only.** No tracking or play-type data, so defense is reduced to steals,
  blocks and defensive rebounds. A great positional defender with few steals looks average.
- **Early seasons have gaps.** Seasons before 1979-80 lack 3-point stats, and those before
  1977-78 also lack usage and turnover %. They are still compared fairly (missing means
  typical), but on fewer stats, which the app shows.
- **Similarity is not value.** Two seasons can have the same *shape* at very different
  levels of impact. Use the score as "plays like", not "as good as".
- **The map uses fixed default weights**, because it is precomputed.
- **Career projections are rough.** They summarize 10–20 similar careers, not a model of
  development, and ages are whole years (a player born in January and one born in December
  share an "age").

## Data and license

The statistics come from [Basketball-Reference](https://www.basketball-reference.com/)
(Sports Reference LLC). This repository contains no scraped data. The database, cached pages
and test fixtures are built locally and gitignored. If you run the scraper, follow Sports
Reference's terms of use and don't redistribute the data.

The code is published for viewing as a portfolio project. No license is granted, so all
rights are reserved.
