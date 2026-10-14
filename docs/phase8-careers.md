# Phase 8 — Career trajectory comps

## Status
- **Implemented:**
  - ABA roster ingestion (`ingestion/aba.py`, run by `nba-ingest`, `--skip-aba` to opt out);
  - the career tables in `nba-features` (`features/career.py`);
  - the `careers/` package (`data`, `windows`, `engine`, `outcomes`, `format`, `lookup`, `cli`)
    and the `nba-career` command;
  - a Careers page in the app;
  - the career tables are included in the deployment export.
- **Tests:** 28 new (261 in total). The career engine is tested on a synthetic league with
  planted careers:
  - the twin ranks first (score 99.9);
  - the opposite trajectory ranks far lower;
  - a 300-minute season beats no season, and both lose to a full one;
  - censored careers are excluded;
  - the bust counts as "out" in the projection;
  - the young comp's outcome stays pending.
- **Checked here:**
  - On the local database, the qualified rows' z-scores equal the season comps' exactly,
    and the stored map is not made stale.
  - The Careers page rendered (Playwright) on the synthetic league.
  - **Not checked here:** real-data results. This environment has only 1973-74 and 2022-23,
    and B-Ref blocks it.
- **Pending (owner, full database):**
  1. `git pull && pip install -e .`
  2. `nba-ingest` (fetches the 9 ABA pages once)
  3. `nba-features`
  4. then the sanity checks below.

## Decisions
These are the owner's choices from the design review (2026-10-14):
- **Window:** age slots weighted toward recent seasons.
- **Playing time:** minutes per game added as a trajectory-only stat.
- **ABA:** the 9 ABA roster pages are fetched.
- **Minimum minutes:** 250 for trajectories.

Implementation details that differ from the proposal:
- **Score baseline:** the expected distance to a random season *at the same age*, from every
  scored season at that age. Using the candidates' own spread made scores unstable when few
  players shared the window ages.
- **Partial seasons (250–499 min):** these scale the slot's points by `min(1, MP / minimum)`,
  instead of re-weighting the slot. A low-minute season is then a blend of its observed
  comparison and "typical". It neither earns full credit nor escapes a full penalty, and a
  missing season is the limit (0 points).
- **Observable outcomes:** a comp counts only if every future season is in the ingested,
  final seasons. A gap in the ingested data never reads as "left the league".

## Phase gate (owner): sanity checks on the full database
- `nba-career "LeBron James" --through-age 21`: young high-usage wings and creators.
- `nba-career "Tim Duncan" --through-age 23`: productive young bigs.
- `nba-career "Victor Wembanyama" --through-age 21`: young rim protectors who also shoot
  from distance; projection from comps with known outcomes; peers such as Chet Holmgren
  shown as pending.
- `nba-career "Moses Malone" --through-age 23`: the window starts at his first NBA season,
  1976-77. His ABA years aren't in our data, so they're neither compared nor counted as
  missing. As a *candidate*, he is excluded from windows that reach back into his ABA ages.
- Projections for low-scoring comps show a non-zero "not in the NBA" share.
