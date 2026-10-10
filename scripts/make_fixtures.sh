#!/usr/bin/env bash
# Build the parser test fixtures in tests/fixtures/ from cached Basketball-Reference pages.
# Fixtures are gitignored: scraped data is not redistributed, so each checkout builds its own.
# Downloads (politely, via nba-ingest) only the pages not already in data/raw/.
set -euo pipefail
cd "$(dirname "$0")/.."

for season in 1974 2023; do
    if [[ ! -f data/raw/NBA_${season}_per_poss.html || ! -f data/raw/NBA_${season}_advanced.html ]]; then
        python -m nba_similarity.ingestion --start "$season" --end "$season"
    fi
done

python scripts/trim_fixture.py data/raw/NBA_2023_per_poss.html per_poss tests/fixtures/NBA_2023_per_poss.html.gz
python scripts/trim_fixture.py data/raw/NBA_2023_advanced.html advanced tests/fixtures/NBA_2023_advanced.html.gz
python scripts/trim_fixture.py data/raw/NBA_1974_advanced.html advanced tests/fixtures/NBA_1974_advanced.html --max-rows 12
echo "fixtures written to tests/fixtures/"
