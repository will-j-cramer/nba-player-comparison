"""Trim a saved Basketball-Reference season page down to its main stats table.

Used to build the test fixtures in tests/fixtures/ from pages cached in data/raw/:

    python scripts/trim_fixture.py data/raw/NBA_2023_advanced.html advanced \
        tests/fixtures/NBA_2023_advanced.html.gz [--max-rows N]

A destination ending in .gz is written gzip-compressed.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

from bs4 import BeautifulSoup, SoupStrainer

# Attributes the parser reads; everything else (sort keys, tooltips, links) is dropped.
KEEP_ATTRS = {"id", "data-stat", "data-append-csv"}


def trim(html: str, table_id: str, max_rows: int | None = None) -> str:
    """Keep only the stats table `table_id` (optionally its first `max_rows` rows), with
    only the attributes the parser reads, so test fixtures stay small.
    """
    soup = BeautifulSoup(html, "lxml", parse_only=SoupStrainer("table", id=table_id))
    table = soup.find("table", id=table_id)
    if table is None:
        raise SystemExit(f"no <table id={table_id!r}> found")
    if max_rows is not None:
        rows = table.tbody.find_all("tr")
        keep = rows[:max_rows] + [r for r in rows[max_rows:] if "norank" in (r.get("class") or [])]
        for row in rows:
            if row not in keep:
                row.decompose()
    for link in table.find_all("a"):
        link.unwrap()
    for tag in [table, *table.find_all(True)]:
        keep = KEEP_ATTRS | {"class"} if tag.name == "tr" else KEEP_ATTRS
        tag.attrs = {k: v for k, v in tag.attrs.items() if k in keep}
    return f"<html><body>\n{table}\n</body></html>\n"


def main() -> None:
    """Trim a cached page into a test fixture; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path)
    parser.add_argument("table_id", choices=["per_poss", "advanced"])
    parser.add_argument("dest", type=Path)
    parser.add_argument("--max-rows", type=int, help="keep only the first N rows (+ league avg)")
    args = parser.parse_args()
    html = args.source.read_text(encoding="utf-8")
    trimmed = trim(html, args.table_id, args.max_rows).encode("utf-8")
    if args.dest.suffix == ".gz":
        args.dest.write_bytes(gzip.compress(trimmed, mtime=0))
    else:
        args.dest.write_bytes(trimmed)


if __name__ == "__main__":
    main()
