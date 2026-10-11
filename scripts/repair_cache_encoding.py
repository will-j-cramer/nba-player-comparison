"""Repair cached pages whose UTF-8 text was decoded as ISO-8859-1 ("Jokić" -> "JokiÄ\x87").

Pages fetched before the client defaulted to UTF-8 were garbled this way. The damage is exactly
reversible, so this fixes data/raw/ in place without downloading anything. Files that are
already correct are left untouched. Afterwards, rebuild the database from the cache:

    python scripts/repair_cache_encoding.py
    rm data/nba.sqlite && nba-ingest
"""

from __future__ import annotations

import argparse
from pathlib import Path


def repair(text: str) -> str | None:
    """Return the repaired text, or None if `text` is not garbled this way."""
    try:
        fixed = text.encode("iso-8859-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None  # contains characters a garbled page cannot have: already correct
    return fixed if fixed != text else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    repaired = 0
    pages = sorted(args.raw_dir.glob("NBA_*.html"))
    for path in pages:
        fixed = repair(path.read_text(encoding="utf-8"))
        if fixed is not None:
            path.write_text(fixed, encoding="utf-8")
            repaired += 1
    print(f"repaired {repaired} of {len(pages)} cached pages in {args.raw_dir}")


if __name__ == "__main__":
    main()
