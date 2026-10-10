from __future__ import annotations

import gzip
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent.parent / "fixtures"


def read_fixture(name: str) -> str:
    """Read a saved (trimmed) Basketball-Reference page; `.gz` fixtures are decompressed.

    Fixtures are generated locally by scripts/make_fixtures.sh and not committed (scraped
    data is not redistributed), so tests that need them skip when they are missing.
    """
    path = FIXTURES / name
    if not path.exists():
        pytest.skip(f"fixture {name} missing; run scripts/make_fixtures.sh")
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode("utf-8")
    return path.read_text(encoding="utf-8")
