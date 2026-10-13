"""Default locations of the project's data files."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DB = Path("data/nba.sqlite")


def default_db() -> Path:
    """The SQLite database: `$NBA_DB` if set, else data/nba.sqlite (relative to the cwd)."""
    return Path(os.environ.get("NBA_DB") or DEFAULT_DB)
