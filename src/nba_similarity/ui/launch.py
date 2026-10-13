"""`nba-app`: start the Streamlit app with the Python that has this package installed.

A bare `streamlit` command can belong to another environment (e.g. conda's base, which ships
its own Streamlit) and then fails with "No module named 'nba_similarity'". Running
`<this python> -m streamlit run app.py` always uses the right one.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

APP = Path(__file__).with_name("app.py")


def command(argv: Sequence[str] = ()) -> list[str]:
    """The streamlit command line; extra arguments (e.g. --server.port 8502) pass through."""
    return [sys.executable, "-m", "streamlit", "run", str(APP), *argv]


def main(argv: Sequence[str] | None = None) -> int:
    return subprocess.call(command(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
