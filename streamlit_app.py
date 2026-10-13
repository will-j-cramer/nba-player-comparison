"""Entry point for Streamlit Community Cloud (its default main file).

Makes the `src/` package importable without installing it, then runs the app. Locally, use
`nba-app` instead.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nba_similarity.ui.app import main  # noqa: E402

main()
