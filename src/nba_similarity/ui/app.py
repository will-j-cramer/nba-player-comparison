"""Streamlit app: `nba-app` (or `python -m streamlit run src/nba_similarity/ui/app.py`, or
`streamlit_app.py` at the repository root, which Streamlit Community Cloud runs).

Entry point: draws the shared query sidebar, then runs the selected page (search_page.py,
map_page.py or career_page.py). Layout and caching only: comparisons come from the engines,
tables and series from ui/view.py, figures from ui/charts.py, cached loaders from
ui/cache.py. The app reads SQLite (database: $NBA_DB) and never scrapes or recomputes
features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

from nba_similarity.environment import MIN_STREAMLIT, streamlit_too_old
from nba_similarity.ui import sidebar

HERE = Path(__file__).parent


def main() -> None:
    """Configure the page, draw the shared sidebar and run the selected page."""
    st.set_page_config(page_title="NBA Player Similarity", page_icon="🏀", layout="wide")
    if streamlit_too_old(st.__version__):
        minimum = ".".join(map(str, MIN_STREAMLIT))
        st.error(
            f"This app needs Streamlit {minimum} or newer, but {sys.executable} has "
            f"{st.__version__}. Run it with `nba-app` from the project's virtualenv."
        )
        st.stop()
    page = st.navigation(
        [
            st.Page(HERE / "search_page.py", title="Search", icon="🔎", default=True),
            st.Page(HERE / "map_page.py", title="Map", icon="🗺️"),
            st.Page(HERE / "career_page.py", title="Careers", icon="📈"),
        ],
        position="top",
    )
    sidebar.render()
    page.run()


if __name__ == "__main__":
    main()
