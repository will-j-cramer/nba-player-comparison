"""Streamlit app: `nba-app` (or `python -m streamlit run src/nba_similarity/ui/app.py`, or
`streamlit_app.py` at the repository root, which Streamlit Community Cloud runs).

Entry point: draws the shared query sidebar, then runs the selected page (search_page.py or
map_page.py). Layout and caching only: comparisons come from the similarity engine, tables
and series from ui/view.py, figures from ui/charts.py, cached loaders from ui/cache.py. The
app reads SQLite (database: $NBA_DB) and never scrapes or recomputes features.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from nba_similarity.ui import sidebar

HERE = Path(__file__).parent


def main() -> None:
    """Configure the page, draw the shared sidebar and run the selected page."""
    st.set_page_config(page_title="NBA Player Similarity", page_icon="🏀", layout="wide")
    page = st.navigation(
        [
            st.Page(HERE / "search_page.py", title="Search", icon="🔎", default=True),
            st.Page(HERE / "map_page.py", title="Map", icon="🗺️"),
        ],
        position="top",
    )
    sidebar.render()
    page.run()


if __name__ == "__main__":
    main()
