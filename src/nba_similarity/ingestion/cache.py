"""On-disk cache of raw Basketball-Reference HTML, so no page is downloaded twice."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

BASE_URL = "https://www.basketball-reference.com/leagues"

Page = Literal["per_poss", "advanced"]


class Fetcher(Protocol):
    def get(self, url: str) -> str: ...


def page_url(season: int, page: Page) -> str:
    """URL of a league season page; `season` is the end year (2023 = 2022-23)."""
    return f"{BASE_URL}/NBA_{season}_{page}.html"


def cache_path(raw_dir: Path, season: int, page: Page) -> Path:
    return raw_dir / f"NBA_{season}_{page}.html"


def fetch_page(
    season: int, page: Page, client: Fetcher, raw_dir: Path, *, refresh: bool = False
) -> str:
    """Return the page HTML from the cache, downloading and caching it if missing or `refresh`."""
    path = cache_path(raw_dir, season, page)
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    html = client.get(page_url(season, page))
    raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(path)
    return html
