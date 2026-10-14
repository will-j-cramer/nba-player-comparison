"""On-disk cache of raw Basketball-Reference HTML, so no page is downloaded twice."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

BASE_URL = "https://www.basketball-reference.com/leagues"

Page = Literal["per_poss", "advanced"]


class Fetcher(Protocol):
    """Anything that can GET a page's HTML (PoliteClient, or a fake in tests)."""

    def get(self, url: str) -> str:
        """Return the body of `url`."""
        ...


def page_url(season: int, page: Page) -> str:
    """URL of a league season page; `season` is the end year (2023 = 2022-23)."""
    return f"{BASE_URL}/NBA_{season}_{page}.html"


def cache_path(raw_dir: Path, season: int, page: Page) -> Path:
    """Where a season page's HTML is cached under `raw_dir`."""
    return raw_dir / f"NBA_{season}_{page}.html"


def fetch_page(
    season: int, page: Page, client: Fetcher, raw_dir: Path, *, refresh: bool = False
) -> str:
    """Return the page HTML from the cache, downloading and caching it if missing or `refresh`."""
    return fetch_cached(page_url(season, page), cache_path(raw_dir, season, page), client,
                        refresh=refresh)  # fmt: skip


def fetch_cached(url: str, path: Path, client: Fetcher, *, refresh: bool = False) -> str:
    """Return `path`'s HTML if cached; otherwise download `url` and cache it atomically."""
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    html = client.get(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(path)
    return html
