"""Fetches the app database on first start when it isn't on disk (Streamlit Community Cloud).

The database is never committed (the repository is public and scraped data must not be
redistributed). For a deployment, `scripts/export_app_db.py` makes a slim copy holding only the
tables the app reads; it is uploaded somewhere private and its URL (plus an optional token)
goes in the app's secrets. Works with GitHub's release-asset API, which needs the
`Accept: application/octet-stream` header, and with any direct download link.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

SQLITE_HEADER = b"SQLite format 3\x00"
TIMEOUT_S = 60.0
CHUNK_BYTES = 1 << 20


class DownloadError(Exception):
    """The database could not be downloaded or is not a SQLite file."""


def download_path(db_path: Path, url: str) -> Path:
    """Where a download from `url` is kept: next to `db_path`, named after a hash of the URL,
    so pointing the secret at a new file triggers a fresh download."""
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    return db_path.with_name(f"{db_path.stem}-{digest}{db_path.suffix}")


def ensure_db(
    path: Path,
    url: str | None,
    token: str | None = None,
    fetch: Callable[..., Any] | None = None,
) -> Path | None:
    """Return `path`, downloading it from `url` first if it doesn't exist.

    None when the file is missing and no URL is configured. The download goes to a temporary
    file that is checked for the SQLite header and only then renamed into place, so a failed
    or partial download never leaves a broken database behind. `fetch` defaults to
    requests.get (tests pass a fake).
    """
    if path.exists():
        return path
    if not url:
        return None
    fetch = fetch or requests.get
    headers = {"Accept": "application/octet-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    part = path.with_name(path.name + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with fetch(url, headers=headers, stream=True, timeout=TIMEOUT_S) as response:
            if response.status_code != 200:
                raise DownloadError(f"download failed: HTTP {response.status_code}")
            with part.open("wb") as out:
                for chunk in response.iter_content(CHUNK_BYTES):
                    out.write(chunk)
        with part.open("rb") as f:
            if f.read(len(SQLITE_HEADER)) != SQLITE_HEADER:
                raise DownloadError("downloaded file is not a SQLite database (check the URL)")
        part.replace(path)
    except requests.RequestException as exc:
        raise DownloadError(f"download failed: {exc}") from exc
    finally:
        part.unlink(missing_ok=True)
    return path
