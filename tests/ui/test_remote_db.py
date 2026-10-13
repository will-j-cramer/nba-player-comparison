from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import requests

from nba_similarity.ui.remote_db import SQLITE_HEADER, DownloadError, download_path, ensure_db


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body, self.status_code = body, status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def iter_content(self, size: int) -> list[bytes]:
        return [self.body[i : i + size] for i in range(0, len(self.body), size)] or [b""]


class FakeFetch:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


DB_BYTES = SQLITE_HEADER + b"\x00" * 100


def test_existing_file_is_used_without_fetching(tmp_path: Path) -> None:
    path = tmp_path / "nba.sqlite"
    path.write_bytes(DB_BYTES)
    fetch = FakeFetch(FakeResponse(b""))
    assert ensure_db(path, "https://example.test/db", fetch=fetch) == path
    assert fetch.calls == []


def test_no_file_and_no_url_returns_none(tmp_path: Path) -> None:
    assert ensure_db(tmp_path / "nba.sqlite", None) is None


def test_downloads_with_token_and_octet_stream_headers(tmp_path: Path) -> None:
    path = tmp_path / "data" / "nba.sqlite"
    fetch = FakeFetch(FakeResponse(DB_BYTES))
    assert ensure_db(path, "https://example.test/db", "secret", fetch=fetch) == path
    assert path.read_bytes() == DB_BYTES
    url, kwargs = fetch.calls[0]
    assert kwargs["headers"] == {
        "Accept": "application/octet-stream",
        "Authorization": "Bearer secret",
    }
    assert kwargs["stream"] is True
    assert not path.with_name("nba.sqlite.part").exists()


def test_no_token_sends_no_authorization(tmp_path: Path) -> None:
    fetch = FakeFetch(FakeResponse(DB_BYTES))
    ensure_db(tmp_path / "nba.sqlite", "https://example.test/db", fetch=fetch)
    assert "Authorization" not in fetch.calls[0][1]["headers"]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (FakeResponse(b"<html>Not Found</html>"), "not a SQLite database"),
        (FakeResponse(b"", status=404), "HTTP 404"),
        (requests.ConnectionError("offline"), "offline"),
    ],
)
def test_failed_download_leaves_nothing_behind(
    tmp_path: Path, response: FakeResponse | Exception, message: str
) -> None:
    path = tmp_path / "nba.sqlite"
    with pytest.raises(DownloadError, match=message):
        ensure_db(path, "https://example.test/db", fetch=FakeFetch(response))
    assert list(tmp_path.iterdir()) == []


def test_download_path_changes_with_the_url(tmp_path: Path) -> None:
    db = tmp_path / "nba.sqlite"
    a, b = download_path(db, "https://x.test/1"), download_path(db, "https://x.test/2")
    assert a != b and a.parent == db.parent and a.suffix == ".sqlite"
    assert download_path(db, "https://x.test/1") == a
