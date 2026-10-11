from __future__ import annotations

import pytest
import requests

from nba_similarity.ingestion.client import (
    BACKOFF_BASE_S,
    BlockedError,
    FetchError,
    PageNotFoundError,
    PoliteClient,
    RateLimitedError,
)

URL = "https://example.test/page.html"


class FakeResponse:
    """Mimics requests.Response: `text` falls back to ISO-8859-1 when no charset is declared."""

    def __init__(self, status: int, body: str = "", headers: dict[str, str] | None = None):
        self.status_code = status
        self.headers = headers or {}
        self.content = body.encode("utf-8")
        declared = "charset=" in self.headers.get("Content-Type", "").lower()
        self.text = body if declared else self.content.decode("iso-8859-1")


class FakeSession:
    """Returns (or raises) the scripted outcomes in order and records request times."""

    def __init__(self, outcomes: list[FakeResponse | Exception], clock: FakeClock):
        self.outcomes = list(outcomes)
        self.headers: dict[str, str] = {}
        self.clock = clock
        self.request_times: list[float] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.request_times.append(self.clock.now)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(
    outcomes: list[FakeResponse | Exception], delay_s: float = 4.0
) -> tuple[PoliteClient, FakeSession, FakeClock]:
    clock = FakeClock()
    session = FakeSession(outcomes, clock)
    client = PoliteClient(
        delay_s,
        session=session,  # type: ignore[arg-type]
        sleep=clock.sleep,
        clock=clock,
        jitter=lambda: 0.0,
    )
    return client, session, clock


def test_returns_body_and_sets_user_agent() -> None:
    client, session, _ = make_client([FakeResponse(200, "<html>ok</html>")])
    assert client.get(URL) == "<html>ok</html>"
    assert "nba-similarity" in session.headers["User-Agent"]


def test_requests_are_spaced_by_delay() -> None:
    client, session, _ = make_client([FakeResponse(200)] * 3, delay_s=4.0)
    for _ in range(3):
        client.get(URL)
    gaps = [b - a for a, b in zip(session.request_times, session.request_times[1:], strict=False)]
    assert gaps == [4.0, 4.0]


def test_retries_5xx_with_exponential_backoff() -> None:
    client, _, clock = make_client([FakeResponse(503), FakeResponse(502), FakeResponse(200, "ok")])
    assert client.get(URL) == "ok"
    backoffs = [s for s in clock.sleeps if s >= BACKOFF_BASE_S]
    assert backoffs == [BACKOFF_BASE_S, BACKOFF_BASE_S * 2]


def test_retries_connection_errors() -> None:
    client, _, _ = make_client([requests.ConnectionError("reset"), FakeResponse(200, "ok")])
    assert client.get(URL) == "ok"


def test_gives_up_after_max_attempts() -> None:
    client, session, _ = make_client([FakeResponse(503)] * 5)
    with pytest.raises(FetchError, match="HTTP 503"):
        client.get(URL)
    assert len(session.request_times) == 5


def test_429_waits_for_retry_after() -> None:
    client, _, clock = make_client(
        [FakeResponse(429, headers={"Retry-After": "60"}), FakeResponse(200, "ok")]
    )
    assert client.get(URL) == "ok"
    assert 60.0 in clock.sleeps


def test_429_with_long_retry_after_aborts() -> None:
    client, session, _ = make_client([FakeResponse(429, headers={"Retry-After": "3600"})])
    with pytest.raises(RateLimitedError):
        client.get(URL)
    assert len(session.request_times) == 1


def test_404_raises_page_not_found_without_retry() -> None:
    client, session, _ = make_client([FakeResponse(404)])
    with pytest.raises(PageNotFoundError):
        client.get(URL)
    assert len(session.request_times) == 1


def test_other_4xx_fails_without_retry() -> None:
    client, session, _ = make_client([FakeResponse(403)])
    with pytest.raises(FetchError, match="HTTP 403"):
        client.get(URL)
    assert len(session.request_times) == 1


@pytest.mark.parametrize("status", [403, 503])
def test_cloudflare_challenge_raises_blocked_without_retry(status: int) -> None:
    challenge = FakeResponse(status, "<html><head><title>Just a moment...</title></head></html>")
    client, session, _ = make_client([challenge])
    with pytest.raises(BlockedError, match="Cloudflare"):
        client.get(URL)
    assert len(session.request_times) == 1


def test_cf_mitigated_header_raises_blocked() -> None:
    client, _, _ = make_client([FakeResponse(200, "", headers={"cf-mitigated": "challenge"})])
    with pytest.raises(BlockedError):
        client.get(URL)


def test_decodes_utf8_when_server_sends_no_charset() -> None:
    page = FakeResponse(200, "<td>Nikola Jokić</td>", headers={"Content-Type": "text/html"})
    client, _, _ = make_client([page])
    assert client.get(URL) == "<td>Nikola Jokić</td>"


def test_uses_declared_charset() -> None:
    page = FakeResponse(200, "Dončić", headers={"Content-Type": "text/html; charset=UTF-8"})
    client, _, _ = make_client([page])
    assert client.get(URL) == "Dončić"
