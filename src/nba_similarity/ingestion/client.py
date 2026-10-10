"""Rate-limited HTTP client for Basketball-Reference with retries and backoff."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable

import requests

logger = logging.getLogger(__name__)

USER_AGENT = (
    "nba-similarity/0.1 (personal research project; "
    "https://github.com/will-j-cramer/nba-player-comparison)"
)
DEFAULT_DELAY_S = 4.0
DEFAULT_TIMEOUT_S = 30.0
MAX_ATTEMPTS = 5
BACKOFF_BASE_S = 5.0
MAX_RETRY_AFTER_S = 300.0
RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


class FetchError(Exception):
    """A page could not be fetched after all retries."""


class PageNotFoundError(FetchError):
    """The server returned 404 (e.g. a season that has not started yet)."""


class RateLimitedError(FetchError):
    """The server asked us to wait longer than we are willing to (likely a temporary ban)."""


class PoliteClient:
    """Fetches pages no faster than one request per `delay_s`, retrying transient failures.

    Retries connection errors, timeouts and 5xx responses with exponential backoff and jitter
    (BACKOFF_BASE_S * 2**attempt). A 429 waits for its Retry-After header when that is at most
    MAX_RETRY_AFTER_S; anything longer raises RateLimitedError so the run can be resumed later.
    """

    def __init__(
        self,
        delay_s: float = DEFAULT_DELAY_S,
        *,
        max_attempts: int = MAX_ATTEMPTS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.delay_s = delay_s
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s
        self._session = session or requests.Session()
        self._session.headers["User-Agent"] = USER_AGENT
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter
        self._last_request_at: float | None = None

    def get(self, url: str) -> str:
        """Return the body of `url`, raising a FetchError subclass on permanent failure."""
        for attempt in range(self.max_attempts):
            last_try = attempt == self.max_attempts - 1
            self._wait_for_slot()
            try:
                response = self._session.get(url, timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if last_try:
                    raise FetchError(f"{url}: {exc}") from exc
                self._backoff(attempt, url, str(exc))
                continue

            status = response.status_code
            if status == 200:
                return response.text
            if status == 404:
                raise PageNotFoundError(url)
            if status == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                if retry_after is None or retry_after > MAX_RETRY_AFTER_S or last_try:
                    raise RateLimitedError(
                        f"{url}: rate limited (Retry-After={retry_after}); resume later"
                    )
                logger.warning("429 for %s; waiting %.0fs as requested", url, retry_after)
                self._sleep(retry_after)
                continue
            if status in RETRYABLE_STATUS and not last_try:
                self._backoff(attempt, url, f"HTTP {status}")
                continue
            raise FetchError(f"{url}: HTTP {status}")
        raise AssertionError("unreachable")  # pragma: no cover

    def _wait_for_slot(self) -> None:
        now = self._clock()
        if self._last_request_at is not None:
            remaining = self.delay_s - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_at = self._clock()

    def _backoff(self, attempt: int, url: str, reason: str) -> None:
        wait = BACKOFF_BASE_S * 2**attempt + self._jitter()
        logger.warning("%s for %s; retry %d in %.1fs", reason, url, attempt + 1, wait)
        self._sleep(wait)


def _parse_retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
