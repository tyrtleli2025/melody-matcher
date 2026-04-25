"""Last.fm API client with rate limiting and exponential backoff.

Setup
-----
1. Register for a free API key at https://www.last.fm/api/account/create
2. Create a ``.env`` file in the project root::

       LASTFM_API_KEY=your_key_here

3. ``pip install python-dotenv requests`` (or install via pyproject.toml extras)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

_BASE_URL = "https://ws.audioscrobbler.com/2.0/"

# Last.fm embeds application-level errors inside HTTP 200 bodies.
# Code 6 = "Track not found", codes 11/29 = rate-limited / service offline.
_NOT_FOUND_CODE = 6
_RETRYABLE_CODES = frozenset({11, 29})


class LastFmError(Exception):
    """Non-retryable Last.fm API error (invalid key, bad method, etc.)."""


class LastFmClient:
    """Thin, synchronous Last.fm API wrapper.

    Args:
        api_key: Last.fm API key. Defaults to ``LASTFM_API_KEY`` env var.
        calls_per_second: Sustained request rate (default 3.0 — comfortably
            under Last.fm's undocumented ~5 req/s soft limit).
        max_retries: Attempts before giving up on a single call.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        calls_per_second: float = 3.0,
        max_retries: int = 6,
    ) -> None:
        resolved = api_key or os.environ.get("LASTFM_API_KEY", "")
        if not resolved:
            raise ValueError(
                "Last.fm API key not found. "
                "Set LASTFM_API_KEY in your .env file or pass api_key=."
            )
        self._api_key = resolved
        self._min_interval: float = 1.0 / max(calls_per_second, 0.01)
        self._last_call: float = 0.0
        self._max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "melody-matcher/0.1 (research; tjli@mit.edu)"}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_track_info(
        self, artist: str, track: str
    ) -> dict[str, int] | None:
        """Fetch play count and listener count for a track.

        Uses ``autocorrect=1`` so minor spelling variations are tolerated.

        Returns:
            ``{"playcount": int, "listeners": int}`` on success,
            ``None`` if the track is not found on Last.fm.

        Raises:
            LastFmError: On non-retryable API errors (invalid key, etc.).
        """
        params = {
            "method": "track.getInfo",
            "api_key": self._api_key,
            "artist": artist,
            "track": track,
            "autocorrect": "1",
            "format": "json",
        }
        body = self._call(params)
        if body is None:
            return None

        t = body.get("track", {})
        try:
            return {
                "playcount": int(t.get("playcount", 0) or 0),
                "listeners": int(t.get("listeners", 0) or 0),
            }
        except (ValueError, TypeError) as exc:
            log.warning("Unexpected track.getInfo shape for %s – %s: %s", artist, track, exc)
            return None

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "LastFmClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(self, params: dict[str, str]) -> dict[str, Any] | None:
        """Single API call with throttle + retry logic."""
        for attempt in range(self._max_retries):
            self._throttle()
            try:
                resp = self._session.get(_BASE_URL, params=params, timeout=15)
            except requests.RequestException as exc:
                log.warning("Network error (attempt %d/%d): %s", attempt + 1, self._max_retries, exc)
                self._backoff(attempt)
                continue

            if resp.status_code == 200:
                try:
                    body: dict[str, Any] = resp.json()
                except ValueError:
                    log.warning("Non-JSON response on attempt %d", attempt + 1)
                    self._backoff(attempt)
                    continue

                err_code = body.get("error")
                if err_code is None:
                    return body
                if err_code == _NOT_FOUND_CODE:
                    return None
                if err_code in _RETRYABLE_CODES:
                    log.warning("Last.fm soft rate-limit (code %d), backing off …", err_code)
                    self._backoff(attempt)
                    continue
                raise LastFmError(
                    f"Last.fm API error {err_code}: {body.get('message', '(no message)')}"
                )

            if resp.status_code in (429, 500, 502, 503, 504):
                log.warning(
                    "HTTP %d on attempt %d/%d — backing off …",
                    resp.status_code, attempt + 1, self._max_retries,
                )
                self._backoff(attempt)
                continue

            # 4xx other than 429 are not retryable
            log.error("Non-retryable HTTP %d for %s", resp.status_code, params.get("method"))
            return None

        log.error("Exhausted %d retries — giving up.", self._max_retries)
        return None

    def _throttle(self) -> None:
        """Block until the minimum inter-call interval has elapsed."""
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    @staticmethod
    def _backoff(attempt: int) -> None:
        """Exponential backoff capped at 64 seconds."""
        time.sleep(min(2 ** attempt, 64))
