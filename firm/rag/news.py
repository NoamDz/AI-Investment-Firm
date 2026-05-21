"""NewsCorpusSource — headlines/articles from Polygon or NewsAPI.

Provider choice is automatic based on which API key is set:
  - POLYGON_API_KEY set       → Polygon /v2/reference/news
  - NEWSAPI_KEY set (only)    → NewsAPI /v2/everything
  - neither                   → no-op (logs once at construct, iter_docs yields nothing)

Window: rolling 12 months from `clock.now()`. The implementation polls
one ticker at a time and yields one FilingDoc per news item.

T20a: adds TokenBucket rate limiter, per-(ticker, day) SQLite cache, and the
FIRM_NEWS_ENABLED opt-in gate. FIRM_NEWS_ENABLED defaults to false; production
opts in by setting it to "true", "1", or "yes".
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests as _requests  # type: ignore[import-untyped]
from dateutil.relativedelta import relativedelta  # type: ignore[import-untyped]

from firm.core.clock import Clock
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.rag._rate_limit import TokenBucket
from firm.rag.source import FilingDoc

logger = logging.getLogger(__name__)

_POLYGON_URL = "https://api.polygon.io/v2/reference/news"
_NEWSAPI_URL = "https://newsapi.org/v2/everything"

_DEFAULT_DB_PATH = Path(os.environ.get("FIRM_DB_PATH", "data/firm.db"))


def _default_http_client(url: str, params: dict[str, str]) -> dict[str, Any]:
    resp = _requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _parse_dt(raw: str) -> datetime:
    """Parse an ISO 8601 string, converting Z suffix to +00:00."""
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _newsapi_hash(ticker: str, published_at: str, title: str) -> str:
    key = f"{ticker}|{published_at}|{title}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _news_enabled() -> bool:
    return os.environ.get("FIRM_NEWS_ENABLED", "false").lower() in {"true", "1", "yes"}


class NewsCorpusSource:
    """CorpusSource backed by Polygon or NewsAPI news endpoints.

    Parameters
    ----------
    tickers:
        Universe symbols to poll (one HTTP call per ticker).
    clock:
        Source of "now" for computing the 12-month window. Tests inject
        a ReplayClock so the window is deterministic.
    http_client:
        Optional injected HTTP client of shape
          ``(url: str, params: dict[str, str]) -> dict[str, Any]``
        that returns the parsed JSON body. Production default reads
        POLYGON_API_KEY / NEWSAPI_KEY from env and wraps requests.get
        with a 10-second timeout. Tests pass a stub so no network I/O.
    bucket:
        Optional TokenBucket for rate limiting. Defaults to
        TokenBucket(rate=4, per_seconds=60). Tests inject a bucket with
        stub clock/sleep for deterministic behaviour.
    db_path:
        SQLite DB path for the per-(ticker, day) cache. Defaults to
        env FIRM_DB_PATH or data/firm.db. The news_cache table is
        initialised lazily on first cache access.
    sleep_fn:
        Callable used for retry backoff sleeps. Injectable for tests.
    """

    name = "news"

    def __init__(
        self,
        *,
        tickers: list[str],
        clock: Clock,
        http_client: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
        bucket: TokenBucket | None = None,
        db_path: Path | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._tickers = tickers
        self._clock = clock
        self._http = http_client if http_client is not None else _default_http_client
        self._bucket = bucket if bucket is not None else TokenBucket(rate=4, per_seconds=60)
        self._db_path = db_path if db_path is not None else _DEFAULT_DB_PATH
        self._sleep_fn = sleep_fn
        self._db_initialised = False

        polygon_key = os.environ.get("POLYGON_API_KEY")
        newsapi_key = os.environ.get("NEWSAPI_KEY")

        if polygon_key:
            self._provider: str | None = "polygon"
            self._api_key = polygon_key
        elif newsapi_key:
            self._provider = "newsapi"
            self._api_key = newsapi_key
        else:
            self._provider = None
            self._api_key = ""
            if _news_enabled():
                logger.warning(
                    "NewsCorpusSource: no POLYGON_API_KEY or NEWSAPI_KEY set;"
                    " news ingest is a no-op."
                )

    def _ensure_db(self) -> None:
        """Initialise news_cache table on first cache access (idempotent)."""
        if not self._db_initialised:
            init_db(self._db_path)
            self._db_initialised = True

    def _cache_get(self, ticker: str, day: str, provider: str) -> dict[str, Any] | None:
        """Return cached response JSON or None on a cache miss."""
        self._ensure_db()
        with closing(get_conn(self._db_path)) as conn:
            row = conn.execute(
                "SELECT response_json FROM news_cache WHERE ticker=? AND day=? AND provider=?",
                (ticker, day, provider),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["response_json"])  # type: ignore[no-any-return]

    def _cache_set(
        self, ticker: str, day: str, provider: str, response: dict[str, Any]
    ) -> None:
        """Insert or replace a cache row."""
        self._ensure_db()
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        with closing(get_conn(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO news_cache"
                " (ticker, day, provider, response_json, created_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (ticker, day, provider, json.dumps(response), now_iso),
            )

    def _http_with_retries(
        self,
        url: str,
        params: dict[str, str],
        *,
        max_retries: int = 3,
        base: float = 1.0,
    ) -> dict[str, Any]:
        """Call self._http with exponential backoff on HTTP 429."""
        for attempt in range(max_retries + 1):
            try:
                return self._http(url, params)
            except _requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 429:
                    if attempt < max_retries:
                        wait = base * (2**attempt)
                        self._sleep_fn(wait)
                        continue
                raise
        # Unreachable but satisfies mypy
        raise RuntimeError("_http_with_retries: exhausted retries")  # pragma: no cover

    def iter_docs(self) -> Iterator[FilingDoc]:
        # Gate 1: feature flag (checked at call time, not construct time)
        if not _news_enabled():
            logger.debug("NewsCorpusSource: FIRM_NEWS_ENABLED is off; skipping news ingest.")
            return

        # Gate 2: no credentials
        if self._provider is None:
            return

        now = self._clock.now()
        twelve_months_ago = now - relativedelta(months=12)
        now_iso = now.isoformat()
        ago_iso = twelve_months_ago.isoformat()
        today = now.date().isoformat()

        for ticker in self._tickers:
            try:
                if self._provider == "polygon":
                    yield from self._poll_polygon(ticker, ago_iso, now_iso, today)
                else:
                    yield from self._poll_newsapi(ticker, ago_iso, now_iso, today)
            except Exception:
                logger.warning(
                    "NewsCorpusSource: HTTP failure for ticker %s; skipping.",
                    ticker,
                )

    def _poll_polygon(
        self, ticker: str, from_iso: str, to_iso: str, today: str
    ) -> Iterator[FilingDoc]:
        provider = "polygon"
        cached = self._cache_get(ticker, today, provider)
        if cached is not None:
            data = cached
        else:
            params: dict[str, str] = {
                "ticker": ticker,
                "published_utc.gte": from_iso,
                "published_utc.lte": to_iso,
                "limit": "50",
                "apiKey": self._api_key,
            }
            self._bucket.acquire()
            data = self._http_with_retries(_POLYGON_URL, params)
            self._cache_set(ticker, today, provider, data)

        results: list[dict[str, Any]] = data.get("results") or []
        for item in results:
            title = str(item.get("title") or "")
            description = str(item.get("description") or "")
            html = description if description else title
            if not html:
                logger.debug(
                    "NewsCorpusSource (polygon): skipping item with empty"
                    " title and description for ticker %s",
                    ticker,
                )
                continue
            published_raw = str(item.get("published_utc") or "")
            try:
                published_at = _parse_dt(published_raw)
            except (ValueError, TypeError):
                logger.debug(
                    "NewsCorpusSource (polygon): skipping item with"
                    " unparseable published_utc %r for ticker %s",
                    published_raw,
                    ticker,
                )
                continue
            item_id = str(item.get("id") or "")
            if not item_id:
                item_id = _newsapi_hash(ticker, published_raw, title)
            doc_id = f"news-polygon-{item_id}"
            article_url = str(item.get("article_url") or "") or None
            yield FilingDoc(
                doc_id=doc_id,
                ticker=ticker,
                filing_type="news",
                published_at=published_at,
                title=title,
                html=html,
                url=article_url,
                metadata={"provider": "polygon", "polled_for_ticker": ticker},
            )

    def _poll_newsapi(
        self, ticker: str, from_iso: str, to_iso: str, today: str
    ) -> Iterator[FilingDoc]:
        provider = "newsapi"
        cached = self._cache_get(ticker, today, provider)
        if cached is not None:
            data = cached
        else:
            params: dict[str, str] = {
                "q": ticker,
                "from": from_iso,
                "to": to_iso,
                "sortBy": "publishedAt",
                "pageSize": "50",
                "apiKey": self._api_key,
            }
            self._bucket.acquire()
            data = self._http_with_retries(_NEWSAPI_URL, params)
            self._cache_set(ticker, today, provider, data)

        articles: list[dict[str, Any]] = data.get("articles") or []
        for item in articles:
            title = str(item.get("title") or "")
            description = str(item.get("description") or "")
            html = description if description else title
            if not html:
                logger.debug(
                    "NewsCorpusSource (newsapi): skipping item with empty"
                    " title and description for ticker %s",
                    ticker,
                )
                continue
            published_raw = str(item.get("publishedAt") or "")
            try:
                published_at = _parse_dt(published_raw)
            except (ValueError, TypeError):
                logger.debug(
                    "NewsCorpusSource (newsapi): skipping item with"
                    " unparseable publishedAt %r for ticker %s",
                    published_raw,
                    ticker,
                )
                continue
            stable_hash = _newsapi_hash(ticker, published_raw, title)
            doc_id = f"news-newsapi-{stable_hash}"
            article_url = str(item.get("url") or "") or None
            yield FilingDoc(
                doc_id=doc_id,
                ticker=ticker,
                filing_type="news",
                published_at=published_at,
                title=title,
                html=html,
                url=article_url,
                metadata={"provider": "newsapi", "polled_for_ticker": ticker},
            )
