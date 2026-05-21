"""NewsCorpusSource — headlines/articles from Polygon or NewsAPI.

Provider choice is automatic based on which API key is set:
  - POLYGON_API_KEY set       → Polygon /v2/reference/news
  - NEWSAPI_KEY set (only)    → NewsAPI /v2/everything
  - neither                   → no-op (logs once at construct, iter_docs yields nothing)

Window: rolling 12 months from `clock.now()`. The implementation polls
one ticker at a time and yields one FilingDoc per news item.

T20a will add a TokenBucket rate limiter, SQLite per-(ticker, day) cache,
and the FIRM_NEWS_ENABLED opt-in gate on top of this adapter.
"""
from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Iterator
from datetime import datetime

from dateutil.relativedelta import relativedelta  # type: ignore[import-untyped]
from typing import Any

from firm.core.clock import Clock
from firm.rag.source import FilingDoc

logger = logging.getLogger(__name__)

_POLYGON_URL = "https://api.polygon.io/v2/reference/news"
_NEWSAPI_URL = "https://newsapi.org/v2/everything"


def _default_http_client(url: str, params: dict[str, str]) -> dict[str, Any]:
    import requests  # type: ignore[import-untyped]  # lazy import

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _parse_dt(raw: str) -> datetime:
    """Parse an ISO 8601 string, converting Z suffix to +00:00."""
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def _newsapi_hash(ticker: str, published_at: str, title: str) -> str:
    key = f"{ticker}|{published_at}|{title}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


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
    """

    name = "news"

    def __init__(
        self,
        *,
        tickers: list[str],
        clock: Clock,
        http_client: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
    ) -> None:
        self._tickers = tickers
        self._clock = clock
        self._http = http_client if http_client is not None else _default_http_client

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
            logger.warning(
                "NewsCorpusSource: no POLYGON_API_KEY or NEWSAPI_KEY set;"
                " news ingest is a no-op."
            )

    def iter_docs(self) -> Iterator[FilingDoc]:
        if self._provider is None:
            return

        now = self._clock.now()
        twelve_months_ago = now - relativedelta(months=12)
        now_iso = now.isoformat()
        ago_iso = twelve_months_ago.isoformat()

        for ticker in self._tickers:
            try:
                if self._provider == "polygon":
                    yield from self._poll_polygon(ticker, ago_iso, now_iso)
                else:
                    yield from self._poll_newsapi(ticker, ago_iso, now_iso)
            except Exception:
                logger.warning(
                    "NewsCorpusSource: HTTP failure for ticker %s; skipping.",
                    ticker,
                )

    def _poll_polygon(
        self, ticker: str, from_iso: str, to_iso: str
    ) -> Iterator[FilingDoc]:
        params: dict[str, str] = {
            "ticker": ticker,
            "published_utc.gte": from_iso,
            "published_utc.lte": to_iso,
            "limit": "50",
            "apiKey": self._api_key,
        }
        data = self._http(_POLYGON_URL, params)
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
        self, ticker: str, from_iso: str, to_iso: str
    ) -> Iterator[FilingDoc]:
        params: dict[str, str] = {
            "q": ticker,
            "from": from_iso,
            "to": to_iso,
            "sortBy": "publishedAt",
            "pageSize": "50",
            "apiKey": self._api_key,
        }
        data = self._http(_NEWSAPI_URL, params)
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
