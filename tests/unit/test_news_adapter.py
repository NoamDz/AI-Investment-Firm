"""Tests for NewsCorpusSource adapter (Plan 3 T20)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pytest

from firm.core.clock import ReplayClock
from firm.rag.news import NewsCorpusSource
from firm.rag.source import CorpusSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED_NOW = datetime(2024, 3, 13, 16, 0, 0, tzinfo=timezone.utc)

_POLYGON_ITEM_1: dict[str, Any] = {
    "id": "abc123",
    "title": "Apple hits new high",
    "description": "Apple Inc. stock rose sharply on strong earnings.",
    "article_url": "https://example.com/apple-1",
    "published_utc": "2024-03-13T16:00:00Z",
    "tickers": ["AAPL"],
}
_POLYGON_ITEM_2: dict[str, Any] = {
    "id": "def456",
    "title": "Apple Watch update",
    "description": "Apple releases watchOS update.",
    "article_url": "https://example.com/apple-2",
    "published_utc": "2024-03-12T10:00:00Z",
    "tickers": ["AAPL"],
}
_POLYGON_RESPONSE_2: dict[str, Any] = {"results": [_POLYGON_ITEM_1, _POLYGON_ITEM_2]}
_POLYGON_RESPONSE_1: dict[str, Any] = {"results": [_POLYGON_ITEM_1]}

_NEWSAPI_ITEM_1: dict[str, Any] = {
    "source": {"name": "Reuters"},
    "title": "NVDA sets record",
    "description": "Nvidia shares hit an all-time high.",
    "url": "https://example.com/nvda-1",
    "publishedAt": "2024-03-13T14:00:00Z",
}
_NEWSAPI_RESPONSE_1: dict[str, Any] = {"articles": [_NEWSAPI_ITEM_1]}


def _make_stub(
    responses: dict[str, Any] | list[Any],
) -> Callable[[str, dict[str, str]], dict[str, Any]]:
    """Build an http_client stub.

    If ``responses`` is a dict, it is returned directly (same response every
    call — useful when a single provider shape is needed regardless of URL or
    params).
    If ``responses`` is a list, return entries in call order; if the list is
    exhausted, return an empty provider-neutral dict.
    """
    if isinstance(responses, dict):
        return lambda url, params: responses  # type: ignore[return-value]
    call_iter = iter(responses)

    def _stub(url: str, params: dict[str, str]) -> dict[str, Any]:
        try:
            return next(call_iter)
        except StopIteration:
            return {}

    return _stub


def _replay(fixed: datetime = _FIXED_NOW) -> ReplayClock:
    return ReplayClock(fixed=fixed)


# ---------------------------------------------------------------------------
# 1. No creds → empty iter
# ---------------------------------------------------------------------------


def test_no_creds_returns_empty_iter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    def _fail_stub(url: str, params: dict[str, str]) -> dict[str, Any]:
        pytest.fail("must not call http_client when no creds are set")

    source = NewsCorpusSource(
        tickers=["AAPL"], clock=_replay(), http_client=_fail_stub
    )
    assert list(source.iter_docs()) == []


# ---------------------------------------------------------------------------
# 2. No creds → logs warning at least once
# ---------------------------------------------------------------------------


def test_no_creds_logs_warning_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    with caplog.at_level(logging.WARNING, logger="firm.rag.news"):
        NewsCorpusSource(tickers=["AAPL"], clock=_replay())
        NewsCorpusSource(tickers=["NVDA"], clock=_replay())

    warnings = [
        r for r in caplog.records if r.levelno == logging.WARNING and "no-op" in r.message
    ]
    assert len(warnings) >= 1, "Expected at least one warning about missing API keys"


# ---------------------------------------------------------------------------
# 3. Polygon path → yields FilingDocs
# ---------------------------------------------------------------------------


def test_polygon_path_yields_filing_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test_key")
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    source = NewsCorpusSource(
        tickers=["AAPL"],
        clock=_replay(),
        http_client=_make_stub(_POLYGON_RESPONSE_2),
    )
    docs = list(source.iter_docs())
    assert len(docs) == 2
    for doc in docs:
        assert doc.ticker == "AAPL"
        assert doc.filing_type == "news"
        assert doc.published_at.tzinfo is not None
        assert doc.doc_id.startswith("news-polygon-")


# ---------------------------------------------------------------------------
# 4. NewsAPI path → yields FilingDocs with correct doc_id form
# ---------------------------------------------------------------------------


def test_newsapi_path_yields_filing_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setenv("NEWSAPI_KEY", "test_key")

    source = NewsCorpusSource(
        tickers=["NVDA"],
        clock=_replay(),
        http_client=_make_stub(_NEWSAPI_RESPONSE_1),
    )
    docs = list(source.iter_docs())
    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id.startswith("news-newsapi-")
    # doc_id suffix must be a 16-char hex string (sha256 truncated)
    suffix = doc.doc_id[len("news-newsapi-"):]
    assert len(suffix) == 16
    assert all(c in "0123456789abcdef" for c in suffix)


# ---------------------------------------------------------------------------
# 5. Polygon preferred when both keys are set
# ---------------------------------------------------------------------------


def test_polygon_preferred_when_both_keys_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "poly_key")
    monkeypatch.setenv("NEWSAPI_KEY", "newsapi_key")

    captured_urls: list[str] = []

    def _capturing_stub(url: str, params: dict[str, str]) -> dict[str, Any]:
        captured_urls.append(url)
        return {"results": []}

    source = NewsCorpusSource(
        tickers=["AAPL"],
        clock=_replay(),
        http_client=_capturing_stub,
    )
    list(source.iter_docs())

    assert len(captured_urls) == 1
    assert "polygon.io" in captured_urls[0], (
        f"Expected Polygon URL, got {captured_urls[0]!r}"
    )
    assert "newsapi.org" not in captured_urls[0]


# ---------------------------------------------------------------------------
# 6. Rolling 12-month window
# ---------------------------------------------------------------------------


def test_window_is_rolling_12_months(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test_key")
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    fixed_now = datetime(2024, 3, 13, 16, 0, 0, tzinfo=timezone.utc)
    expected_from = datetime(2023, 3, 13, 16, 0, 0, tzinfo=timezone.utc)

    captured_params: list[dict[str, str]] = []

    def _capturing_stub(url: str, params: dict[str, str]) -> dict[str, Any]:
        captured_params.append(dict(params))
        return {"results": []}

    source = NewsCorpusSource(
        tickers=["AAPL"],
        clock=ReplayClock(fixed=fixed_now),
        http_client=_capturing_stub,
    )
    list(source.iter_docs())

    assert len(captured_params) == 1
    from_str = captured_params[0].get("published_utc.gte", "")
    assert from_str, "Expected published_utc.gte param to be set"
    parsed_from = datetime.fromisoformat(from_str.replace("Z", "+00:00"))
    diff = abs((parsed_from - expected_from).total_seconds())
    assert diff <= 1.0, (
        f"Expected from={expected_from.isoformat()} but got {from_str!r}"
        f" (diff={diff}s)"
    )


# ---------------------------------------------------------------------------
# 7. Single ticker HTTP failure does not abort iteration
# ---------------------------------------------------------------------------


def test_one_ticker_http_failure_does_not_abort_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test_key")
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    def _selective_stub(url: str, params: dict[str, str]) -> dict[str, Any]:
        ticker = params.get("ticker", "")
        if ticker == "AAPL":
            raise RuntimeError("simulated HTTP failure for AAPL")
        return {"results": [_POLYGON_ITEM_1]}

    source = NewsCorpusSource(
        tickers=["AAPL", "NVDA"],
        clock=_replay(),
        http_client=_selective_stub,
    )
    # Patch _POLYGON_ITEM_1 ticker field in results; item ticker doesn't matter
    # since we use polled ticker — just need 1 result from NVDA
    docs = list(source.iter_docs())
    assert len(docs) == 1
    assert docs[0].ticker == "NVDA"


# ---------------------------------------------------------------------------
# 8. Polygon Z-suffix published_at parses to UTC
# ---------------------------------------------------------------------------


def test_polygon_z_suffix_published_at_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLYGON_API_KEY", "test_key")
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    item: dict[str, Any] = {
        "id": "ztest1",
        "title": "Z suffix test",
        "description": "Test article.",
        "article_url": "https://example.com/z",
        "published_utc": "2024-03-13T16:00:00Z",
        "tickers": ["AAPL"],
    }

    source = NewsCorpusSource(
        tickers=["AAPL"],
        clock=_replay(),
        http_client=_make_stub({"results": [item]}),
    )
    docs = list(source.iter_docs())
    assert len(docs) == 1
    doc = docs[0]
    assert doc.published_at.tzinfo is not None
    # UTC offset must be zero
    offset = doc.published_at.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0.0


# ---------------------------------------------------------------------------
# 9. Implements CorpusSource protocol
# ---------------------------------------------------------------------------


def test_implements_corpus_source_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    source = NewsCorpusSource(tickers=["AAPL"], clock=_replay())
    assert isinstance(source, CorpusSource)
    assert source.name == "news"


# ---------------------------------------------------------------------------
# 10. Empty description falls back to title as html
# ---------------------------------------------------------------------------


def test_empty_description_uses_title_as_html(monkeypatch: pytest.MonkeyPatch) -> None:
    """When description is empty but title is not, title is used as html content."""
    monkeypatch.setenv("POLYGON_API_KEY", "test_key")
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    item: dict[str, Any] = {
        "id": "nodesc1",
        "title": "No description article",
        "description": "",  # empty description
        "article_url": "https://example.com/nodesc",
        "published_utc": "2024-03-13T16:00:00Z",
        "tickers": ["AAPL"],
    }

    source = NewsCorpusSource(
        tickers=["AAPL"],
        clock=_replay(),
        http_client=_make_stub({"results": [item]}),
    )
    docs = list(source.iter_docs())
    assert len(docs) == 1
    assert docs[0].html == "No description article"
