"""Tests for T20a: rate limiting + retry + opt-in for NewsCorpusSource."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import requests

from firm.core.clock import ReplayClock
from firm.rag._rate_limit import TokenBucket
from firm.rag.news import NewsCorpusSource


# --- Helpers ---


class _FakeTime:
    """Deterministic monotonic clock for TokenBucket tests."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _replay_clock(iso: str = "2024-03-13T16:00:00+00:00") -> ReplayClock:
    return ReplayClock(fixed=datetime.fromisoformat(iso))


def _polygon_response(ticker: str) -> dict[str, Any]:
    return {
        "results": [
            {
                "id": f"id-{ticker}-1",
                "title": f"{ticker} earnings beat",
                "description": "summary",
                "published_utc": "2024-03-12T13:30:00Z",
                "article_url": "https://example.com/a",
            }
        ]
    }


# --- Tests ---


def test_disabled_flag_makes_zero_http_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """FIRM_NEWS_ENABLED unset → adapter is a no-op even with valid creds."""
    monkeypatch.delenv("FIRM_NEWS_ENABLED", raising=False)
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    calls: list[tuple[str, dict[str, str]]] = []

    def stub_http(url: str, params: dict[str, str]) -> dict[str, Any]:
        calls.append((url, params))
        raise AssertionError("HTTP must not be called when FIRM_NEWS_ENABLED is off")

    source = NewsCorpusSource(
        tickers=["AAPL"],
        clock=_replay_clock(),
        http_client=stub_http,
        db_path=tmp_path / "firm.db",
    )
    docs = list(source.iter_docs())
    assert docs == []
    assert calls == []


def test_six_rapid_calls_two_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bucket rate=4/60s: 4 calls go through immediately, 5th and 6th sleep."""
    monkeypatch.setenv("FIRM_NEWS_ENABLED", "true")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    fake = _FakeTime()
    bucket = TokenBucket(rate=4, per_seconds=60, clock_fn=fake.time, sleep_fn=fake.sleep)
    calls: list[str] = []

    def stub_http(url: str, params: dict[str, str]) -> dict[str, Any]:
        calls.append(params["ticker"])
        return _polygon_response(params["ticker"])

    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    source = NewsCorpusSource(
        tickers=tickers,
        clock=_replay_clock(),
        http_client=stub_http,
        db_path=tmp_path / "firm.db",
        bucket=bucket,
    )
    list(source.iter_docs())
    assert len(calls) == 6, "All six tickers should reach the wire (after waits)"
    # Of 6 calls: first 4 are free, 5 and 6 each wait 15s (per_seconds / rate)
    assert len(fake.sleeps) == 2
    assert fake.sleeps == [pytest.approx(15.0), pytest.approx(15.0)]


def test_429_triggers_one_backoff_before_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 from the upstream → adapter waits and retries the same ticker."""
    monkeypatch.setenv("FIRM_NEWS_ENABLED", "true")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")

    attempts: list[int] = [0]

    def flaky_http(url: str, params: dict[str, str]) -> dict[str, Any]:
        attempts[0] += 1
        if attempts[0] == 1:
            resp = requests.Response()
            resp.status_code = 429
            raise requests.HTTPError("Too Many Requests", response=resp)
        return _polygon_response(params["ticker"])

    sleeps: list[float] = []
    source = NewsCorpusSource(
        tickers=["AAPL"],
        clock=_replay_clock(),
        http_client=flaky_http,
        db_path=tmp_path / "firm.db",
        sleep_fn=lambda s: sleeps.append(s),
    )
    docs = list(source.iter_docs())
    assert attempts[0] == 2, "Expected one retry after 429"
    assert len(sleeps) == 1, f"Expected exactly one backoff sleep, got {sleeps}"
    assert sleeps[0] == pytest.approx(1.0)
    assert len(docs) == 1


def test_cache_hit_skips_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Second call with same (ticker, day) reads cache; no second HTTP."""
    monkeypatch.setenv("FIRM_NEWS_ENABLED", "true")
    monkeypatch.setenv("POLYGON_API_KEY", "test-key")
    call_count = [0]

    def counting_http(url: str, params: dict[str, str]) -> dict[str, Any]:
        call_count[0] += 1
        return _polygon_response(params["ticker"])

    db = tmp_path / "firm.db"
    src1 = NewsCorpusSource(
        tickers=["AAPL"], clock=_replay_clock(), http_client=counting_http, db_path=db
    )
    list(src1.iter_docs())
    assert call_count[0] == 1

    # Second instance, same day: should read cache, no new HTTP
    src2 = NewsCorpusSource(
        tickers=["AAPL"], clock=_replay_clock(), http_client=counting_http, db_path=db
    )
    docs = list(src2.iter_docs())
    assert call_count[0] == 1, "Second call should hit cache, not HTTP"
    assert len(docs) == 1


def test_token_bucket_unit() -> None:
    """Direct TokenBucket test: 4 acquires free, then ~15s wait."""
    fake = _FakeTime()
    b = TokenBucket(rate=4, per_seconds=60, clock_fn=fake.time, sleep_fn=fake.sleep)
    for _ in range(4):
        b.acquire()
    assert fake.sleeps == []
    b.acquire()
    assert fake.sleeps == [pytest.approx(15.0)]
