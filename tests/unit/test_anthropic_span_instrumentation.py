"""Span-instrumentation tests for :class:`CachedAnthropicClient` (Plan 3 T04).

The wrapper must stamp the *currently active* OTel span with the four
token/cost fields from the spec §10.1 schema:

* **Cached hit** -> ``cached_tokens`` (= sum of cached input + output), and
  ``cost_usd = 0.0``. ``input_tokens`` / ``output_tokens`` are NOT set
  (they semantically imply a real, billed call).
* **Live call** -> ``input_tokens``, ``output_tokens``, ``cost_usd`` from the
  per-model rate card in ``config/router.yaml``. ``cached_tokens`` is NOT set.

The active span is established by :func:`firm.obs.llm_span` in T03's
research / pm wiring; here we drive it directly so the unit under test is
isolated from the agent layer.

These tests pin all six branches of the messages_create matrix:
CACHED-hit / LIVE / RECORD-miss / RECORD-hit, plus the no-active-span and
unknown-model edge cases.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from firm.core.clock import ReplayClock
from firm.db.migrations import init_db
from firm.llm.anthropic_client import (
    CachedAnthropicClient,
    LlmMode,
)
from firm.llm.cache import LlmCache, hash_prompt
from firm.obs import llm_span
from firm.obs.tracer import use_sync_exporter


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeTransport:
    """Recording fake transport returning a canned response dict per call."""

    def __init__(self, response: dict[str, object]) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def messages_create(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return dict(self._response)


def _make_cache(tmp_path: Path) -> LlmCache:
    db = tmp_path / "t04.db"
    init_db(db)
    clock = ReplayClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))
    return LlmCache(db, clock)


def _make_clock() -> ReplayClock:
    return ReplayClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))


def _read_spans(traces_root: Path) -> list[dict[str, Any]]:
    """Return all span dicts under *traces_root* (recursively)."""
    spans: list[dict[str, Any]] = []
    for jsonl_file in traces_root.rglob("*.jsonl"):
        for raw in jsonl_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line:
                spans.append(json.loads(line))
    return spans


def _llm_call_span(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the single ``llm.call`` span (raise if missing or duplicated)."""
    matches = [s for s in spans if s.get("operation") == "llm.call"]
    assert len(matches) == 1, (
        f"expected exactly one llm.call span, got {len(matches)}: {matches}"
    )
    return matches[0]


def _redirect_traces(tmp_path: Path, run_id: str) -> Path:
    """Point the OTel sync exporter at a per-test ``traces`` dir and return it."""
    traces_root = tmp_path / "traces"
    use_sync_exporter(traces_root=traces_root, run_id=run_id)
    return traces_root


# ---------------------------------------------------------------------------
# 1. CACHED-mode hit stamps cached_tokens + cost_usd=0.0, not input/output
# ---------------------------------------------------------------------------


def test_cached_hit_stamps_cached_tokens_and_zero_cost(tmp_path: Path) -> None:
    traces_root = _redirect_traces(tmp_path, run_id="t04test01cachedhit00000000")

    cache = _make_cache(tmp_path)
    clock = _make_clock()

    # Pre-populate cache with known usage.
    model = "claude-sonnet-4-6"
    system = "sys"
    messages: list[dict[str, object]] = [{"role": "user", "content": "hi"}]
    canned = {
        "content": [{"type": "text", "text": "hello"}],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }
    cache.put(
        prompt_hash=hash_prompt(system=system, messages=messages, tools=None),
        model=model,
        response_json=json.dumps(canned),
        input_tokens=100,
        output_tokens=50,
    )

    client = CachedAnthropicClient(
        api_key=None,
        cache=cache,
        mode=LlmMode.CACHED,
        clock=clock,
        transport=None,
    )

    with llm_span("anthropic", model):
        raw = client.messages_create(
            model=model, system=system, messages=messages, max_tokens=100
        )
    assert raw["_cache_hit"] is True

    span = _llm_call_span(_read_spans(traces_root))
    assert span["cached_tokens"] == 150
    assert span["cost_usd"] == 0.0
    # Live-call fields must remain at the schema's natural zero (not set).
    assert span["input_tokens"] == 0
    assert span["output_tokens"] == 0


# ---------------------------------------------------------------------------
# 2. LIVE-mode stamps input/output/cost, not cached_tokens
# ---------------------------------------------------------------------------


def test_live_call_stamps_tokens_and_computed_cost(tmp_path: Path) -> None:
    traces_root = _redirect_traces(tmp_path, run_id="t04test02livecall000000000")

    cache = _make_cache(tmp_path)
    clock = _make_clock()
    transport = _FakeTransport(
        {
            "content": [{"type": "text", "text": "live"}],
            "usage": {"input_tokens": 1000, "output_tokens": 500},
        }
    )

    client = CachedAnthropicClient(
        api_key="sk-test",
        cache=cache,
        mode=LlmMode.LIVE,
        clock=clock,
        transport=transport,
    )

    model = "claude-sonnet-4-6"  # priced at 3.00/15.00 per Mtok in router.yaml
    with llm_span("anthropic", model):
        raw = client.messages_create(
            model=model,
            system="s",
            messages=[{"role": "user", "content": "go"}],
            max_tokens=200,
        )
    assert raw["_cache_hit"] is False

    span = _llm_call_span(_read_spans(traces_root))
    assert span["input_tokens"] == 1000
    assert span["output_tokens"] == 500
    # 1000/1e6 * 3.00 + 500/1e6 * 15.00 = 0.003 + 0.0075 = 0.0105
    assert span["cost_usd"] == pytest.approx(0.0105, rel=1e-9)
    # cached_tokens must remain at schema zero (we didn't serve from cache).
    assert span["cached_tokens"] == 0


# ---------------------------------------------------------------------------
# 3. No active span -> no crash, no exception, returns the dict
# ---------------------------------------------------------------------------


def test_no_active_span_does_not_crash(tmp_path: Path) -> None:
    _redirect_traces(tmp_path, run_id="t04test03noactivespan00000")

    cache = _make_cache(tmp_path)
    clock = _make_clock()
    transport = _FakeTransport(
        {
            "content": [{"type": "text", "text": "x"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )

    client = CachedAnthropicClient(
        api_key="sk-test",
        cache=cache,
        mode=LlmMode.LIVE,
        clock=clock,
        transport=transport,
    )

    # No surrounding llm_span: active span is INVALID_SPAN. Must not raise.
    raw = client.messages_create(
        model="claude-sonnet-4-6",
        system="s",
        messages=[{"role": "user", "content": "go"}],
        max_tokens=10,
    )
    assert raw["_cache_hit"] is False
    assert raw["content"][0]["text"] == "x"


# ---------------------------------------------------------------------------
# 4. Unknown model -> cost_usd=0.0 (graceful degradation)
# ---------------------------------------------------------------------------


def test_unknown_model_yields_zero_cost(tmp_path: Path) -> None:
    traces_root = _redirect_traces(
        tmp_path, run_id="t04test04unknownmodel00000"
    )

    cache = _make_cache(tmp_path)
    clock = _make_clock()
    transport = _FakeTransport(
        {
            "content": [{"type": "text", "text": "x"}],
            "usage": {"input_tokens": 200, "output_tokens": 100},
        }
    )

    client = CachedAnthropicClient(
        api_key="sk-test",
        cache=cache,
        mode=LlmMode.LIVE,
        clock=clock,
        transport=transport,
    )

    model = "unknown-model-id"
    with llm_span("anthropic", model):
        client.messages_create(
            model=model,
            system="s",
            messages=[{"role": "user", "content": "go"}],
            max_tokens=10,
        )

    span = _llm_call_span(_read_spans(traces_root))
    # Tokens are recorded (we know the count), cost is 0.0 because we have
    # no rate-card entry for the model.
    assert span["input_tokens"] == 200
    assert span["output_tokens"] == 100
    assert span["cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# 5. RECORD-mode miss is a live call (stamps tokens + cost)
# ---------------------------------------------------------------------------


def test_record_miss_stamps_like_live(tmp_path: Path) -> None:
    traces_root = _redirect_traces(
        tmp_path, run_id="t04test05recordmiss0000000"
    )

    cache = _make_cache(tmp_path)
    clock = _make_clock()
    transport = _FakeTransport(
        {
            "content": [{"type": "text", "text": "fresh"}],
            "usage": {"input_tokens": 400, "output_tokens": 200},
        }
    )

    client = CachedAnthropicClient(
        api_key="sk-test",
        cache=cache,
        mode=LlmMode.RECORD,
        clock=clock,
        transport=transport,
    )

    model = "claude-haiku-4-5"  # priced at 0.80/4.00 per Mtok in router.yaml
    with llm_span("anthropic", model):
        raw = client.messages_create(
            model=model,
            system="s",
            messages=[{"role": "user", "content": "rec"}],
            max_tokens=128,
        )
    assert raw["_cache_hit"] is False

    span = _llm_call_span(_read_spans(traces_root))
    assert span["input_tokens"] == 400
    assert span["output_tokens"] == 200
    # 400/1e6 * 0.80 + 200/1e6 * 4.00 = 0.00032 + 0.00080 = 0.00112
    assert span["cost_usd"] == pytest.approx(0.00112, rel=1e-9)
    assert span["cached_tokens"] == 0


# ---------------------------------------------------------------------------
# 6. RECORD-mode hit is a cached call (stamps cached_tokens + zero cost)
# ---------------------------------------------------------------------------


def test_record_hit_stamps_like_cached(tmp_path: Path) -> None:
    traces_root = _redirect_traces(tmp_path, run_id="t04test06recordhit00000000")

    cache = _make_cache(tmp_path)
    clock = _make_clock()

    model = "claude-haiku-4-5"
    system = "s"
    messages: list[dict[str, object]] = [{"role": "user", "content": "rec"}]
    # Pre-populate cache so the RECORD path takes the cache-hit branch.
    cache.put(
        prompt_hash=hash_prompt(system=system, messages=messages, tools=None),
        model=model,
        response_json=json.dumps(
            {
                "content": [{"type": "text", "text": "cached"}],
                "usage": {"input_tokens": 60, "output_tokens": 40},
            }
        ),
        input_tokens=60,
        output_tokens=40,
    )

    transport = _FakeTransport(
        {
            "content": [{"type": "text", "text": "should-not-be-used"}],
            "usage": {"input_tokens": 9999, "output_tokens": 9999},
        }
    )

    client = CachedAnthropicClient(
        api_key="sk-test",
        cache=cache,
        mode=LlmMode.RECORD,
        clock=clock,
        transport=transport,
    )

    with llm_span("anthropic", model):
        raw = client.messages_create(
            model=model, system=system, messages=messages, max_tokens=128
        )
    assert raw["_cache_hit"] is True
    assert transport.calls == []  # cache-first must short-circuit

    span = _llm_call_span(_read_spans(traces_root))
    assert span["cached_tokens"] == 100
    assert span["cost_usd"] == 0.0
    assert span["input_tokens"] == 0
    assert span["output_tokens"] == 0
