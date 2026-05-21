"""Tests for the cached, mode-aware Anthropic client wrapper. See Plan 2 §T16.

The wrapper supports three modes:

* ``LIVE`` -- always call the real API; never write to cache.
* ``CACHED`` -- only serve from the local cache. Cache misses raise
  :class:`LlmCacheMissError`. Used for fully deterministic replay (no API key
  needed).
* ``RECORD`` -- cache-first; on miss call the real API and write the response.

The four canonical tests below pin the per-mode behaviour and the Citations API
response-shape passthrough required by T18.
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
    LlmCacheMissError,
    LlmMode,
)
from firm.llm.cassettes import CassetteClient, CassetteMissError
from firm.llm.cache import LlmCache, hash_prompt


class FakeAnthropicTransport:
    """Recording fake transport that returns a canned response dict per call."""

    def __init__(self, response: dict[str, object]) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def messages_create(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        # Shallow copy so caller mutations (e.g. _cache_hit) do not bleed back
        # into our canned response and corrupt subsequent calls.
        return dict(self._response)


def _make_cache(tmp_path: Path) -> LlmCache:
    db = tmp_path / "t16.db"
    init_db(db)
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))
    return LlmCache(db, clock)


def _basic_response() -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": "hello world"}],
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }


def test_client_in_cached_mode_returns_cache_or_raises(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))
    transport = FakeAnthropicTransport(_basic_response())

    client = CachedAnthropicClient(
        api_key=None,
        cache=cache,
        mode=LlmMode.CACHED,
        clock=clock,
        transport=transport,
    )

    model = "claude-haiku-4-5"
    system = "sys"
    messages: list[dict[str, object]] = [{"role": "user", "content": "hi"}]

    # Empty cache + CACHED mode -> miss must raise.
    with pytest.raises(LlmCacheMissError):
        client.messages_create(
            model=model, system=system, messages=messages, max_tokens=100
        )

    # Transport must NOT be called in CACHED mode, even on miss.
    assert transport.calls == []

    # Seed cache directly with a matching prompt_hash.
    canned = _basic_response()
    cache.put(
        prompt_hash=hash_prompt(system=system, messages=messages, tools=None),
        model=model,
        response_json=json.dumps(canned),
        input_tokens=12,
        output_tokens=7,
    )

    raw = client.messages_create(
        model=model, system=system, messages=messages, max_tokens=100
    )
    assert raw["content"] == canned["content"]
    assert raw["_cache_hit"] is True
    assert transport.calls == []  # still no live call


def test_client_in_record_mode_calls_api_and_writes_cache(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))
    canned = _basic_response()
    transport = FakeAnthropicTransport(canned)

    client = CachedAnthropicClient(
        api_key="sk-test",
        cache=cache,
        mode=LlmMode.RECORD,
        clock=clock,
        transport=transport,
    )

    model = "claude-sonnet-4-6"
    system = "you are an analyst"
    messages: list[dict[str, object]] = [{"role": "user", "content": "explain"}]

    raw1 = client.messages_create(
        model=model, system=system, messages=messages, max_tokens=200
    )
    assert raw1["_cache_hit"] is False
    assert raw1["content"] == canned["content"]
    assert len(transport.calls) == 1

    # Verify cache has the entry.
    prompt_hash = hash_prompt(system=system, messages=messages, tools=None)
    hit = cache.get(prompt_hash=prompt_hash, model=model)
    assert hit is not None
    assert json.loads(hit.response_json)["content"] == canned["content"]

    # Second identical call must be served from cache, not the transport.
    raw2 = client.messages_create(
        model=model, system=system, messages=messages, max_tokens=200
    )
    assert raw2["_cache_hit"] is True
    assert len(transport.calls) == 1  # NOT incremented


def test_client_in_live_mode_bypasses_cache_writes(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))
    transport = FakeAnthropicTransport(_basic_response())

    client = CachedAnthropicClient(
        api_key="sk-test",
        cache=cache,
        mode=LlmMode.LIVE,
        clock=clock,
        transport=transport,
    )

    model = "claude-sonnet-4-6"
    system = "live system"
    messages: list[dict[str, object]] = [{"role": "user", "content": "go"}]

    raw1 = client.messages_create(
        model=model, system=system, messages=messages, max_tokens=64
    )
    assert raw1["_cache_hit"] is False
    raw2 = client.messages_create(
        model=model, system=system, messages=messages, max_tokens=64
    )
    assert raw2["_cache_hit"] is False
    assert len(transport.calls) == 2

    # Cache must remain empty after LIVE calls.
    prompt_hash = hash_prompt(system=system, messages=messages, tools=None)
    assert cache.get(prompt_hash=prompt_hash, model=model) is None


def test_client_handles_citations_response_shape(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))

    citations_response: dict[str, Any] = {
        "content": [
            {
                "type": "text",
                "text": "Apple's Q3 revenue was significant.",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "Q3 revenue rose 8%",
                        "document_index": 0,
                        "document_title": "10-K",
                        "start_char_index": 0,
                        "end_char_index": 50,
                    }
                ],
            }
        ],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }
    transport = FakeAnthropicTransport(citations_response)

    client = CachedAnthropicClient(
        api_key="sk-test",
        cache=cache,
        mode=LlmMode.RECORD,
        clock=clock,
        transport=transport,
    )

    model = "claude-sonnet-4-6"
    system = "cite your sources"
    messages: list[dict[str, object]] = [{"role": "user", "content": "what is Q3?"}]

    raw1 = client.messages_create(
        model=model, system=system, messages=messages, max_tokens=300
    )
    # Citations must survive the wrapper untouched.
    block = raw1["content"][0]
    assert isinstance(block, dict)
    assert block["type"] == "text"
    assert block["citations"][0]["cited_text"] == "Q3 revenue rose 8%"
    assert block["citations"][0]["document_title"] == "10-K"
    assert raw1["_cache_hit"] is False

    # Second call: served from cache. Citations must survive JSON round-trip.
    raw2 = client.messages_create(
        model=model, system=system, messages=messages, max_tokens=300
    )
    assert raw2["_cache_hit"] is True
    block2 = raw2["content"][0]
    assert isinstance(block2, dict)
    assert block2["citations"] == citations_response["content"][0]["citations"]
    assert len(transport.calls) == 1


def test_complete_protocol_method_returns_completion_response(tmp_path: Path) -> None:
    """The wrapper also implements the T9 AnthropicClient Protocol via .complete()."""
    cache = _make_cache(tmp_path)
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))
    transport = FakeAnthropicTransport(
        {
            "content": [
                {"type": "text", "text": "part one. "},
                {"type": "text", "text": "part two."},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
    )
    client = CachedAnthropicClient(
        api_key="sk-test",
        cache=cache,
        mode=LlmMode.LIVE,
        clock=clock,
        transport=transport,
    )

    resp = client.complete(
        model="claude-haiku-4-5",
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=128,
    )
    assert resp.text == "part one. part two."
    assert resp.input_tokens == 5
    assert resp.output_tokens == 3
    assert resp.cache_hit is False


def test_from_env_reads_firm_llm_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = _make_cache(tmp_path)
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))

    monkeypatch.setenv("FIRM_LLM_MODE", "cached")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = CachedAnthropicClient.from_env(cache=cache, clock=clock)
    assert client.mode == LlmMode.CACHED

    # cached mode is permitted without an API key.
    with pytest.raises(LlmCacheMissError):
        client.messages_create(
            model="claude-haiku-4-5",
            system="s",
            messages=[{"role": "user", "content": "x"}],
            max_tokens=10,
        )


def test_live_or_record_without_api_key_raises(tmp_path: Path) -> None:
    cache = _make_cache(tmp_path)
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        CachedAnthropicClient(
            api_key=None,
            cache=cache,
            mode=LlmMode.LIVE,
            clock=clock,
            transport=None,
        )


def test_from_env_vcr_mode_wires_cassette_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """from_env() with FIRM_LLM_MODE=live + FIRM_VCR_MODE=replay wraps the
    SDK transport in a CassetteClient; messages_create raises CassetteMissError
    because tmp_path has no cassettes."""
    cache = _make_cache(tmp_path)
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))

    monkeypatch.setenv("FIRM_LLM_MODE", "live")
    monkeypatch.setenv("FIRM_VCR_MODE", "replay")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-for-test")
    monkeypatch.setenv("FIRM_CASSETTE_DIR", str(tmp_path))

    client = CachedAnthropicClient.from_env(cache=cache, clock=clock)

    # The transport must be a CassetteClient (not raw SDK transport).
    assert isinstance(client._transport, CassetteClient)

    # Replay with no cassettes must raise CassetteMissError.
    with pytest.raises(CassetteMissError):
        client.messages_create(
            model="claude-haiku-4-5",
            system="test system",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
        )
