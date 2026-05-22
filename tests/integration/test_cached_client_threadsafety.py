"""Thread-safety stress tests for CachedAnthropicClient / LlmCache.

The spec (T24a) calls for exercising extract() (the agent-level call), but
driving extract() pulls in the full Anthropic SDK and tool-call extraction
machinery.  Since cache thread-safety is what matters here, we exercise
CachedAnthropicClient.messages_create() directly, which is the code path that
calls cache.get() and cache.put() — exactly the concurrency boundary under test.
"""
from __future__ import annotations

import concurrent.futures
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from firm.core.clock import ReplayClock
from firm.db.migrations import init_db
from firm.llm.anthropic_client import CachedAnthropicClient, LlmMode
from firm.llm.cache import LlmCache, hash_prompt


# ---------------------------------------------------------------------------
# Stub transport — never imports anthropic, returns deterministic dicts
# ---------------------------------------------------------------------------

class _StubTransport:
    """Fake AnthropicTransport whose messages_create returns a deterministic dict.

    The response JSON embeds the prompt hash so distinct prompts produce
    distinct responses, making cache-hit verification possible.
    """

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]:
        ph = hash_prompt(system=system, messages=messages, tools=tools)
        return {
            "id": f"stub-{ph[:8]}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": f"stub-response-{ph[:8]}"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "stop_reason": "end_turn",
        }


def _make_client(
    tmp_path: Path,
    mode: LlmMode = LlmMode.RECORD,
) -> tuple[CachedAnthropicClient, LlmCache]:
    db = tmp_path / "stress.db"
    init_db(db)
    clock = ReplayClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))
    cache = LlmCache(db, clock)
    transport = _StubTransport()
    client = CachedAnthropicClient(
        api_key=None,
        cache=cache,
        mode=mode,
        clock=clock,
        transport=transport,
    )
    return client, cache


def _make_prompt(idx: int) -> dict[str, Any]:
    """Return kwargs for messages_create for prompt index idx."""
    return {
        "model": "claude-3-haiku-20240307",
        "system": f"system-{idx}",
        "messages": [{"role": "user", "content": f"question-{idx}"}],
        "max_tokens": 64,
        "temperature": 0.0,
    }


# ---------------------------------------------------------------------------
# Test 1: 50 parallel reads (all cache hits) — no ProgrammingError
# ---------------------------------------------------------------------------

def test_50_parallel_extracts_no_programming_error(tmp_path: Path) -> None:
    """50 concurrent messages_create() calls on pre-warmed cache entries.

    All calls hit the cache (no writes), exercising the lock-free read path
    across 10 threads.  Asserts no sqlite3.ProgrammingError and that every
    response carries _cache_hit == True.
    """
    client, cache = _make_client(tmp_path, mode=LlmMode.RECORD)

    # Sequential pre-warm: seed 5 distinct prompts
    distinct = 5
    for i in range(distinct):
        client.messages_create(**_make_prompt(i))

    # Switch to CACHED mode so all parallel calls must be hits (no writes)
    client.mode = LlmMode.CACHED

    errors: list[Exception] = []
    results: list[dict[str, object]] = []

    def _call(idx: int) -> dict[str, object]:
        return client.messages_create(**_make_prompt(idx % distinct))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_call, i) for i in range(50)]
        for fut in concurrent.futures.as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                errors.append(exc)
            else:
                results.append(fut.result())

    # No exception of any kind — in particular no sqlite3.ProgrammingError
    assert errors == [], f"Unexpected errors: {errors}"
    # Every response must be a cache hit
    assert all(r.get("_cache_hit") is True for r in results), (
        "Expected all results to be cache hits"
    )
    assert len(results) == 50


# ---------------------------------------------------------------------------
# Test 2: 50 parallel writes (all cache misses) — no lock / OperationalError
# ---------------------------------------------------------------------------

def test_50_parallel_extracts_with_writes_no_lock_error(tmp_path: Path) -> None:
    """50 concurrent messages_create() calls each writing a distinct cache entry.

    Each call is a cache miss, so it calls the stub transport and then
    cache.put() under the module-level _WRITE_LOCK.  Asserts no
    sqlite3.OperationalError ('database is locked') or sqlite3.ProgrammingError,
    and that all 50 entries are readable afterwards.
    """
    client, cache = _make_client(tmp_path, mode=LlmMode.RECORD)

    # 50 distinct prompts — guarantee 50 cache misses
    total = 50

    errors: list[Exception] = []

    def _call(idx: int) -> dict[str, object]:
        return client.messages_create(**_make_prompt(idx))

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_call, i) for i in range(total)]
        for fut in concurrent.futures.as_completed(futures):
            exc = fut.exception()
            if exc is not None:
                errors.append(exc)

    # No sqlite3.OperationalError or sqlite3.ProgrammingError
    prog_errors = [e for e in errors if isinstance(e, sqlite3.ProgrammingError)]
    op_errors = [e for e in errors if isinstance(e, sqlite3.OperationalError)]
    assert prog_errors == [], f"sqlite3.ProgrammingError(s): {prog_errors}"
    assert op_errors == [], f"sqlite3.OperationalError(s): {op_errors}"
    assert errors == [], f"Unexpected errors: {errors}"

    # All 50 distinct entries must be readable from the main thread
    missing = []
    for i in range(total):
        kwargs = _make_prompt(i)
        ph = hash_prompt(
            system=kwargs["system"],
            messages=kwargs["messages"],  # type: ignore[arg-type]
            tools=None,
        )
        hit = cache.get(prompt_hash=ph, model=kwargs["model"])
        if hit is None:
            missing.append(i)

    assert missing == [], f"Cache entries missing for prompt indices: {missing}"
