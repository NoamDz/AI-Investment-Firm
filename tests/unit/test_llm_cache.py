"""Tests for the SQLite-backed LLM cache. See Plan 2 §T15."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from firm.core.clock import ReplayClock
from firm.db.migrations import init_db
from firm.llm.cache import CachedResponse, LlmCache, hash_prompt


def _make_cache(tmp_path: Path) -> tuple[LlmCache, Path]:
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))
    return LlmCache(db, clock), db


def test_cache_miss_then_hit(tmp_path: Path) -> None:
    cache, _ = _make_cache(tmp_path)

    assert cache.get(prompt_hash="abc", model="sonnet") is None

    cache.put(
        prompt_hash="abc",
        model="sonnet",
        response_json='{"text": "hello"}',
        input_tokens=10,
        output_tokens=5,
    )

    hit = cache.get(prompt_hash="abc", model="sonnet")
    assert hit is not None
    assert isinstance(hit, CachedResponse)
    assert hit.response_json == '{"text": "hello"}'
    assert hit.input_tokens == 10
    assert hit.output_tokens == 5
    # created_at must be tz-aware UTC matching the ReplayClock.
    assert hit.created_at == datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)


def test_cache_key_includes_model_id(tmp_path: Path) -> None:
    cache, _ = _make_cache(tmp_path)

    cache.put(
        prompt_hash="shared",
        model="sonnet",
        response_json='{"text": "from-sonnet"}',
        input_tokens=1,
        output_tokens=1,
    )
    cache.put(
        prompt_hash="shared",
        model="haiku",
        response_json='{"text": "from-haiku"}',
        input_tokens=2,
        output_tokens=2,
    )

    sonnet = cache.get(prompt_hash="shared", model="sonnet")
    haiku = cache.get(prompt_hash="shared", model="haiku")
    assert sonnet is not None and haiku is not None
    assert sonnet.response_json == '{"text": "from-sonnet"}'
    assert haiku.response_json == '{"text": "from-haiku"}'
    # Distinct token counts confirm rows are not shadowed across models.
    assert sonnet.input_tokens == 1
    assert haiku.input_tokens == 2


def test_cache_invalidates_on_prompt_change(tmp_path: Path) -> None:
    cache, _ = _make_cache(tmp_path)

    hash_a = hash_prompt(system="a", messages=[{"role": "user", "content": "x"}], tools=None)
    hash_b = hash_prompt(system="b", messages=[{"role": "user", "content": "x"}], tools=None)
    assert hash_a != hash_b

    cache.put(
        prompt_hash=hash_a,
        model="sonnet",
        response_json='{"text": "A"}',
        input_tokens=0,
        output_tokens=0,
    )

    # The other prompt_hash must be a miss.
    assert cache.get(prompt_hash=hash_b, model="sonnet") is None

    hit_a = cache.get(prompt_hash=hash_a, model="sonnet")
    assert hit_a is not None
    assert hit_a.response_json == '{"text": "A"}'


def test_cache_stores_token_counts(tmp_path: Path) -> None:
    cache, _ = _make_cache(tmp_path)

    cache.put(
        prompt_hash="t",
        model="sonnet",
        response_json="{}",
        input_tokens=100,
        output_tokens=50,
    )

    hit = cache.get(prompt_hash="t", model="sonnet")
    assert hit is not None
    assert hit.input_tokens == 100
    assert hit.output_tokens == 50


def test_hash_prompt_is_canonical() -> None:
    """Different dict key orderings must produce the same hash (sort_keys=True)."""
    h1 = hash_prompt(
        system="sys",
        messages=[{"role": "user", "content": "hi", "name": "x"}],
        tools=[{"a": 1, "b": 2}],
    )
    h2 = hash_prompt(
        system="sys",
        messages=[{"name": "x", "content": "hi", "role": "user"}],
        tools=[{"b": 2, "a": 1}],
    )
    assert h1 == h2

    # And a sanity check: any value change does flip the hash.
    h3 = hash_prompt(
        system="sys",
        messages=[{"role": "user", "content": "different"}],
        tools=[{"a": 1, "b": 2}],
    )
    assert h1 != h3


def test_hash_prompt_changes_on_system_change() -> None:
    msgs = [{"role": "user", "content": "same"}]
    h1 = hash_prompt(system="alpha", messages=msgs, tools=None)
    h2 = hash_prompt(system="beta", messages=msgs, tools=None)
    assert h1 != h2

    # tools=None vs tools=[] must also be distinguishable inputs.
    h3 = hash_prompt(system="alpha", messages=msgs, tools=[])
    assert h1 != h3
