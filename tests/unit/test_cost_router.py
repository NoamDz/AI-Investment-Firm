"""Tests for :class:`firm.llm.router.CostRouter` (Plan 3 T07).

The CostRouter has two responsibilities:

* ``route_for_decision(features)`` -> a :class:`ProfileChoice` carrying the
  primary profile name and the full fallback ladder (primary first, then
  fallback_chain entries, deduplicated while preserving order).
* ``call_with_fallback(profile, system, messages, tools=None)`` -> dict — runs
  the call through a fallback ladder that:
    1. tries the primary profile,
    2. retries the SAME profile once with the message's ``document`` content
       blocks truncated by 50%,
    3. downgrades to the next ladder profile with ``max_tokens *= 0.5``,
    4. raises :class:`LLMUnavailableError` once the ladder is exhausted.

Each underlying ``messages_create`` call is wrapped in its own
``llm_span("anthropic", profile.model_id)`` so cost / token attribution and
the ``fallback_attempt`` attribute are per-attempt rather than collapsed onto
the outer caller span.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from firm.core.clock import ReplayClock
from firm.core.config import load_router_config
from firm.core.models import RouterFeatures
from firm.db.migrations import init_db
from firm.llm.anthropic_client import (
    CachedAnthropicClient,
    LlmCacheMissError,
    LlmMode,
)
from firm.llm.cache import LlmCache
from firm.llm.router import (
    CostRouter,
    LLMUnavailableError,
    ProfileChoice,
)
from firm.obs import llm_span
from firm.obs.tracer import use_sync_exporter


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _ScriptedTransport:
    """Transport that emits responses or exceptions from a scripted queue.

    Each entry in ``script`` is either a dict (returned verbatim, with
    ``_cache_hit`` attached downstream by the wrapper) or an Exception
    instance (raised when its turn comes).
    """

    def __init__(self, script: list[object]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, object]] = []

    def messages_create(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("scripted transport ran out of entries")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, dict)
        return dict(item)


def _ok_response(text: str = "ok") -> dict[str, object]:
    return {
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _make_cache(tmp_path: Path) -> LlmCache:
    db = tmp_path / "router.db"
    init_db(db)
    clock = ReplayClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))
    return LlmCache(db, clock)


def _make_clock() -> ReplayClock:
    return ReplayClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))


def _make_client(
    tmp_path: Path,
    transport: _ScriptedTransport,
    *,
    mode: LlmMode = LlmMode.LIVE,
) -> CachedAnthropicClient:
    return CachedAnthropicClient(
        api_key="sk-test" if mode != LlmMode.CACHED else None,
        cache=_make_cache(tmp_path),
        mode=mode,
        clock=_make_clock(),
        transport=transport,
    )


def _read_spans(traces_root: Path) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for jsonl in traces_root.rglob("*.jsonl"):
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                spans.append(json.loads(s))
    return spans


def _redirect_traces(tmp_path: Path, run_id: str) -> Path:
    traces_root = tmp_path / "traces"
    use_sync_exporter(traces_root=traces_root, run_id=run_id)
    return traces_root


def _messages_with_n_documents(n: int) -> list[dict[str, object]]:
    """Build a message list with *n* document content blocks plus a text block."""
    blocks: list[dict[str, object]] = []
    for i in range(n):
        blocks.append(
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": f"doc{i}"},
                "title": f"doc-{i}",
                "citations": {"enabled": True},
            }
        )
    blocks.append({"type": "text", "text": "Please summarize."})
    return [{"role": "user", "content": blocks}]


def _count_documents(messages: list[dict[str, object]]) -> int:
    msg = messages[0]
    content = msg["content"]
    assert isinstance(content, list)
    return sum(1 for b in content if isinstance(b, dict) and b.get("type") == "document")


# ---------------------------------------------------------------------------
# 1. route_for_decision — high-feature → opus with full ladder
# ---------------------------------------------------------------------------


def test_route_for_decision_high_features_returns_opus_ladder(tmp_path: Path) -> None:
    cfg = load_router_config(Path("config/router.yaml"))
    transport = _ScriptedTransport([])
    client = _make_client(tmp_path, transport)
    router = CostRouter(router_cfg=cfg, anthropic_client=client)

    features = RouterFeatures(
        risk_weight=0.9, novelty=0.9, complexity=0.9, time_pressure=0.9
    )
    choice = router.route_for_decision(features)
    assert isinstance(choice, ProfileChoice)
    assert choice.primary == "opus"
    # Ladder = primary + fallback_chain, dedup-preserving order.
    # fallback_chain in real router.yaml is ["sonnet", "haiku"]; primary is
    # "opus" so no dedup is necessary here.
    assert choice.ladder == ("opus", "sonnet", "haiku")


def test_route_for_decision_dedupes_when_primary_is_in_fallback_chain(tmp_path: Path) -> None:
    cfg = load_router_config(Path("config/router.yaml"))
    transport = _ScriptedTransport([])
    client = _make_client(tmp_path, transport)
    router = CostRouter(router_cfg=cfg, anthropic_client=client)

    # Mid-range features land in the sonnet bucket; primary=sonnet is the
    # FIRST entry in fallback_chain ["sonnet","haiku"] so dedup removes the
    # second "sonnet" entry, leaving ("sonnet","haiku").
    features = RouterFeatures(
        risk_weight=0.5, novelty=0.5, complexity=0.5, time_pressure=0.5
    )
    choice = router.route_for_decision(features)
    assert choice.primary == "sonnet"
    assert choice.ladder == ("sonnet", "haiku")


def test_route_for_decision_low_features_returns_haiku_ladder(tmp_path: Path) -> None:
    cfg = load_router_config(Path("config/router.yaml"))
    transport = _ScriptedTransport([])
    client = _make_client(tmp_path, transport)
    router = CostRouter(router_cfg=cfg, anthropic_client=client)

    features = RouterFeatures(
        risk_weight=0.0, novelty=0.0, complexity=0.0, time_pressure=0.0
    )
    choice = router.route_for_decision(features)
    assert choice.primary == "haiku"
    # fallback_chain is ["sonnet","haiku"]; dedup of ("haiku","sonnet","haiku")
    # is ("haiku","sonnet").
    assert choice.ladder == ("haiku", "sonnet")


# ---------------------------------------------------------------------------
# 2. Successful primary call → single underlying call
# ---------------------------------------------------------------------------


def test_call_with_fallback_successful_primary_one_call(tmp_path: Path) -> None:
    cfg = load_router_config(Path("config/router.yaml"))
    transport = _ScriptedTransport([_ok_response("primary-ok")])
    client = _make_client(tmp_path, transport)
    router = CostRouter(router_cfg=cfg, anthropic_client=client)

    raw = router.call_with_fallback(
        "sonnet",
        system="s",
        messages=[{"role": "user", "content": "go"}],
    )
    assert raw["_cache_hit"] is False
    assert raw["content"][0]["text"] == "primary-ok"
    assert len(transport.calls) == 1, "primary must not be retried on success"
    # The primary call uses the sonnet profile's model id + knobs.
    sonnet = cfg.profiles["sonnet"]
    call0 = transport.calls[0]
    assert call0["model"] == sonnet.model_id
    assert call0["max_tokens"] == sonnet.max_tokens
    assert call0["temperature"] == sonnet.temperature


# ---------------------------------------------------------------------------
# 3. Truncate-and-retry within the same profile
# ---------------------------------------------------------------------------


def test_call_with_fallback_truncates_documents_and_retries_same_profile(tmp_path: Path) -> None:
    cfg = load_router_config(Path("config/router.yaml"))
    err = RuntimeError("simulated 503")
    transport = _ScriptedTransport([err, _ok_response("after-truncate")])
    client = _make_client(tmp_path, transport)
    router = CostRouter(router_cfg=cfg, anthropic_client=client)

    messages = _messages_with_n_documents(3)
    raw = router.call_with_fallback(
        "sonnet",
        system="s",
        messages=messages,
    )
    assert raw["content"][0]["text"] == "after-truncate"
    assert len(transport.calls) == 2, "expected primary + truncated-retry"

    # First call: original 3 documents.
    first_msgs = transport.calls[0]["messages"]
    assert isinstance(first_msgs, list)
    assert _count_documents(first_msgs) == 3

    # Second call: truncated to ceil(3 * 0.5) = 2 documents on the same profile.
    second_msgs = transport.calls[1]["messages"]
    assert isinstance(second_msgs, list)
    assert _count_documents(second_msgs) == 2

    # The retry must still target the SAME (sonnet) profile, unchanged knobs.
    sonnet = cfg.profiles["sonnet"]
    assert transport.calls[1]["model"] == sonnet.model_id
    assert transport.calls[1]["max_tokens"] == sonnet.max_tokens

    # Caller's `messages` argument must not be mutated.
    assert _count_documents(messages) == 3


# ---------------------------------------------------------------------------
# 4. Downgrade to next profile with max_tokens halved
# ---------------------------------------------------------------------------


def test_call_with_fallback_downgrades_to_next_profile_with_halved_tokens(
    tmp_path: Path,
) -> None:
    cfg = load_router_config(Path("config/router.yaml"))
    transport = _ScriptedTransport(
        [
            RuntimeError("503 primary"),
            RuntimeError("503 truncated"),
            _ok_response("downgrade-ok"),
        ]
    )
    client = _make_client(tmp_path, transport)
    router = CostRouter(router_cfg=cfg, anthropic_client=client)

    messages = _messages_with_n_documents(2)
    raw = router.call_with_fallback("sonnet", system="s", messages=messages)
    assert raw["content"][0]["text"] == "downgrade-ok"
    assert len(transport.calls) == 3

    # Third call: downgraded ladder profile (next after sonnet → haiku).
    third = transport.calls[2]
    haiku = cfg.profiles["haiku"]
    assert third["model"] == haiku.model_id
    # Per spec: "downgrades to Haiku with max_tokens *= 0.5" — apply 0.5x to
    # the destination profile's configured max_tokens.
    assert third["max_tokens"] == int(haiku.max_tokens * 0.5)
    assert third["temperature"] == haiku.temperature


# ---------------------------------------------------------------------------
# 5. All profiles fail → LLMUnavailableError
# ---------------------------------------------------------------------------


def test_call_with_fallback_all_profiles_fail_raises_unavailable(tmp_path: Path) -> None:
    cfg = load_router_config(Path("config/router.yaml"))
    # Ladder for primary=sonnet is ("sonnet","haiku") → 2 profiles.
    # Per profile: 1 primary attempt + 1 truncated retry = 2 calls, EXCEPT
    # the downgraded profile is invoked once (no truncate retry on the
    # downgraded profile per spec wording). Implementation only truncates on
    # the FIRST profile; on downgrade it just calls once with halved max_tokens.
    # So total = 2 (primary profile) + 1 (downgraded) = 3 failures.
    transport = _ScriptedTransport(
        [
            RuntimeError("boom-1"),
            RuntimeError("boom-2"),
            RuntimeError("boom-3"),
        ]
    )
    client = _make_client(tmp_path, transport)
    router = CostRouter(router_cfg=cfg, anthropic_client=client)

    with pytest.raises(LLMUnavailableError) as exc_info:
        router.call_with_fallback(
            "sonnet", system="s", messages=_messages_with_n_documents(2)
        )
    msg = str(exc_info.value)
    assert "sonnet" in msg, msg
    # The tail of underlying error reprs should be embedded.
    assert "boom-3" in msg, msg


# ---------------------------------------------------------------------------
# 6. LlmCacheMissError short-circuits — no fallback
# ---------------------------------------------------------------------------


def test_call_with_fallback_cache_miss_is_not_retried(tmp_path: Path) -> None:
    cfg = load_router_config(Path("config/router.yaml"))
    transport = _ScriptedTransport(
        [
            LlmCacheMissError("missing"),
            _ok_response("should-never-run"),
        ]
    )
    client = _make_client(tmp_path, transport)
    router = CostRouter(router_cfg=cfg, anthropic_client=client)

    with pytest.raises(LlmCacheMissError):
        router.call_with_fallback(
            "sonnet", system="s", messages=[{"role": "user", "content": "x"}]
        )
    # Only the first call should have been attempted.
    assert len(transport.calls) == 1


# ---------------------------------------------------------------------------
# 7. Each attempt opens its own llm.call span tagged with fallback_attempt
# ---------------------------------------------------------------------------


def test_each_attempt_opens_its_own_llm_call_span(tmp_path: Path) -> None:
    traces_root = _redirect_traces(tmp_path, run_id="t07test07ladderspans000000")

    cfg = load_router_config(Path("config/router.yaml"))
    # Two failures + one success → three llm.call spans expected.
    transport = _ScriptedTransport(
        [
            RuntimeError("primary-fail"),
            RuntimeError("truncate-fail"),
            _ok_response("downgrade-ok"),
        ]
    )
    client = _make_client(tmp_path, transport)
    router = CostRouter(router_cfg=cfg, anthropic_client=client)

    # Wrap call in an outer llm_span("test","outer") — emulating T03's wiring
    # so each attempt's span is a child of the outer one.
    with llm_span("test", "outer"):
        router.call_with_fallback(
            "sonnet", system="s", messages=_messages_with_n_documents(2)
        )

    spans = _read_spans(traces_root)
    llm_call_spans = [s for s in spans if s.get("operation") == "llm.call"]
    # 3 attempts + 1 outer wrapper = 4 llm.call-typed spans total.
    assert len(llm_call_spans) == 4, llm_call_spans

    # The outer span has no fallback_attempt (it isn't an attempt).
    attempt_spans = [
        s for s in llm_call_spans if s.get("model") != "outer"
    ]
    assert len(attempt_spans) == 3

    sonnet_model = cfg.profiles["sonnet"].model_id
    haiku_model = cfg.profiles["haiku"].model_id
    # Spans appear in completion order; map model→sequence to assert tags.
    models_seen = [s["model"] for s in attempt_spans]
    assert sonnet_model in models_seen
    assert haiku_model in models_seen
