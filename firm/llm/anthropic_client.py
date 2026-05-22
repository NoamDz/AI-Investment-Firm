"""Cached, mode-aware Anthropic SDK wrapper. See Plan 2 §T16.

Three modes are supported via :class:`LlmMode`:

* ``LIVE`` -- always call the real Anthropic API; never write to cache. Useful
  for ad-hoc experiments outside of the deterministic eval loop.
* ``CACHED`` -- only serve from the local SQLite cache. Any cache miss raises
  :class:`LlmCacheMissError`. This is the default mode for replay; it allows
  running the full firm pipeline with no API key.
* ``RECORD`` -- cache-first; on miss, call the real API and write the response
  back to the cache. Used to populate fixtures during a single eval run.

The class also implements the T9 :class:`firm.llm.client.AnthropicClient`
Protocol via :meth:`complete`, so existing callers (e.g. ``ContextualAugmenter``)
keep working unchanged. New T18 callers can use :meth:`messages_create`
directly to get the raw dict response, which is required to preserve the
Citations API content-block shape (``citations`` arrays inside text blocks).

The Anthropic SDK is *lazy-imported* inside the default transport so that this
module can be imported and exercised in CACHED mode with no API key and no
``anthropic`` install hiccups in CI.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

from opentelemetry import trace

from firm.core.clock import Clock
from firm.llm.cache import LlmCache, hash_prompt
from firm.llm.client import CompletionResponse
from firm.llm.cost import extract_cost_fields, get_router_config


class LlmCacheMissError(Exception):
    """Raised in CACHED mode when the requested prompt is not in the cache."""


class LlmMode(StrEnum):
    LIVE = "live"
    CACHED = "cached"
    RECORD = "record"


@runtime_checkable
class AnthropicTransport(Protocol):
    """Minimal transport boundary so tests can inject a fake SDK client."""

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]: ...


class _AnthropicSdkTransport:
    """Default transport backed by the real ``anthropic`` Python SDK.

    The SDK is imported lazily inside ``__init__`` so importing this module
    does not require ``anthropic`` to be installed (CACHED mode needs no SDK).
    """

    def __init__(self, api_key: str) -> None:
        import anthropic  # lazy import

        self._sdk_client = anthropic.Anthropic(api_key=api_key)

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
        kwargs: dict[str, object] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools
        response = self._sdk_client.messages.create(**kwargs)  # type: ignore[call-overload]
        # Pydantic v2 model -> plain dict. ``mode="python"`` preserves nested
        # dicts/lists (including the Citations API ``citations`` arrays) as
        # native Python objects rather than re-serializing to JSON strings.
        dumped: dict[str, object] = response.model_dump(mode="python")
        return dumped


class CachedAnthropicClient:
    """Cached, mode-aware Anthropic client.

    Implements the T9 :class:`firm.llm.client.AnthropicClient` Protocol via
    :meth:`complete`, and additionally exposes :meth:`messages_create` for
    callers that need the raw response dict (T18 cited-claim extraction).

    Concurrency model
    -----------------
    The injected :class:`~firm.llm.cache.LlmCache` uses one SQLite connection
    per thread, opened lazily via ``threading.local()``.  Reads (``cache.get``)
    are lock-free; writes (``cache.put``, which occur only on RECORD-mode cache
    misses) acquire a module-level ``threading.Lock`` so concurrent writers
    across N threads do not surface ``"database is locked"`` SQLite errors.
    The underlying DB is WAL-mode, so concurrent reads and a single write do
    not block each other beyond the write lock.  This concurrency model is
    stress-tested in ``tests/integration/test_cached_client_threadsafety.py``
    with 50 parallel ``messages_create()`` calls across 10 threads.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        cache: LlmCache,
        mode: LlmMode,
        clock: Clock,
        transport: AnthropicTransport | None = None,
    ) -> None:
        self._cache = cache
        self.mode = mode
        self._clock = clock

        resolved_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")

        if transport is None:
            if mode in (LlmMode.LIVE, LlmMode.RECORD):
                if not resolved_key:
                    raise ValueError(
                        "ANTHROPIC_API_KEY required for LIVE/RECORD mode "
                        "(set the env var or pass api_key explicitly)"
                    )
                self._transport: AnthropicTransport = _AnthropicSdkTransport(resolved_key)
            else:
                # CACHED mode without an injected transport: lazily construct
                # only if a live call is ever attempted (it should not be).
                self._transport = _LazyMissingTransport()
        else:
            self._transport = transport

    @classmethod
    def from_env(cls, *, cache: LlmCache, clock: Clock) -> CachedAnthropicClient:
        """Construct a client honoring the ``FIRM_LLM_MODE`` env var.

        Defaults to ``cached`` so a missing env var yields deterministic replay
        semantics. Unknown values fall back to ``cached`` rather than crashing.
        """
        raw = os.environ.get("FIRM_LLM_MODE", "cached").strip().lower()
        try:
            mode = LlmMode(raw)
        except ValueError:
            mode = LlmMode.CACHED
        return cls(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            cache=cache,
            mode=mode,
            clock=clock,
        )

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        """Return the raw Anthropic response dict, cached per (prompt_hash, model).

        The returned dict carries a synthetic ``_cache_hit`` boolean so callers
        (notably :meth:`complete`) can report cache provenance via
        :class:`CompletionResponse`.

        Side-effect: stamps the currently-active OTel span (set by the
        enclosing :func:`firm.obs.llm_span`) with the token / cost attributes
        from the spec §10.1 schema.  See :meth:`_stamp_llm_cost`.
        """
        prompt_hash = hash_prompt(system=system, messages=messages, tools=tools)

        if self.mode == LlmMode.CACHED:
            cached = self._cache.get(prompt_hash=prompt_hash, model=model)
            if cached is None:
                raise LlmCacheMissError(
                    f"No cache entry for prompt_hash={prompt_hash[:12]}... model={model}"
                )
            raw = json.loads(cached.response_json)
            assert isinstance(raw, dict)
            raw["_cache_hit"] = True
            _stamp_llm_cost(raw, model=model)
            return raw

        if self.mode == LlmMode.LIVE:
            raw = self._transport.messages_create(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            raw["_cache_hit"] = False
            _stamp_llm_cost(raw, model=model)
            return raw

        # RECORD: cache-first, then live + write.
        cached = self._cache.get(prompt_hash=prompt_hash, model=model)
        if cached is not None:
            raw = json.loads(cached.response_json)
            assert isinstance(raw, dict)
            raw["_cache_hit"] = True
            _stamp_llm_cost(raw, model=model)
            return raw

        raw = self._transport.messages_create(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        usage_raw = raw.get("usage", {})
        usage = usage_raw if isinstance(usage_raw, dict) else {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        self._cache.put(
            prompt_hash=prompt_hash,
            model=model,
            response_json=json.dumps(raw),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        raw["_cache_hit"] = False
        _stamp_llm_cost(raw, model=model)
        return raw

    def complete(
        self,
        *,
        model: str,
        system: str | None,
        messages: Sequence[dict[str, object]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> CompletionResponse:
        """T9-Protocol method: returns a :class:`CompletionResponse`.

        Concatenates all ``text``-type content blocks into a single string. For
        callers that need the structured response (e.g. Citations API blocks),
        use :meth:`messages_create` directly.
        """
        raw = self.messages_create(
            model=model,
            system=system if system is not None else "",
            messages=list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text_parts: list[str] = []
        content = raw.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_val = block.get("text", "")
                    if isinstance(text_val, str):
                        text_parts.append(text_val)
        text = "".join(text_parts)

        usage_raw = raw.get("usage", {})
        usage = usage_raw if isinstance(usage_raw, dict) else {}
        cache_hit_val = raw.get("_cache_hit", False)
        return CompletionResponse(
            text=text,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            cache_hit=bool(cache_hit_val),
        )


class _LazyMissingTransport:
    """Placeholder transport used in CACHED mode when no real SDK is wired.

    Any attempt to call the live API in CACHED mode is a logic error; this
    transport raises a descriptive exception instead of failing deep in the SDK.
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
        raise RuntimeError(
            "CachedAnthropicClient is in CACHED mode but a live call was attempted. "
            "This indicates a missing cache entry; check FIRM_LLM_MODE / the prompt hash."
        )


def _stamp_llm_cost(
    raw: dict[str, object], *, model: str
) -> None:
    """Write token / cost attributes onto the currently active OTel span.

    Field convention (spec §10.1):

    * **cache hit**  -> ``cached_tokens`` = input + output (the work the cache
      saved), ``cost_usd`` = 0.0.  ``input_tokens`` / ``output_tokens`` are
      *not* written, because those fields imply a real, billed call.
    * **live call**  -> ``input_tokens`` / ``output_tokens`` from the response,
      ``cost_usd`` computed via the ``config/router.yaml`` rate card.
      ``cached_tokens`` is not written.

    No-op (no exception) when called outside any :func:`firm.obs.llm_span`
    context — e.g. directly from a unit test or a startup probe.  This keeps
    the wrapper safe to call from any callsite without requiring a span to
    be active.

    Implementation note: delegates to :func:`firm.llm.cost.extract_cost_fields`
    so the (cache-hit / live) -> field-set mapping cannot drift from the T09
    cost-ledger writer.  Cache-hit provenance is read from ``raw["_cache_hit"]``
    directly; the caller attaches it before invoking this function.
    """
    span = trace.get_current_span()
    if not span.is_recording():
        return

    fields = extract_cost_fields(raw, model=model, router_cfg=get_router_config())

    if fields.cached_tokens is not None:
        span.set_attribute("cached_tokens", fields.cached_tokens)
        span.set_attribute("cost_usd", fields.cost_usd)
        return

    # Live call branch — input/output tokens are ints (extract_cost_fields
    # only nulls them on cache hit).
    assert fields.input_tokens is not None
    assert fields.output_tokens is not None
    span.set_attribute("input_tokens", fields.input_tokens)
    span.set_attribute("output_tokens", fields.output_tokens)
    span.set_attribute("cost_usd", fields.cost_usd)
