"""Cassette layer for deterministic LLM replay.

This module sits *between* the SQLite cache and the live Anthropic SDK so that
eval runs can be replayed without populating the cache for every call.

Call order (after T02 wiring):
  CachedAnthropicClient → cache lookup → on miss → CassetteClient → real SDK

Cassette key
------------
sha256 of canonical JSON of ``{"model": …, "system": …, "messages": …,
"tools": …}`` with ``sort_keys=True``.  ``max_tokens`` and ``temperature``
are *not* part of the key (they go into the YAML for traceability only).
This means prompt drift — any change to model/system/messages/tools —
surfaces as a hard miss in REPLAY mode rather than silently returning a
stale response.

Modes
-----
* ``LIVE``   — pass-through to the real transport; no file I/O.
* ``RECORD`` — pass-through to the real transport, then write
  ``{cassette_dir}/{key}.yaml`` (overwrites on re-record).
* ``REPLAY`` — compute key, load ``{cassette_dir}/{key}.yaml``; raise
  :class:`CassetteMissError` if absent.
"""
from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from firm.llm.anthropic_client import AnthropicTransport


class CassetteMode(StrEnum):
    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"


class CassetteMissError(Exception):
    """Raised in REPLAY mode when no cassette matches the request."""


def _cassette_key(
    *,
    model: str,
    system: str,
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
) -> str:
    """sha256 hex digest of the canonical request (model + prompt inputs)."""
    canonical = json.dumps(
        {
            "model": model,
            "system": system,
            "messages": messages,
            "tools": tools,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CassetteClient:
    """Anthropic-transport-compatible cassette wrapper.

    Implements :class:`~firm.llm.anthropic_client.AnthropicTransport` so it
    is substitutable wherever ``_AnthropicSdkTransport`` is used.

    Parameters
    ----------
    real_transport:
        The downstream transport to call in LIVE/RECORD modes.
    mode:
        One of :class:`CassetteMode`.
    cassette_dir:
        Directory where ``.yaml`` cassette files are stored.  Created
        lazily on first RECORD write.
    """

    def __init__(
        self,
        *,
        real_transport: AnthropicTransport,
        mode: CassetteMode,
        cassette_dir: Path,
    ) -> None:
        self._real_transport = real_transport
        self._mode = mode
        self._cassette_dir = cassette_dir

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
        """Dispatch according to the current :class:`CassetteMode`."""
        if self._mode == CassetteMode.LIVE:
            return self._real_transport.messages_create(
                model=model,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        key = _cassette_key(
            model=model, system=system, messages=messages, tools=tools
        )
        cassette_path = self._cassette_dir / f"{key}.yaml"

        if self._mode == CassetteMode.REPLAY:
            if not cassette_path.exists():
                raise CassetteMissError(
                    f"no cassette for key={key[:12]}... model={model}"
                )
            data: dict[str, Any] = yaml.safe_load(cassette_path.read_text(encoding="utf-8"))
            response: dict[str, object] = data["response"]
            return response

        # RECORD: call the real transport, then persist.
        raw = self._real_transport.messages_create(
            model=model,
            system=system,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._cassette_dir.mkdir(parents=True, exist_ok=True)
        cassette_content: dict[str, object] = {
            "request": {
                "model": model,
                "system": system,
                "messages": messages,
                "tools": tools,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            "response": raw,
        }
        cassette_path.write_text(
            yaml.safe_dump(cassette_content, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        return raw
