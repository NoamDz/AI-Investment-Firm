"""Tests for the cassette layer (T01).

Exercises CassetteClient in all three modes (LIVE, RECORD, REPLAY) using a
FakeAnthropicTransport spy so no network calls are made.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from firm.llm.cassettes import CassetteClient, CassetteMissError, CassetteMode


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


class FakeAnthropicTransport:
    """Recording fake transport that returns a canned response dict per call."""

    def __init__(self, response: dict[str, object]) -> None:
        self._response = response
        self.calls: list[dict[str, object]] = []

    def messages_create(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return dict(self._response)


def _basic_response() -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": "hello world"}],
        "usage": {"input_tokens": 12, "output_tokens": 7},
    }


def _make_client(
    transport: FakeAnthropicTransport,
    mode: CassetteMode,
    cassette_dir: Path,
) -> CassetteClient:
    return CassetteClient(
        real_transport=transport,
        mode=mode,
        cassette_dir=cassette_dir,
    )


_DEFAULT_ARGS: dict[str, Any] = {
    "model": "claude-haiku-4-5",
    "system": "you are an analyst",
    "messages": [{"role": "user", "content": "What is the market cap?"}],
    "tools": None,
    "max_tokens": 256,
    "temperature": 0.0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_live_mode_passthrough_no_recording(tmp_path: Path) -> None:
    """LIVE mode forwards to real transport and writes no cassette files."""
    transport = FakeAnthropicTransport(_basic_response())
    client = _make_client(transport, CassetteMode.LIVE, tmp_path)

    result = client.messages_create(**_DEFAULT_ARGS)

    assert result["content"] == _basic_response()["content"]
    assert len(transport.calls) == 1
    yaml_files = list(tmp_path.glob("*.yaml"))
    assert yaml_files == [], f"Expected no cassette files, found: {yaml_files}"


def test_record_then_replay_round_trip(tmp_path: Path) -> None:
    """RECORD writes a cassette; REPLAY returns an identical response without calling real transport."""
    # Phase 1: RECORD
    canned = _basic_response()
    record_transport = FakeAnthropicTransport(canned)
    record_client = _make_client(record_transport, CassetteMode.RECORD, tmp_path)

    recorded = record_client.messages_create(**_DEFAULT_ARGS)

    assert len(record_transport.calls) == 1
    yaml_files = list(tmp_path.glob("*.yaml"))
    assert len(yaml_files) == 1, "Expected exactly one cassette file after RECORD"

    # Phase 2: REPLAY — fresh client, fresh (unused) transport
    replay_transport = FakeAnthropicTransport({"content": [{"type": "text", "text": "WRONG"}]})
    replay_client = _make_client(replay_transport, CassetteMode.REPLAY, tmp_path)

    replayed = replay_client.messages_create(**_DEFAULT_ARGS)

    # Real transport must NOT be called in replay mode.
    assert replay_transport.calls == [], "REPLAY mode must not call the real transport"
    # Response must match the original.
    assert replayed["content"] == recorded["content"]
    assert replayed["usage"] == recorded["usage"]


def test_replay_miss_raises(tmp_path: Path) -> None:
    """REPLAY with an empty cassette dir raises CassetteMissError."""
    transport = FakeAnthropicTransport(_basic_response())
    client = _make_client(transport, CassetteMode.REPLAY, tmp_path)

    with pytest.raises(CassetteMissError) as exc_info:
        client.messages_create(**_DEFAULT_ARGS)

    assert "claude-haiku-4-5" in str(exc_info.value)
    # Real transport must NOT be called.
    assert transport.calls == []


def test_prompt_drift_forces_replay_miss(tmp_path: Path) -> None:
    """Changing messages causes a cassette miss in REPLAY mode (prompt drift surfaces)."""
    # Record with original messages.
    transport = FakeAnthropicTransport(_basic_response())
    record_client = _make_client(transport, CassetteMode.RECORD, tmp_path)
    record_client.messages_create(**_DEFAULT_ARGS)

    # Replay with a mutated messages list (one character changed).
    drifted_args = dict(_DEFAULT_ARGS)
    drifted_args["messages"] = [{"role": "user", "content": "What is the market cap?!"}]

    replay_transport = FakeAnthropicTransport(_basic_response())
    replay_client = _make_client(replay_transport, CassetteMode.REPLAY, tmp_path)

    with pytest.raises(CassetteMissError):
        replay_client.messages_create(**drifted_args)


def test_cassette_key_includes_model(tmp_path: Path) -> None:
    """A different model name produces a distinct cassette key → REPLAY miss."""
    # Record with haiku.
    transport = FakeAnthropicTransport(_basic_response())
    record_client = _make_client(transport, CassetteMode.RECORD, tmp_path)
    record_client.messages_create(**_DEFAULT_ARGS)

    # Replay with sonnet (same prompt, different model).
    sonnet_args = dict(_DEFAULT_ARGS)
    sonnet_args["model"] = "claude-sonnet-4-6"

    replay_transport = FakeAnthropicTransport(_basic_response())
    replay_client = _make_client(replay_transport, CassetteMode.REPLAY, tmp_path)

    with pytest.raises(CassetteMissError) as exc_info:
        replay_client.messages_create(**sonnet_args)

    assert "claude-sonnet-4-6" in str(exc_info.value)


def test_record_overwrites_existing_cassette(tmp_path: Path) -> None:
    """Re-recording with the same args but a different response overwrites the cassette."""
    first_response: dict[str, Any] = {
        "content": [{"type": "text", "text": "first response"}],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }
    second_response: dict[str, Any] = {
        "content": [{"type": "text", "text": "second response"}],
        "usage": {"input_tokens": 5, "output_tokens": 4},
    }

    # First RECORD call.
    transport1 = FakeAnthropicTransport(first_response)
    client1 = _make_client(transport1, CassetteMode.RECORD, tmp_path)
    client1.messages_create(**_DEFAULT_ARGS)

    # Second RECORD call — different canned response, same args.
    transport2 = FakeAnthropicTransport(second_response)
    client2 = _make_client(transport2, CassetteMode.RECORD, tmp_path)
    client2.messages_create(**_DEFAULT_ARGS)

    # Only one cassette file should exist.
    yaml_files = list(tmp_path.glob("*.yaml"))
    assert len(yaml_files) == 1

    # REPLAY should return the *second* response (overwrite semantics).
    replay_transport = FakeAnthropicTransport(_basic_response())
    replay_client = _make_client(replay_transport, CassetteMode.REPLAY, tmp_path)
    replayed = replay_client.messages_create(**_DEFAULT_ARGS)

    assert replayed["content"] == second_response["content"]
