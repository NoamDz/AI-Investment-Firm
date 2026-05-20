"""Test that OTel spans are written to JSONL with all spec §10.1 fields.

TDD: this test is written *before* the implementation; it drives the
``firm.obs.tracer`` module design.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from firm.obs.tracer import get_tracer, init_tracer, use_sync_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {
    "trace_id",
    "span_id",
    "parent_span_id",
    "agent",
    "operation",
    "decision_id",
    "duration_ms",
    "model",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "cost_usd",
    "citations",
    "failure_mode",
    "status",
}


def _read_spans(traces_dir: Path) -> list[dict[str, object]]:
    """Return all span dicts found in any .jsonl file under *traces_dir*."""
    spans: list[dict[str, object]] = []
    for jsonl_file in traces_dir.rglob("*.jsonl"):
        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_span_file_created_and_json_parsable(tmp_path: Path) -> None:
    """A completed span produces at least one JSONL file that is parsable."""
    use_sync_exporter(traces_root=tmp_path, run_id="01TEST000000000000000000000")
    tracer = get_tracer("test-agent")

    with tracer.start_as_current_span("test.operation") as span:
        span.set_attribute("agent", "test-agent")
        span.set_attribute("operation", "test.operation")
        span.set_attribute("decision_id", "dec-001")
        span.set_attribute("model", "claude-haiku-4-5")
        span.set_attribute("input_tokens", 10)
        span.set_attribute("output_tokens", 5)
        span.set_attribute("cached_tokens", 0)
        span.set_attribute("cost_usd", 0.0001)
        span.set_attribute("citations", 0)
        span.set_attribute("failure_mode", "")
        span.set_attribute("status", "ok")

    spans = _read_spans(tmp_path)
    assert len(spans) >= 1, "Expected at least one span in JSONL output"


def test_all_spec_fields_present(tmp_path: Path) -> None:
    """Every field from spec §10.1 must be present in the emitted span JSON."""
    use_sync_exporter(traces_root=tmp_path, run_id="01TEST000000000000000000001")
    tracer = get_tracer("research")

    with tracer.start_as_current_span("llm.call") as span:
        span.set_attribute("agent", "research")
        span.set_attribute("operation", "llm.call")
        span.set_attribute("decision_id", "dec-002")
        span.set_attribute("model", "claude-sonnet-4-6")
        span.set_attribute("input_tokens", 120)
        span.set_attribute("output_tokens", 80)
        span.set_attribute("cached_tokens", 40)
        span.set_attribute("cost_usd", 0.0025)
        span.set_attribute("citations", 3)
        span.set_attribute("failure_mode", "")
        span.set_attribute("status", "ok")

    spans = _read_spans(tmp_path)
    assert spans, "No spans were written to JSONL"

    span_obj = spans[-1]  # last span written
    missing = _REQUIRED_FIELDS - set(span_obj.keys())
    assert not missing, f"Span missing spec §10.1 fields: {missing!r}"


def test_span_fields_have_correct_types(tmp_path: Path) -> None:
    """Numeric fields are numeric, string fields are strings after JSON round-trip."""
    use_sync_exporter(traces_root=tmp_path, run_id="01TEST000000000000000000002")
    tracer = get_tracer("pm")

    with tracer.start_as_current_span("vote.aggregate") as span:
        span.set_attribute("agent", "pm")
        span.set_attribute("operation", "vote.aggregate")
        span.set_attribute("decision_id", "dec-003")
        span.set_attribute("model", "claude-sonnet-4-6")
        span.set_attribute("input_tokens", 200)
        span.set_attribute("output_tokens", 150)
        span.set_attribute("cached_tokens", 50)
        span.set_attribute("cost_usd", 0.0042)
        span.set_attribute("citations", 2)
        span.set_attribute("failure_mode", "")
        span.set_attribute("status", "ok")

    spans = _read_spans(tmp_path)
    assert spans
    s = spans[-1]

    # IDs are hex strings
    assert isinstance(s["trace_id"], str), "trace_id must be str"
    assert isinstance(s["span_id"], str), "span_id must be str"
    # parent_span_id is None or str
    assert s["parent_span_id"] is None or isinstance(
        s["parent_span_id"], str
    ), "parent_span_id must be str or None"

    # Numeric fields
    assert isinstance(s["duration_ms"], (int, float)), "duration_ms must be numeric"
    assert isinstance(s["input_tokens"], int), "input_tokens must be int"
    assert isinstance(s["output_tokens"], int), "output_tokens must be int"
    assert isinstance(s["cached_tokens"], int), "cached_tokens must be int"
    assert isinstance(s["cost_usd"], float), "cost_usd must be float"
    assert isinstance(s["citations"], int), "citations must be int"

    # String fields
    assert isinstance(s["agent"], str), "agent must be str"
    assert isinstance(s["operation"], str), "operation must be str"
    assert isinstance(s["decision_id"], str), "decision_id must be str"
    assert isinstance(s["model"], str), "model must be str"
    assert isinstance(s["status"], str), "status must be str"


def test_parent_span_id_chains_correctly(tmp_path: Path) -> None:
    """A child span must record its parent's span_id in parent_span_id."""
    use_sync_exporter(traces_root=tmp_path, run_id="01TEST000000000000000000003")
    tracer = get_tracer("risk")

    with tracer.start_as_current_span("parent.op") as parent:
        parent.set_attribute("agent", "risk")
        parent.set_attribute("operation", "parent.op")
        parent.set_attribute("decision_id", "dec-004")
        parent.set_attribute("model", "")
        parent.set_attribute("input_tokens", 0)
        parent.set_attribute("output_tokens", 0)
        parent.set_attribute("cached_tokens", 0)
        parent.set_attribute("cost_usd", 0.0)
        parent.set_attribute("citations", 0)
        parent.set_attribute("failure_mode", "")
        parent.set_attribute("status", "ok")

        ctx = parent.get_span_context()
        parent_span_id_hex = format(ctx.span_id, "016x")

        with tracer.start_as_current_span("child.op") as child:
            child.set_attribute("agent", "risk")
            child.set_attribute("operation", "child.op")
            child.set_attribute("decision_id", "dec-004")
            child.set_attribute("model", "claude-haiku-4-5")
            child.set_attribute("input_tokens", 50)
            child.set_attribute("output_tokens", 30)
            child.set_attribute("cached_tokens", 0)
            child.set_attribute("cost_usd", 0.0005)
            child.set_attribute("citations", 1)
            child.set_attribute("failure_mode", "")
            child.set_attribute("status", "ok")

    spans = _read_spans(tmp_path)
    # Find the child span by operation name
    child_spans = [s for s in spans if s.get("operation") == "child.op"]
    assert child_spans, "Child span not found in output"
    child_span = child_spans[-1]

    assert child_span["parent_span_id"] == parent_span_id_hex, (
        f"Child's parent_span_id {child_span['parent_span_id']!r} "
        f"does not match parent's span_id {parent_span_id_hex!r}"
    )


def test_firm_otel_sync_env_activates_sync_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting ``FIRM_OTEL_SYNC=1`` causes ``init_tracer`` to use ``SimpleSpanProcessor``.

    We assert the sync-mode behaviour observationally: a span ended inside the
    ``with`` block must be on disk by the time we read it back, with no
    explicit flush.  ``BatchSpanProcessor`` would buffer it.
    """
    monkeypatch.setenv("FIRM_OTEL_SYNC", "1")

    # No importlib.reload: the env var is consulted lazily inside init_tracer
    # / get_tracer, not at import time.  use_sync_exporter rebinds the
    # exporter in place so the existing provider routes to tmp_path.
    init_tracer(traces_root=tmp_path, run_id="01TEST000000000000000000004")
    use_sync_exporter(traces_root=tmp_path, run_id="01TEST000000000000000000004")

    tracer = get_tracer("exec")

    with tracer.start_as_current_span("exec.order") as span:
        span.set_attribute("agent", "exec")
        span.set_attribute("operation", "exec.order")
        span.set_attribute("decision_id", "dec-005")
        span.set_attribute("model", "")
        span.set_attribute("input_tokens", 0)
        span.set_attribute("output_tokens", 0)
        span.set_attribute("cached_tokens", 0)
        span.set_attribute("cost_usd", 0.0)
        span.set_attribute("citations", 0)
        span.set_attribute("failure_mode", "")
        span.set_attribute("status", "ok")

    spans = _read_spans(tmp_path)
    assert any(
        s.get("operation") == "exec.order" for s in spans
    ), "Sync-mode span not written"


def test_otlp_exporter_raises_when_dependency_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``OTEL_EXPORTER=otlp`` raises ``ImportError`` when the optional dep is absent.

    ``opentelemetry-exporter-otlp-proto-grpc`` is not in our core deps; the
    OTLP branch must fail loud with a helpful message rather than silently
    falling back to the file exporter.
    """
    monkeypatch.setenv("OTEL_EXPORTER", "otlp")
    with pytest.raises(ImportError, match="opentelemetry-exporter-otlp-proto-grpc"):
        init_tracer(traces_root=tmp_path, run_id="01TEST000000000000000000005")


def test_failure_mode_string_round_trips(tmp_path: Path) -> None:
    """A non-empty ``failure_mode`` attribute round-trips through JSON exactly."""
    use_sync_exporter(traces_root=tmp_path, run_id="01TEST000000000000000000006")
    tracer = get_tracer("research")

    with tracer.start_as_current_span("llm.call.failed") as span:
        span.set_attribute("agent", "research")
        span.set_attribute("operation", "llm.call")
        span.set_attribute("decision_id", "dec-006")
        span.set_attribute("model", "claude-haiku-4-5")
        span.set_attribute("input_tokens", 100)
        span.set_attribute("output_tokens", 0)
        span.set_attribute("cached_tokens", 0)
        span.set_attribute("cost_usd", 0.0)
        span.set_attribute("citations", 0)
        span.set_attribute("failure_mode", "LLM_UNAVAILABLE")
        span.set_attribute("status", "error")

    spans = _read_spans(tmp_path)
    failed = [s for s in spans if s.get("operation") == "llm.call"]
    assert failed, "Failed span not found in output"
    assert failed[-1]["failure_mode"] == "LLM_UNAVAILABLE"
    assert failed[-1]["status"] == "error"
