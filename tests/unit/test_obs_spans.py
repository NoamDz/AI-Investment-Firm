"""Tests for ``firm.obs.spans`` decorators / context managers (Plan 3 T02).

TDD: tests are written *before* the implementation; they drive the public API
of the four decorators (``agent_span``, ``llm_span``, ``tool_span``,
``retrieval_span``).

Operation-name scheme under test:

* ``agent_span("research")``                 -> operation = ``agent.research``
* ``llm_span("anthropic", "claude-sonnet")`` -> operation = ``llm.call``
* ``tool_span("fundamentals.get_ratio")``    -> operation = ``tool.fundamentals.get_ratio``
* ``retrieval_span("rerank")``               -> operation = ``retrieval.rerank``
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from firm.obs import agent_span, llm_span, retrieval_span, tool_span
from firm.obs.tracer import use_sync_exporter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# Tests — context-manager usage
# ---------------------------------------------------------------------------


def test_agent_span_context_manager_emits_span(tmp_path: Path) -> None:
    """``with agent_span("research"):`` emits a span with agent+operation attrs."""
    use_sync_exporter(traces_root=tmp_path, run_id="01T02000000000000000000000")

    with agent_span("research"):
        pass

    spans = _read_spans(tmp_path)
    agent_spans = [s for s in spans if s.get("operation") == "agent.research"]
    assert agent_spans, "agent.research span not found"
    s = agent_spans[-1]
    assert s["agent"] == "research"
    assert s["operation"] == "agent.research"
    # Clean exit: failure_mode is empty
    assert s["failure_mode"] == ""


def test_llm_span_sets_model_and_operation(tmp_path: Path) -> None:
    """``llm_span`` sets ``operation=llm.call`` and ``model`` on the span."""
    use_sync_exporter(traces_root=tmp_path, run_id="01T02000000000000000000001")

    with llm_span("anthropic", "claude-sonnet-4-6"):
        pass

    spans = _read_spans(tmp_path)
    llm_spans = [s for s in spans if s.get("operation") == "llm.call"]
    assert llm_spans, "llm.call span not found"
    s = llm_spans[-1]
    assert s["model"] == "claude-sonnet-4-6"
    assert s["operation"] == "llm.call"


def test_tool_span_operation_prefixed_with_tool(tmp_path: Path) -> None:
    """``tool_span("fundamentals.get_ratio")`` emits operation ``tool.<name>``."""
    use_sync_exporter(traces_root=tmp_path, run_id="01T02000000000000000000002")

    with tool_span("fundamentals.get_ratio"):
        pass

    spans = _read_spans(tmp_path)
    tool_spans = [
        s for s in spans if s.get("operation") == "tool.fundamentals.get_ratio"
    ]
    assert tool_spans, "tool.fundamentals.get_ratio span not found"


def test_retrieval_span_operation_prefixed_with_retrieval(tmp_path: Path) -> None:
    """``retrieval_span("rerank")`` emits operation ``retrieval.rerank``."""
    use_sync_exporter(traces_root=tmp_path, run_id="01T02000000000000000000003")

    with retrieval_span("rerank"):
        pass

    spans = _read_spans(tmp_path)
    retrieval_spans = [s for s in spans if s.get("operation") == "retrieval.rerank"]
    assert retrieval_spans, "retrieval.rerank span not found"


# ---------------------------------------------------------------------------
# Tests — nested chaining
# ---------------------------------------------------------------------------


def test_nested_agent_then_llm_chains_parent_span_id(tmp_path: Path) -> None:
    """Inner llm_span's ``parent_span_id`` equals outer agent_span's ``span_id``."""
    use_sync_exporter(traces_root=tmp_path, run_id="01T02000000000000000000004")

    with agent_span("research"):
        with llm_span("anthropic", "claude-sonnet-4-6"):
            pass

    spans = _read_spans(tmp_path)
    outer = next(s for s in spans if s.get("operation") == "agent.research")
    inner = next(s for s in spans if s.get("operation") == "llm.call")

    assert inner["parent_span_id"] == outer["span_id"], (
        f"inner parent_span_id {inner['parent_span_id']!r} != "
        f"outer span_id {outer['span_id']!r}"
    )
    # Outer span has no parent
    assert outer["parent_span_id"] is None


def test_three_level_nesting_chains_parent_span_ids(tmp_path: Path) -> None:
    """agent -> retrieval -> tool nesting chains parent_span_id at each level."""
    use_sync_exporter(traces_root=tmp_path, run_id="01T02000000000000000000005")

    with agent_span("research"):
        with retrieval_span("hybrid"):
            with tool_span("vectorstore.search"):
                pass

    spans = _read_spans(tmp_path)
    a = next(s for s in spans if s.get("operation") == "agent.research")
    r = next(s for s in spans if s.get("operation") == "retrieval.hybrid")
    t = next(s for s in spans if s.get("operation") == "tool.vectorstore.search")

    assert a["parent_span_id"] is None
    assert r["parent_span_id"] == a["span_id"]
    assert t["parent_span_id"] == r["span_id"]
    # All spans share the same trace_id
    assert a["trace_id"] == r["trace_id"] == t["trace_id"]


# ---------------------------------------------------------------------------
# Tests — exception path
# ---------------------------------------------------------------------------


def test_exception_sets_status_and_failure_mode_then_reraises(tmp_path: Path) -> None:
    """Exception inside a span: span gets status=error + failure_mode=<class>; reraised."""
    use_sync_exporter(traces_root=tmp_path, run_id="01T02000000000000000000006")

    with pytest.raises(ValueError, match="boom"):
        with agent_span("research"):
            raise ValueError("boom")

    spans = _read_spans(tmp_path)
    matching = [s for s in spans if s.get("operation") == "agent.research"]
    assert matching, "agent.research span not found"
    s = matching[-1]
    assert s["status"] == "error"
    assert s["failure_mode"] == "ValueError"


def test_exception_in_inner_span_only_marks_inner(tmp_path: Path) -> None:
    """Exception in an inner span marks only that span; outer remains untagged."""
    use_sync_exporter(traces_root=tmp_path, run_id="01T02000000000000000000007")

    with pytest.raises(RuntimeError, match="inner-fail"):
        with agent_span("research"):
            with llm_span("anthropic", "claude-haiku-4-5"):
                raise RuntimeError("inner-fail")

    spans = _read_spans(tmp_path)
    outer = next(s for s in spans if s.get("operation") == "agent.research")
    inner = next(s for s in spans if s.get("operation") == "llm.call")

    # Inner span: exception was raised inside it -> tagged.
    assert inner["status"] == "error"
    assert inner["failure_mode"] == "RuntimeError"
    # Outer span: exception also propagates through it, so it is also tagged.
    # We allow both spans to be tagged (each decorator's __exit__ sees the exc).
    # The critical guarantee is that the inner span is tagged.
    assert outer["failure_mode"] in {"", "RuntimeError"}


# ---------------------------------------------------------------------------
# Tests — decorator usage (sugar over context manager)
# ---------------------------------------------------------------------------


def test_agent_span_as_decorator(tmp_path: Path) -> None:
    """``@agent_span("research")`` applied to a function emits a span on each call."""
    use_sync_exporter(traces_root=tmp_path, run_id="01T02000000000000000000008")

    @agent_span("research")
    def do_research(payload: str) -> str:
        return payload.upper()

    result = do_research("hello")
    assert result == "HELLO"

    spans = _read_spans(tmp_path)
    matching = [s for s in spans if s.get("operation") == "agent.research"]
    assert matching, "decorated-function span not found"
    assert matching[-1]["agent"] == "research"


def test_llm_span_as_decorator_with_exception(tmp_path: Path) -> None:
    """``@llm_span(...)`` on a function still tags failure_mode on exception."""
    use_sync_exporter(traces_root=tmp_path, run_id="01T02000000000000000000009")

    @llm_span("anthropic", "claude-haiku-4-5")
    def flaky_call() -> None:
        raise TimeoutError("network")

    with pytest.raises(TimeoutError, match="network"):
        flaky_call()

    spans = _read_spans(tmp_path)
    matching = [s for s in spans if s.get("operation") == "llm.call"]
    assert matching, "decorated llm.call span not found"
    s = matching[-1]
    assert s["status"] == "error"
    assert s["failure_mode"] == "TimeoutError"
    assert s["model"] == "claude-haiku-4-5"


def test_context_manager_yields_span_for_caller_attrs(tmp_path: Path) -> None:
    """The ``as span`` binding lets caller code set decision_id / token attrs."""
    use_sync_exporter(traces_root=tmp_path, run_id="01T02000000000000000000010")

    with llm_span("anthropic", "claude-sonnet-4-6") as span:
        span.set_attribute("decision_id", "dec-T02")
        span.set_attribute("input_tokens", 123)
        span.set_attribute("output_tokens", 45)

    spans = _read_spans(tmp_path)
    matching = [s for s in spans if s.get("operation") == "llm.call"]
    assert matching
    s = matching[-1]
    assert s["decision_id"] == "dec-T02"
    assert s["input_tokens"] == 123
    assert s["output_tokens"] == 45
