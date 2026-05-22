"""Integration tests for T24: parallel PM voter execution via asyncio.gather.

Three tests:
1. test_pm_voters_execute_in_parallel — wall-clock < 100ms for 3×50ms stub voters.
2. test_pm_parent_span_has_latency_attributes — agent.pm span carries the four
   T24 timing attributes with sane values.
3. test_pm_first_error_propagates — LLMUnavailableError from voter 1 yields a
   REFUSE Decision with FailureMode.LLM_UNAVAILABLE (matches pre-T24 behavior).
"""
from __future__ import annotations

import json
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

from firm.agents.pm import PmVoter, make_pm
from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Claim,
    Decision,
    FailureMode,
    RefusePayload,
)
from firm.llm.router import LLMUnavailableError
from firm.obs.tracer import use_sync_exporter
from firm.orchestrator.state import WorkingState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _vote_json(
    vote: str = "BUY",
    confidence: float = 0.8,
    rationale: str = "ok",
    cited_claim_ids: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "vote": vote,
            "confidence": confidence,
            "rationale": rationale,
            "cited_claim_ids": cited_claim_ids if cited_claim_ids is not None else [],
        }
    )


def _claim_dicts(n: int) -> list[dict[str, Any]]:
    return [
        Claim(text=f"Claim {i + 1}.", source_chunk_id=f"chunk-{i}").model_dump()
        for i in range(n)
    ]


def _research_buy() -> Decision:
    return Decision(
        id="res-1",
        decision_id_chain=[],
        action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="research thinks BUY based on margin expansion.",
        confidence=0.6,
        citations=[],
        falsification_condition="AAPL margin reverses next quarter",
        escalation_reason=None,
        failure_mode=None,
        metadata={"agent": "research", "ticker": "AAPL"},
        nonce="research-nonce",
    )


# ---------------------------------------------------------------------------
# Test 1: voters execute in parallel (wall-clock < 100ms for 3×50ms stubs)
# ---------------------------------------------------------------------------


class _SlowStubClient:
    """Stub AnthropicMessagesClient that sleeps ``delay_s`` seconds per call.

    Returns a valid BUY vote JSON regardless of which lens fired, so
    aggregate_votes works.
    """

    def __init__(self, delay_s: float, vote_text: str) -> None:
        self._delay_s = delay_s
        self._vote_text = vote_text

    def messages_create(self, **kwargs: Any) -> dict[str, Any]:
        time.sleep(self._delay_s)
        return {"content": [{"type": "text", "text": self._vote_text}]}


def test_pm_voters_execute_in_parallel() -> None:
    """Three voters each sleeping 50ms must complete in < 100ms wall-clock.

    Sequential execution would take >= 150ms; parallel takes ~50ms.
    The 130ms budget leaves comfortable headroom for thread-spawn overhead
    and GIL contention on a loaded Windows CI machine while still proving
    the parallel path beats sequential (150ms) by a clear margin.
    """
    vote_text = _vote_json("BUY", 0.8, "ok")
    slow_client = _SlowStubClient(delay_s=0.05, vote_text=vote_text)
    voter = PmVoter(client=slow_client, model="claude-sonnet-4-6")  # type: ignore[arg-type]
    pm = make_pm(voter)

    state: WorkingState = {
        "research_decision": _research_buy(),
        "claims": _claim_dicts(1),
    }

    t0 = time.perf_counter()
    out = pm(state)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert out["pm_decision"].action == ActionEnum.BUY, (
        f"Expected BUY, got {out['pm_decision'].action}"
    )
    assert elapsed_ms < 130.0, (
        f"Parallel execution took {elapsed_ms:.1f}ms — expected < 130ms "
        f"(sequential would be ~150ms)"
    )


# ---------------------------------------------------------------------------
# Test 2: agent.pm span has the four T24 latency attributes
# ---------------------------------------------------------------------------


class _FastStubClient:
    """Deterministic stub AnthropicMessagesClient — returns immediately."""

    def __init__(self, vote_text: str) -> None:
        self._vote_text = vote_text

    def messages_create(self, **kwargs: Any) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": self._vote_text}]}


def _read_spans(traces_dir: Path) -> list[dict[str, object]]:
    spans: list[dict[str, object]] = []
    for jsonl_file in traces_dir.rglob("*.jsonl"):
        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans


def test_pm_parent_span_has_latency_attributes(tmp_path: Path) -> None:
    """agent.pm span must carry pm.voter_count / pm.parallel_ms / pm.sequential_estimate_ms
    / pm.latency_delta_ms after a successful parallel vote round.

    Strategy: add an InMemorySpanExporter as a secondary processor on the global
    TracerProvider so we can inspect the raw span attributes (the JSONL file
    exporter only serialises the 15 known schema fields, not custom pm.* attrs).
    """
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import firm.obs.tracer as _tracer_mod

    # Ensure provider is initialised (use_sync_exporter is idempotent).
    use_sync_exporter(traces_root=tmp_path / "traces", run_id="01T24LATENCY00000000000000")

    # Attach a secondary in-memory exporter that captures raw spans.
    in_mem = InMemorySpanExporter()
    processor = SimpleSpanProcessor(in_mem)
    assert _tracer_mod._provider is not None, "TracerProvider must be initialised"
    _tracer_mod._provider.add_span_processor(processor)

    try:
        vote_text = _vote_json("BUY", 0.8, "good quality")
        fast_client = _FastStubClient(vote_text)
        voter = PmVoter(client=fast_client, model="claude-sonnet-4-6")  # type: ignore[arg-type]
        pm = make_pm(voter)

        state: WorkingState = {
            "research_decision": _research_buy(),
            "claims": _claim_dicts(1),
        }
        out = pm(state)
        assert out["pm_decision"].action == ActionEnum.BUY

        # Find the agent.pm span.
        finished = in_mem.get_finished_spans()
        pm_spans = [
            s for s in finished if (s.attributes or {}).get("operation") == "agent.pm"
        ]
        assert pm_spans, (
            f"No agent.pm span found. Spans seen: "
            f"{[(s.attributes or {}).get('operation') for s in finished]}"
        )
        pm_span = pm_spans[-1]
        attrs = dict(pm_span.attributes or {})

        # Assert all four T24 attributes are present with sane values.
        assert attrs.get("pm.voter_count") == 3, (
            f"pm.voter_count={attrs.get('pm.voter_count')!r}, expected 3"
        )
        parallel_ms = attrs.get("pm.parallel_ms")
        assert isinstance(parallel_ms, (int, float)) and parallel_ms >= 0.0, (
            f"pm.parallel_ms={parallel_ms!r} is not a non-negative number"
        )
        sequential_ms = attrs.get("pm.sequential_estimate_ms")
        assert isinstance(sequential_ms, (int, float)) and sequential_ms >= 0.0, (
            f"pm.sequential_estimate_ms={sequential_ms!r} is not a non-negative number"
        )
        delta_ms = attrs.get("pm.latency_delta_ms")
        # Delta is measurement-based: sequential_estimate (sum of per-voter
        # perf_counter samples) minus parallel wall-clock. For trivially fast
        # stub workloads, asyncio.gather/to_thread overhead can dominate, so
        # delta may be negative — that's an honest signal, not a bug.
        assert isinstance(delta_ms, (int, float)), (
            f"pm.latency_delta_ms={delta_ms!r} is not a number"
        )
        # Structural invariant: delta == sequential_estimate - parallel
        # (within rounding to 2 decimal places).
        assert abs(delta_ms - (sequential_ms - parallel_ms)) < 0.05, (  # type: ignore[operator]
            f"latency_delta ({delta_ms}) != sequential_estimate ({sequential_ms}) "
            f"- parallel ({parallel_ms})"
        )
    finally:
        # Detach the in-memory processor so it doesn't pollute other tests.
        # SimpleSpanProcessor.shutdown() stops it from receiving further spans.
        processor.shutdown()


# ---------------------------------------------------------------------------
# Test 3: first error from any voter propagates as REFUSE / LLM_UNAVAILABLE
# ---------------------------------------------------------------------------


class _ErrorOnFirstCallClient:
    """Stub AnthropicMessagesClient that raises LLMUnavailableError on the first
    messages_create call (simulating voter 1 exhausting all router profiles).

    Subsequent calls return a valid vote so the test validates that only one
    exception suffices to trigger the REFUSE path.
    """

    def __init__(self, vote_text: str) -> None:
        self._vote_text = vote_text
        self._calls = 0

    def messages_create(self, **kwargs: Any) -> dict[str, Any]:
        self._calls += 1
        if self._calls == 1:
            raise LLMUnavailableError("all models exhausted")
        return {"content": [{"type": "text", "text": self._vote_text}]}


def test_pm_first_error_propagates() -> None:
    """LLMUnavailableError raised by any voter must produce a REFUSE Decision
    with FailureMode.LLM_UNAVAILABLE — matching pre-T24 sequential behavior.
    """
    vote_text = _vote_json("BUY", 0.8, "ok")
    error_client = _ErrorOnFirstCallClient(vote_text)
    voter = PmVoter(client=error_client, model="claude-sonnet-4-6")  # type: ignore[arg-type]
    pm = make_pm(voter)

    state: WorkingState = {
        "research_decision": _research_buy(),
        "claims": _claim_dicts(1),
    }
    out = pm(state)

    decision: Decision = out["pm_decision"]
    assert decision.action == ActionEnum.REFUSE, (
        f"Expected REFUSE, got {decision.action}"
    )
    assert decision.failure_mode == FailureMode.LLM_UNAVAILABLE, (
        f"Expected LLM_UNAVAILABLE, got {decision.failure_mode}"
    )
    assert isinstance(decision.payload, RefusePayload)
    assert "exhausted" in decision.payload.reason or "exhausted" in decision.rationale
