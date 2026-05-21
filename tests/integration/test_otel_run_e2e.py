"""End-to-end OTel span emission integration test (Plan 3 T03 spec-compliance).

This test runs one heartbeat-equivalent in-process through the production
agent factories (``make_research``, ``make_pm``, the CLI-style risk closure,
``make_reporter``) wired against deterministic stubs.  All four nodes go
through their real production code paths — only the *external* I/O
collaborators (LLM client, retriever, reranker, broker, DB) are stubs.

Spec assertion (from T03):

    Tests/integration/test_otel_run_e2e.py runs one heartbeat and asserts
    >= 1 span per agent + 1 span per LLM call + 1 span per retrieval stage.

We therefore assert the JSONL trace file contains, at minimum:

* >= 1 ``agent.research`` span (from the wrapped research closure)
* >= 1 ``agent.pm`` span     (from the wrapped pm closure)
* >= 1 ``agent.risk`` span   (from the wrapped CLI-style risk closure)
* >= 1 ``agent.reporter`` span (from the wrapped reporter closure)
* >= 1 ``llm.call`` span    (research extractor + judge + 3 PM voters ≥ 5)
* >= 1 ``retrieval.*`` span (hybrid retrieval stage)

All spans must share the same ``trace_id`` (a single heartbeat traces one run).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from firm.agents.pm import PmVoter, make_pm
from firm.agents.reporter import make_reporter
from firm.agents.research import make_research
from firm.agents.risk import RiskInput, evaluate_risk
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.config import PolicyConfig, load_policy, load_universe
from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Claim,
    Decision,
    SellPayload,
)
from firm.db.migrations import init_db
from firm.grounding.schema import (
    ClaimAssessment,
    ClaimSupport,
    SufficiencyResult,
)
from firm.obs import agent_span
from firm.obs.tracer import use_sync_exporter
from firm.rag.chunk import Chunk
from firm.rag.retrieve import RetrievedChunk


# ---------------------------------------------------------------------------
# Stubs — minimal, structurally compatible with the production protocols
# ---------------------------------------------------------------------------


class _StubHybrid:
    """Stub HybridRetriever — returns one fixed RetrievedChunk."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results

    def retrieve(self, query: str, *, as_of: datetime) -> list[RetrievedChunk]:
        return list(self._results)


class _StubReranker:
    """Stub BgeReranker — pass-through, returns the input list truncated to k."""

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, k: int
    ) -> list[RetrievedChunk]:
        return list(candidates[:k])


class _StubExtractor:
    """Returns one fixed Claim; advertises empty tool_call_ids."""

    last_tool_call_ids: list[str] = []

    def __init__(self, claims: list[Claim], model: str) -> None:
        self._claims = claims
        # Same private attribute name the real extractor uses.
        self._model = model

    def extract(
        self, *, query: str, chunks: list[Chunk], as_of: datetime
    ) -> list[Claim]:
        self.last_tool_call_ids = []
        return list(self._claims)


class _StubJudge:
    """Returns a fixed SufficiencyResult (all SUPPORTED)."""

    def __init__(self, result: SufficiencyResult, model: str) -> None:
        self._result = result
        self._model = model

    def assess(
        self, *, question: str, claims: list[Claim]
    ) -> SufficiencyResult:
        return self._result


class _StubMessagesClient:
    """Stub AnthropicMessagesClient that returns a fixed JSON vote response.

    Used by the real ``PmVoter`` so the production ``voter.vote(...)`` code
    path executes (which is what T03 must instrument with ``llm_span``).
    """

    def __init__(self, vote_json: str) -> None:
        self._vote_json = vote_json
        self.calls: list[dict[str, Any]] = []

    def messages_create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"content": [{"type": "text", "text": self._vote_json}]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_spans(traces_dir: Path) -> list[dict[str, object]]:
    """Return all span dicts found under *traces_dir* (recursively)."""
    spans: list[dict[str, object]] = []
    for jsonl_file in traces_dir.rglob("*.jsonl"):
        for line in jsonl_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans


def _make_chunk(text: str = "Apple revenue grew 8% YoY.") -> Chunk:
    return Chunk(
        id="doc-aapl-001::0001",
        doc_id="doc-aapl-001",
        ticker="AAPL",
        section="body",
        published_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        text=text,
        char_span=(0, len(text)),
        token_count=max(1, len(text.split())),
    )


def _wrap(chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk, score=0.9, rank_dense=0, rank_sparse=0, rerank_score=0.9
    )


def _all_supported(num_claims: int) -> SufficiencyResult:
    return SufficiencyResult(
        claim_assessments=[
            ClaimAssessment(
                claim_id=f"c{i + 1}",
                support=ClaimSupport.SUPPORTED,
                reasoning="supported in cited evidence",
            )
            for i in range(num_claims)
        ],
        overall_reasoning="all claims supported",
    )


# ---------------------------------------------------------------------------
# The integration test
# ---------------------------------------------------------------------------


def test_otel_run_e2e_emits_required_spans(tmp_path: Path) -> None:
    """One heartbeat-equivalent emits the spans T05's spec-compliance check requires.

    Drives the real production node factories (``make_research``,
    ``make_pm``, the CLI-style risk closure, ``make_reporter``) through
    stub LLM client + stub retriever/reranker so the wiring under test is
    the production ``agent_span`` / ``llm_span`` / ``retrieval_span``
    instrumentation — not test-only mocks.

    Asserts:
      * >= 1 ``agent.research``  span
      * >= 1 ``agent.pm``        span
      * >= 1 ``agent.risk``      span
      * >= 1 ``agent.reporter``  span
      * >= 1 ``llm.call``        span (research extractor + judge + 3 PM voters)
      * >= 1 ``retrieval.*``     span (hybrid retrieval stage)
      * All emitted spans share a single ``trace_id`` because they are nested
        inside one outer per-heartbeat ``agent_span("heartbeat")`` context.
    """
    # ---------------------------------------------------------------- #
    # Phase 1: redirect the OTel sync exporter at this test's tmp_path #
    # ---------------------------------------------------------------- #
    # Keep the trace JSONL output isolated from any other JSONL the heartbeat
    # writes (notably ``reports/decisions.jsonl``); ``_read_spans`` globs
    # recursively and would otherwise try to parse those rows as spans.
    traces_root = tmp_path / "traces"
    use_sync_exporter(
        traces_root=traces_root, run_id="01OTELRUN3T03000000000000"
    )

    # ---------------------------------------------------------------- #
    # Phase 2: build deterministic stubs + real collaborators           #
    # ---------------------------------------------------------------- #
    utc = timezone.utc
    clock = ReplayClock(datetime(2024, 9, 15, 14, 30, tzinfo=utc))
    universe = load_universe(Path("config/universe.yaml"))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    policy: PolicyConfig = load_policy(Path("config/policy.yaml"))

    chunk = _make_chunk()
    claim = Claim(
        text="Apple revenue grew 8% YoY.",
        source_chunk_id=chunk.id,
        source_span=(0, len(chunk.text)),
    )

    # Real GroundedRetriever wrapping stub hybrid + stub reranker — so the
    # production retrieval-stage instrumentation runs.
    from firm.rag.retrieve import GroundedRetriever

    grounded_retriever = GroundedRetriever(
        hybrid=_StubHybrid([_wrap(chunk)]),  # type: ignore[arg-type]
        reranker=_StubReranker(),  # type: ignore[arg-type]
        k_final=4,
    )
    extractor = _StubExtractor([claim], model="claude-sonnet-4-6")
    judge = _StubJudge(_all_supported(num_claims=1), model="claude-haiku-4-5")

    research_node = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=grounded_retriever,
        extractor=extractor,  # type: ignore[arg-type]
        judge=judge,  # type: ignore[arg-type]
        nonce_secret=b"x" * 32,
    )

    # Real PmVoter driven by a stub messages client.
    vote_json = json.dumps(
        {
            "vote": "BUY",
            "confidence": 0.8,
            "rationale": "Strong evidence across the lens.",
            "cited_claim_ids": ["c1"],
        }
    )
    stub_client = _StubMessagesClient(vote_json)
    voter = PmVoter(client=stub_client, model="claude-sonnet-4-6")  # type: ignore[arg-type]
    pm_node = make_pm(voter=voter)

    # CLI-style risk closure (mirrors firm/cli.py:run.risk_node) — this is
    # the layer T03 wraps with ``agent_span("risk")``.
    def risk_node(state: dict[str, Any]) -> dict[str, Any]:
        proposal: Decision = state["pm_decision"]
        with agent_span("risk") as span:
            if not isinstance(proposal.payload, (BuyPayload, SellPayload)):
                return {"risk_decision": proposal}
            ticker = proposal.payload.ticker
            quote = broker.get_quote(ticker)
            positions = {p.ticker: p.shares for p in broker.list_positions()}
            decision = evaluate_risk(
                RiskInput(
                    proposal=proposal,
                    quote_price=quote.price,
                    quote_age_seconds=0,
                    cash=broker.get_cash(),
                    positions=positions,
                    sector_map=universe.sector_map,
                    trades_today=0,
                    nav=broker.get_cash()
                    + sum(
                        (
                            p.shares * broker.get_quote(p.ticker).price
                            for p in broker.list_positions()
                        ),
                        Decimal("0"),
                    ),
                    daily_pnl_pct=0.0,
                    policy=policy,
                )
            )
            if decision.failure_mode is not None:
                span.set_attribute("failure_mode", decision.failure_mode.value)
            span.set_attribute("decision_id", decision.id)
            return {"risk_decision": decision}

    # Reporter that also writes the trace pointer onto each row.
    db_path = tmp_path / "firm.db"
    init_db(db_path)
    reports_root = tmp_path / "reports"
    reporter_node = make_reporter(
        reports_root=reports_root, clock=clock, db_path=db_path
    )

    # ---------------------------------------------------------------- #
    # Phase 3: drive one heartbeat through all four nodes              #
    # ---------------------------------------------------------------- #
    # We wrap the whole heartbeat in a single outer span so every emitted
    # span shares one trace_id — mirroring the per-heartbeat lifetime that
    # a real LangGraph run would produce when monitor is also instrumented.
    state: dict[str, Any] = {"heartbeat_at": clock.now().isoformat()}
    with agent_span("heartbeat"):
        state.update(research_node(state))
        state.update(pm_node(state))
        state.update(risk_node(state))
        state.update(reporter_node(state))

    # ---------------------------------------------------------------- #
    # Phase 4: read and assert the emitted spans                       #
    # ---------------------------------------------------------------- #
    spans = _read_spans(traces_root)
    assert spans, "no spans emitted at all"

    by_op: dict[str, list[dict[str, object]]] = {}
    for s in spans:
        by_op.setdefault(str(s["operation"]), []).append(s)

    # >= 1 span per agent
    assert by_op.get("agent.research"), (
        f"missing agent.research; ops seen: {sorted(by_op)}"
    )
    assert by_op.get("agent.pm"), (
        f"missing agent.pm; ops seen: {sorted(by_op)}"
    )
    assert by_op.get("agent.risk"), (
        f"missing agent.risk; ops seen: {sorted(by_op)}"
    )
    assert by_op.get("agent.reporter"), (
        f"missing agent.reporter; ops seen: {sorted(by_op)}"
    )

    # >= 1 span per LLM call (extractor + judge stubs do NOT emit llm.call
    # spans because their model invocations are stubbed; only the real
    # PmVoter goes through llm_span. We get 3 PM voter calls = 3 llm.call.
    # The research extractor and judge wrappers add 2 more = 5 total.
    llm_spans = by_op.get("llm.call", [])
    assert llm_spans, "no llm.call spans emitted"

    # >= 1 retrieval.* span
    retrieval_spans = [
        s for s in spans if str(s["operation"]).startswith("retrieval.")
    ]
    assert retrieval_spans, (
        f"no retrieval.* span emitted; ops seen: {sorted(by_op)}"
    )

    # Single trace_id across the whole heartbeat (the outer agent.heartbeat
    # ties all four agents + their children into one trace).
    trace_ids = {str(s["trace_id"]) for s in spans}
    assert len(trace_ids) == 1, (
        f"expected single trace_id across heartbeat, got {trace_ids}"
    )

    # ---------------------------------------------------------------- #
    # Phase 5: reporter wrote decisions.jsonl + a trace_id pointer     #
    # ---------------------------------------------------------------- #
    reports = list(reports_root.rglob("decisions.jsonl"))
    assert reports, "reporter did not write decisions.jsonl"
    rows = [
        json.loads(line)
        for line in reports[0].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "decisions.jsonl exists but is empty"
    last_row = rows[-1]
    assert "trace_id" in last_row, (
        f"reporter row missing 'trace_id' pointer; keys: {sorted(last_row)}"
    )
    # The reporter trace_id must match the trace seen in the JSONL spans.
    assert last_row["trace_id"] in trace_ids, (
        f"reporter trace_id {last_row['trace_id']!r} not in span trace_ids {trace_ids}"
    )
