"""Plan 4 T18 — INSUFFICIENT_EVIDENCE end-to-end fixture.

Wires the grounded research factory with a retriever stub that returns zero
chunks for the heartbeat's ticker. Asserts the agent emits a REFUSE Decision
stamped with ``failure_mode=INSUFFICIENT_EVIDENCE`` and that the empty
retrieval is faithfully surfaced on the returned state (no citations, no
claims, empty ``retrieved_chunks`` dump).

Complements (does not replace) the agent-level unit coverage at
``tests/unit/test_research_agent.py::test_research_refuses_when_retriever_returns_empty``
by exercising the same branch through the public ``make_research`` factory
at the integration layer — the extractor and judge stubs assert they are
never called, so any future regression that re-routes empty retrieval into
the LLM path will fail loudly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import NoReturn

from firm.agents.research import make_research
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.config import load_universe
from firm.core.models import ActionEnum, Claim, FailureMode
from firm.grounding.schema import SufficiencyResult
from firm.rag.chunk import Chunk
from firm.rag.retrieve import RetrievedChunk


class _EmptyRetriever:
    """Retriever stub: always returns zero chunks, regardless of query/as_of."""

    def retrieve(
        self, query: str, *, as_of: datetime
    ) -> list[RetrievedChunk]:
        return []


class _ForbiddenExtractor:
    """Extractor stub that must never be invoked on the empty-retrieval branch."""

    last_tool_call_ids: list[str] = []

    def extract(
        self, *, query: str, chunks: list[Chunk], as_of: datetime
    ) -> NoReturn:
        raise AssertionError(
            "extractor must not be called when retriever returns no chunks"
        )


class _ForbiddenJudge:
    """Sufficiency judge stub that must never be invoked on the empty branch."""

    def assess(
        self, *, question: str, claims: list[Claim]
    ) -> SufficiencyResult:
        raise AssertionError(
            "judge must not be called when retriever returns no chunks"
        )


def test_heartbeat_emits_refuse_with_insufficient_evidence_on_empty_retrieval() -> None:
    """Empty retrieval → REFUSE / INSUFFICIENT_EVIDENCE, no citations, no claims."""
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    universe = load_universe(Path("config/universe.yaml"))

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=_EmptyRetriever(),  # type: ignore[arg-type]  # structurally compatible
        extractor=_ForbiddenExtractor(),
        judge=_ForbiddenJudge(),  # type: ignore[arg-type]  # structurally compatible
        nonce_secret=b"x" * 32,
    )
    out = research({"heartbeat_at": clock.now().isoformat()})

    decision = out["research_decision"]
    assert decision.action == ActionEnum.REFUSE
    assert decision.failure_mode == FailureMode.INSUFFICIENT_EVIDENCE
    assert decision.citations == []
    assert out["retrieved_chunks"] == []
    assert out["claims"] == []
