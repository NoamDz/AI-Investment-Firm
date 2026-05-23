"""Plan 4 T22 — UNGROUNDED_CLAIM end-to-end fixture.

Wires the grounded research factory with stubs that fabricate a
``source_chunk_id`` not present in the retrieved chunks. The grounding
validator inserted between extract and sufficiency in
:mod:`firm.agents.research` catches the fabrication and forces a REFUSE
Decision stamped with ``failure_mode=UNGROUNDED_CLAIM``.

The judge stub asserts it is never called — the validator must short-circuit
before Step 5, so any future regression that lets a fabricated chunk_id
reach the sufficiency judge will fail loudly.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.agents.research import make_research
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.config import load_universe
from firm.core.models import ActionEnum, Claim, FailureMode
from firm.grounding.schema import SufficiencyResult
from firm.rag.chunk import Chunk
from firm.rag.retrieve import RetrievedChunk


_REAL_CHUNK_ID = "real_chunk_1"
_GHOST_CHUNK_ID = "GHOST_CHUNK_DOES_NOT_EXIST"


def _make_real_chunk() -> Chunk:
    return Chunk(
        id=_REAL_CHUNK_ID,
        doc_id="doc-aapl-001",
        ticker="AAPL",
        section="body",
        published_at=datetime(2024, 2, 15, tzinfo=timezone.utc),
        text="AAPL reported strong revenue growth in the most recent quarter.",
        char_span=(0, 80),
        token_count=12,
        source="test",
    )


class _OneChunkRetriever:
    """Retriever stub: returns exactly one valid RetrievedChunk."""

    def __init__(self) -> None:
        self._chunk = _make_real_chunk()

    def retrieve(
        self, query: str, *, as_of: datetime
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk=self._chunk,
                score=1.0,
                rank_dense=0,
                rank_sparse=0,
                rerank_score=0.9,
            )
        ]


class _FabricatedChunkIdExtractor:
    """Extractor stub: returns one claim citing a chunk_id not in retrieval."""

    last_tool_call_ids: list[str] = []

    def extract(
        self, *, query: str, chunks: list[Chunk], as_of: datetime
    ) -> list[Claim]:
        return [
            Claim(
                text="aapl q3 revenue rose",
                value=None,
                unit=None,
                source_chunk_id=_GHOST_CHUNK_ID,
                source_span=(0, 10),
                tool_call_id=None,
            )
        ]


class _ForbiddenJudge:
    """Sufficiency judge stub that must never be invoked on the ungrounded branch."""

    def assess(
        self, *, question: str, claims: list[Claim]
    ) -> SufficiencyResult:
        raise AssertionError(
            "judge must not be called when grounding validator catches a "
            "fabricated chunk_id"
        )


def test_heartbeat_emits_refuse_with_ungrounded_claim_on_fabricated_chunk_id() -> None:
    """Fabricated chunk_id => REFUSE / UNGROUNDED_CLAIM, no citations, judge skipped."""
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    universe = load_universe(Path("config/universe.yaml"))

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=_OneChunkRetriever(),  # type: ignore[arg-type]  # structurally compatible
        extractor=_FabricatedChunkIdExtractor(),  # type: ignore[arg-type]  # structurally compatible
        judge=_ForbiddenJudge(),  # type: ignore[arg-type]  # structurally compatible
        nonce_secret=b"x" * 32,
    )
    out = research({"heartbeat_at": clock.now().isoformat()})

    decision = out["research_decision"]
    assert decision.action == ActionEnum.REFUSE
    assert decision.failure_mode == FailureMode.UNGROUNDED_CLAIM
    assert decision.citations == []
    assert len(out["claims"]) == 1
    assert len(out["retrieved_chunks"]) == 1
    assert _GHOST_CHUNK_ID in decision.rationale
