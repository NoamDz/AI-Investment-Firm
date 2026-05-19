"""Unit tests for the grounded research agent (T19).

These tests exercise the grounded path of :func:`firm.agents.research.make_research`
through stub ``GroundedRetriever`` and ``CitedClaimExtractor`` collaborators. The
legacy stub path (no retriever/extractor) is covered by ``test_research.py``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from firm.agents.research import make_research
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.config import UniverseConfig, load_universe
from firm.core.ids import verify_nonce
from firm.core.models import ActionEnum, Claim, FailureMode
from firm.rag.chunk import Chunk
from firm.rag.retrieve import RetrievedChunk


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubRetriever:
    """Returns a fixed list of RetrievedChunks; records the last as_of seen."""

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results
        self.last_as_of: datetime | None = None
        self.last_query: str | None = None

    def retrieve(self, query: str, *, as_of: datetime) -> list[RetrievedChunk]:
        self.last_as_of = as_of
        self.last_query = query
        return self._results


class _StubExtractor:
    """Returns a fixed list of Claims; records the last (query, chunks, as_of)."""

    def __init__(self, claims: list[Claim]) -> None:
        self._claims = claims
        self.last_query: str | None = None
        self.last_chunks: list[Chunk] | None = None
        self.last_as_of: datetime | None = None

    def extract(
        self,
        *,
        query: str,
        chunks: list[Chunk],
        as_of: datetime,
    ) -> list[Claim]:
        self.last_query = query
        self.last_chunks = chunks
        self.last_as_of = as_of
        return list(self._claims)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    *,
    doc_id: str | None = None,
    published_at: datetime | None = None,
    text: str = "body text",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=doc_id if doc_id is not None else chunk_id.split("::")[0],
        ticker="AAPL",
        section="body",
        published_at=published_at
        if published_at is not None
        else datetime(2024, 6, 1, tzinfo=timezone.utc),
        text=text,
        char_span=(0, len(text)),
        token_count=max(1, len(text.split())),
    )


def _wrap(chunk: Chunk, *, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=chunk,
        score=score,
        rank_dense=0,
        rank_sparse=0,
        rerank_score=score,
    )


@pytest.fixture
def universe() -> UniverseConfig:
    return load_universe(Path("config/universe.yaml"))


@pytest.fixture
def broker() -> FakeBroker:
    return FakeBroker(initial_cash=Decimal("100000"))


@pytest.fixture
def clock() -> ReplayClock:
    return ReplayClock(datetime(2024, 9, 15, 14, 30, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Tests (all 7 from the spec)
# ---------------------------------------------------------------------------


def test_research_emits_decision_with_citations_and_claims(
    universe: UniverseConfig, broker: FakeBroker, clock: ReplayClock
) -> None:
    chunk = _make_chunk("doc-a::0001", text="Apple revenue grew 8% YoY.")
    retriever = _StubRetriever([_wrap(chunk)])
    claim = Claim(
        text="Apple revenue grew 8% YoY.",
        source_chunk_id="doc-a::0001",
        source_span=(0, 26),
    )
    extractor = _StubExtractor([claim])

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=retriever,  # type: ignore[arg-type]  # stub is structurally compatible
        extractor=extractor,
        nonce_secret=b"x" * 32,
    )
    out = research({"heartbeat_at": clock.now().isoformat()})

    decision = out["research_decision"]
    assert len(decision.citations) == 1
    assert out["retrieved_chunks"] == [chunk]
    assert out["claims"] == [claim]


def test_research_refuses_when_retriever_returns_empty(
    universe: UniverseConfig, broker: FakeBroker, clock: ReplayClock
) -> None:
    retriever = _StubRetriever([])
    extractor = _StubExtractor([])

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=retriever,  # type: ignore[arg-type]  # stub is structurally compatible
        extractor=extractor,
        nonce_secret=b"x" * 32,
    )
    out = research({"heartbeat_at": clock.now().isoformat()})

    decision = out["research_decision"]
    assert decision.action == ActionEnum.REFUSE
    assert decision.failure_mode == FailureMode.INSUFFICIENT_EVIDENCE


def test_research_uses_pit_filter_with_replay_clock(
    universe: UniverseConfig, broker: FakeBroker, clock: ReplayClock
) -> None:
    chunk = _make_chunk("doc-a::0001")
    retriever = _StubRetriever([_wrap(chunk)])
    extractor = _StubExtractor(
        [
            Claim(
                text="A claim.",
                source_chunk_id=chunk.id,
                source_span=(0, 8),
            )
        ]
    )

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=retriever,  # type: ignore[arg-type]  # stub is structurally compatible
        extractor=extractor,
        nonce_secret=b"x" * 32,
    )
    research({"heartbeat_at": clock.now().isoformat()})

    assert retriever.last_as_of == clock.now()
    assert extractor.last_as_of == clock.now()


def test_research_falsification_condition_non_empty(
    universe: UniverseConfig, broker: FakeBroker, clock: ReplayClock
) -> None:
    chunk = _make_chunk("doc-a::0001", text="A factual statement.")
    retriever = _StubRetriever([_wrap(chunk)])
    extractor = _StubExtractor(
        [
            Claim(
                text="A factual statement.",
                source_chunk_id=chunk.id,
                source_span=(0, 20),
            )
        ]
    )

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=retriever,  # type: ignore[arg-type]  # stub is structurally compatible
        extractor=extractor,
        nonce_secret=b"x" * 32,
    )
    out = research({"heartbeat_at": clock.now().isoformat()})
    decision = out["research_decision"]
    assert decision.falsification_condition
    assert len(decision.falsification_condition) >= 1


def test_research_citation_fields_map_from_claim(
    universe: UniverseConfig, broker: FakeBroker, clock: ReplayClock
) -> None:
    chunk = _make_chunk(
        "doc-citation::0007",
        doc_id="doc-citation",
        text="Revenue rose to $90B in Q3.",
    )
    retriever = _StubRetriever([_wrap(chunk)])
    claim = Claim(
        text="Revenue rose to $90B in Q3.",
        source_chunk_id="doc-citation::0007",
        source_span=(13, 17),
    )
    extractor = _StubExtractor([claim])

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=retriever,  # type: ignore[arg-type]  # stub is structurally compatible
        extractor=extractor,
        nonce_secret=b"x" * 32,
    )
    out = research({"heartbeat_at": clock.now().isoformat()})
    decision = out["research_decision"]

    assert len(decision.citations) == 1
    citation = decision.citations[0]
    assert citation.chunk_id == claim.source_chunk_id
    assert citation.span == claim.source_span
    assert citation.source_id == chunk.doc_id
    assert citation.cited_text == claim.text
    assert citation.document_index is None
    assert citation.document_title is None


def test_research_tool_only_claims_do_not_produce_citation(
    universe: UniverseConfig, broker: FakeBroker, clock: ReplayClock
) -> None:
    chunk = _make_chunk("doc-a::0001", text="Some grounded text.")
    retriever = _StubRetriever([_wrap(chunk)])
    grounded_claim = Claim(
        text="Grounded claim.",
        source_chunk_id=chunk.id,
        source_span=(0, 15),
    )
    tool_claim = Claim(
        text="Tool-derived claim.",
        value=Decimal("1.5"),
        unit="ratio",
        tool_call_id="tc-abc",
    )
    extractor = _StubExtractor([grounded_claim, tool_claim])

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=retriever,  # type: ignore[arg-type]  # stub is structurally compatible
        extractor=extractor,
        nonce_secret=b"x" * 32,
    )
    out = research({"heartbeat_at": clock.now().isoformat()})
    decision = out["research_decision"]

    assert len(decision.citations) == 1
    assert decision.citations[0].chunk_id == chunk.id
    # Both claims remain in state.claims; only the cited one produced a Citation.
    assert out["claims"] == [grounded_claim, tool_claim]


def test_research_surfaces_oldest_filing_age_days(
    universe: UniverseConfig, broker: FakeBroker, clock: ReplayClock
) -> None:
    now = clock.now()
    recent = _make_chunk(
        "doc-recent::0001",
        published_at=now - timedelta(days=30),
        text="Recent text.",
    )
    older = _make_chunk(
        "doc-older::0001",
        published_at=now - timedelta(days=200),
        text="Older text.",
    )
    retriever = _StubRetriever([_wrap(recent), _wrap(older)])
    extractor = _StubExtractor(
        [
            Claim(
                text="Recent text.",
                source_chunk_id=recent.id,
                source_span=(0, 12),
            )
        ]
    )

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=retriever,  # type: ignore[arg-type]  # stub is structurally compatible
        extractor=extractor,
        nonce_secret=b"x" * 32,
    )
    out = research({"heartbeat_at": clock.now().isoformat()})
    decision = out["research_decision"]

    assert decision.metadata["oldest_filing_age_days"] == 200


# ---------------------------------------------------------------------------
# Additional load-bearing assertions (not in the seven-test list but spec-required)
# ---------------------------------------------------------------------------


def test_research_grounded_uses_first_universe_ticker_in_question(
    universe: UniverseConfig, broker: FakeBroker, clock: ReplayClock
) -> None:
    chunk = _make_chunk("doc-a::0001")
    retriever = _StubRetriever([_wrap(chunk)])
    extractor = _StubExtractor(
        [Claim(text="Foo.", source_chunk_id=chunk.id, source_span=(0, 4))]
    )

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=retriever,  # type: ignore[arg-type]  # stub is structurally compatible
        extractor=extractor,
        nonce_secret=b"x" * 32,
    )
    research({"heartbeat_at": clock.now().isoformat()})

    expected_ticker = universe.tickers[0]
    assert retriever.last_query is not None
    assert expected_ticker in retriever.last_query
    assert "financial trajectory" in retriever.last_query


def test_research_nonce_is_hmac_signed(
    universe: UniverseConfig, broker: FakeBroker, clock: ReplayClock
) -> None:
    chunk = _make_chunk("doc-a::0001")
    retriever = _StubRetriever([_wrap(chunk)])
    extractor = _StubExtractor(
        [Claim(text="Foo.", source_chunk_id=chunk.id, source_span=(0, 4))]
    )

    secret = b"x" * 32
    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=retriever,  # type: ignore[arg-type]  # stub is structurally compatible
        extractor=extractor,
        nonce_secret=secret,
    )
    out = research({"heartbeat_at": clock.now().isoformat()})
    decision = out["research_decision"]

    assert verify_nonce(
        secret,
        decision_id=decision.id,
        timestamp=int(clock.now().timestamp()),
        nonce=decision.nonce,
    )


def test_research_grounded_requires_nonce_secret(
    universe: UniverseConfig, broker: FakeBroker, clock: ReplayClock
) -> None:
    """The grounded path must refuse to build when nonce_secret is absent.

    Guards against a T29 wiring slip that would otherwise let every Decision
    ship with an HMAC over a known zero key.
    """
    retriever = _StubRetriever([])
    extractor = _StubExtractor([])

    with pytest.raises(ValueError, match="nonce_secret"):
        make_research(
            clock=clock,
            broker=broker,
            universe=universe,
            retriever=retriever,  # type: ignore[arg-type]  # stub is structurally compatible
            extractor=extractor,
            # nonce_secret intentionally omitted → factory must raise.
        )
