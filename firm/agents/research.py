"""Research agent — grounded LLM-backed implementation (Plan 2 §T19).

The factory function ``make_research`` returns a node callable for the
LangGraph workflow.  Two paths are supported for backwards compatibility:

* **Grounded path** (Plan 2): when both ``retriever`` and ``extractor`` are
  provided, the heartbeat issues a deterministic research question, runs
  ``retriever.retrieve(...)`` with ``as_of=clock.now()``, extracts cited
  Claims, and emits a Decision whose ``citations`` are mapped per-claim
  exactly as the spec specifies.  Empty retrieval → REFUSE / INSUFFICIENT_EVIDENCE.

* **Legacy stub path** (Plan 1): when either collaborator is absent, fall
  back to the deterministic stub that picks the cheapest ticker.  This keeps
  ``test_research.py`` and the unmigrated CLI working until T29 wires the
  real stack.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Callable

from firm.broker.protocol import Broker
from firm.core.clock import Clock
from firm.core.config import UniverseConfig
from firm.core.ids import sign_nonce, ulid_new
from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Citation,
    Claim,
    Decision,
    FailureMode,
    HoldPayload,
    RefusePayload,
)
from firm.llm.citations import CitedClaimExtractor
from firm.rag.chunk import Chunk
from firm.rag.retrieve import GroundedRetriever
from firm.orchestrator.state import WorkingState


# Dev/test default; T29 wires the real secret from config / environment.
_DEFAULT_NONCE_SECRET: bytes = b"\x00" * 32


def _doc_id_of(chunk_id: str, chunks: list[Chunk]) -> str:
    """Look up the ``doc_id`` of the chunk with ``chunk_id`` in ``chunks``.

    Raises ``ValueError`` if no chunk in ``chunks`` has that id — the spec
    explicitly forbids fabricating a doc_id for an unknown chunk.
    """
    lookup: dict[str, str] = {c.id: c.doc_id for c in chunks}
    if chunk_id not in lookup:
        raise ValueError(
            f"chunk_id {chunk_id!r} not found in retrieved chunks; "
            "refusing to fabricate source_id for Citation"
        )
    return lookup[chunk_id]


def _format_question(ticker: str) -> str:
    return (
        f"Summarize {ticker}'s latest reported financial trajectory and any "
        f"near-term catalysts."
    )


def _build_citations(claims: list[Claim], chunks: list[Chunk]) -> list[Citation]:
    """Map each grounded ``Claim`` to a ``Citation`` per the spec.

    A Claim with ``source_chunk_id is None`` (e.g. tool-derived) produces no
    Citation.  The (chunk_id, span, source_id, cited_text) mapping is
    load-bearing — implemented exactly per spec §T19 step 3 #4.
    """
    citations: list[Citation] = []
    for claim in claims:
        if claim.source_chunk_id is None:
            continue
        citations.append(
            Citation(
                source_id=_doc_id_of(claim.source_chunk_id, chunks),
                chunk_id=claim.source_chunk_id,
                span=claim.source_span if claim.source_span is not None else (0, 0),
                cited_text=claim.text,
                document_index=None,
                document_title=None,
            )
        )
    return citations


def _compute_oldest_filing_age_days(
    chunks: list[Chunk], *, now: datetime
) -> int | None:
    """Return age (in days) of the oldest chunk by ``published_at``.

    Returns ``None`` if ``chunks`` is empty.  Clamped to ``>= 0`` so a chunk
    published "in the future" (e.g. eval-corpus dating quirk) cannot produce
    a negative staleness signal that would confuse downstream Risk checks.
    """
    if not chunks:
        return None
    oldest_published = min(c.published_at for c in chunks)
    return max(0, (now.date() - oldest_published.date()).days)


def _make_grounded_research(
    *,
    clock: Clock,
    broker: Broker,  # noqa: ARG001 -- reserved for T26+ ticker rotation by price
    universe: UniverseConfig,
    retriever: GroundedRetriever,
    extractor: CitedClaimExtractor,
    nonce_secret: bytes,
) -> Callable[[WorkingState], dict[str, Any]]:
    """Build the grounded heartbeat node."""

    def research(state: WorkingState) -> dict[str, Any]:  # noqa: ARG001 -- reads clock, not heartbeat
        # Step 1: deterministic ticker selection. Simplest stable rule.
        ticker = universe.tickers[0]
        question = _format_question(ticker)
        now = clock.now()

        # Step 2: retrieve. Empty → REFUSE / INSUFFICIENT_EVIDENCE.
        retrieved = retriever.retrieve(question, as_of=now)
        chunks: list[Chunk] = [rc.chunk for rc in retrieved]

        decision_id = ulid_new()
        nonce = sign_nonce(
            nonce_secret, decision_id=decision_id, timestamp=int(now.timestamp())
        )

        if not chunks:
            refuse_decision = Decision(
                id=decision_id,
                decision_id_chain=[],
                action=ActionEnum.REFUSE,
                payload=RefusePayload(
                    reason=f"no retrieval hits for {ticker} at {now.isoformat()}"
                ),
                rationale="retriever returned no chunks; cannot ground any claim",
                confidence=0.0,
                citations=[],
                falsification_condition=(
                    f"{ticker} retrieval returns chunks at a later heartbeat"
                ),
                escalation_reason=None,
                failure_mode=FailureMode.INSUFFICIENT_EVIDENCE,
                metadata={"agent": "research", "ticker": ticker},
                nonce=nonce,
            )
            return {
                "research_decision": refuse_decision,
                "retrieved_chunks": chunks,
                "claims": [],
            }

        # Step 3: extract cited claims.
        claims = extractor.extract(query=question, chunks=chunks, as_of=now)

        # Step 4: build Decision. Action follows simple claim-sentiment rule
        # (Plan 2: PM owns the real action; research emits an opinion).
        citations = _build_citations(claims, chunks)

        if claims:
            payload: BuyPayload | HoldPayload = BuyPayload(
                ticker=ticker, shares=Decimal("10")
            )
            action = ActionEnum.BUY
            rationale = " ".join(c.text for c in claims)
            falsification_condition = (
                f"{claims[0].text} is contradicted by later filings"
            )
        else:
            payload = HoldPayload(reason="no extractable claims")
            action = ActionEnum.HOLD
            rationale = "no claims extracted"
            falsification_condition = (
                f"{ticker} reports materially different fundamentals next quarter"
            )

        # Step 5: surface oldest_filing_age_days in metadata for Risk.
        metadata: dict[str, Any] = {"agent": "research", "ticker": ticker}
        oldest_age = _compute_oldest_filing_age_days(chunks, now=now)
        if oldest_age is not None:
            metadata["oldest_filing_age_days"] = oldest_age

        decision = Decision(
            id=decision_id,
            decision_id_chain=[],
            action=action,
            payload=payload,
            rationale=rationale,
            confidence=0.6 if claims else 0.3,
            citations=citations,
            falsification_condition=falsification_condition,
            escalation_reason=None,
            failure_mode=None,
            metadata=metadata,
            nonce=nonce,
        )

        # Step 6: write retrieved_chunks + claims to state.
        return {
            "research_decision": decision,
            "retrieved_chunks": chunks,
            "claims": claims,
        }

    return research


def _make_legacy_stub_research(
    *,
    clock: Clock,  # noqa: ARG001 -- accepted for signature parity with grounded path
    broker: Broker,
    universe: UniverseConfig,
) -> Callable[[WorkingState], dict[str, Any]]:
    """Plan 1 deterministic stub. Preserved for backwards compatibility."""

    def research(state: WorkingState) -> dict[str, Any]:
        prices = {t: broker.get_quote(t).price for t in universe.tickers}
        chosen = min(prices, key=lambda t: prices[t])
        decision = Decision(
            id=ulid_new(),
            decision_id_chain=[],
            action=ActionEnum.BUY,
            payload=BuyPayload(ticker=chosen, shares=Decimal("10")),
            rationale=(
                "deterministic stub: cheapest of universe at heartbeat "
                f"{state.get('heartbeat_at')}"
            ),
            confidence=0.5,
            citations=[],
            falsification_condition=f"if {chosen} drops more than 5% by EOD",
            escalation_reason=None,
            failure_mode=None,
            metadata={"agent": "research", "stub": True},
            nonce="research-stub",
        )
        return {"research_decision": decision}

    return research


def make_research(
    *,
    clock: Clock,
    broker: Broker,
    universe: UniverseConfig,
    retriever: GroundedRetriever | None = None,
    extractor: CitedClaimExtractor | None = None,
    nonce_secret: bytes = _DEFAULT_NONCE_SECRET,
) -> Callable[[WorkingState], dict[str, Any]]:
    """Build a research node callable.

    When both ``retriever`` and ``extractor`` are provided, returns the
    grounded heartbeat (Plan 2 §T19).  Otherwise returns the Plan 1
    deterministic stub.  This dual signature lets T29 swap in the real RAG
    stack while keeping existing Plan 1 tests + CLI working today.
    """
    if retriever is not None and extractor is not None:
        return _make_grounded_research(
            clock=clock,
            broker=broker,
            universe=universe,
            retriever=retriever,
            extractor=extractor,
            nonce_secret=nonce_secret,
        )
    return _make_legacy_stub_research(clock=clock, broker=broker, universe=universe)
