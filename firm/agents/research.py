"""Research agent — grounded LLM-backed implementation (Plan 2 §T19, §T21).

The factory function ``make_research`` returns a node callable for the
LangGraph workflow.  Two paths are supported for backwards compatibility:

* **Grounded path** (Plan 2): when ``retriever``, ``extractor``, and
  ``judge`` are all provided, the heartbeat issues a deterministic
  research question, runs ``retriever.retrieve(...)`` with
  ``as_of=clock.now()``, extracts cited Claims, then runs the
  :class:`SufficiencyJudge` to label each claim. Branching on the
  aggregate sufficiency status:

  * ``ok``         → BUY (or HOLD when no claims survived extraction).
  * ``partial``    → ESCALATE with ``escalation_reason='sufficiency:partial'``.
  * ``insufficient`` → REFUSE with ``failure_mode=INSUFFICIENT_EVIDENCE``.

  If the judge raises :class:`JudgeResponseError`, the agent emits a
  REFUSE Decision with ``failure_mode=LLM_UNAVAILABLE``. Every branch
  surfaces a serialized ``sufficiency_result`` on the returned state so
  downstream agents (PM, Risk, HITL reviewer) can introspect it.

* **Legacy stub path** (Plan 1): when any of retriever / extractor /
  judge is absent, fall back to the deterministic stub that picks the
  cheapest ticker.  This keeps ``test_research.py`` and the unmigrated
  CLI working until T29 wires the real stack.
"""
from __future__ import annotations

import copy
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
    EscalatePayload,
    FailureMode,
    HoldPayload,
    RefusePayload,
)
from firm.grounding.judge import JudgeResponseError, JudgeSchemaError, SufficiencyJudge
from firm.grounding.schema import SufficiencyResult
from firm.llm.citations import CitedClaimExtractor
from firm.obs import agent_span, llm_span, retrieval_span, stamp_decision
from firm.rag.chunk import Chunk
from firm.rag.retrieve import GroundedRetriever
from firm.orchestrator.state import WorkingState


# Provider literal used on every llm_span emitted from this module — lifted
# to a constant so a future provider rename only changes one site.
_PROVIDER_ANTHROPIC = "anthropic"


# Default placeholder used when the judge errored before producing a result
# but we still owe downstream a uniform sufficiency_result shape.
_LLM_UNAVAILABLE_SUFFICIENCY: dict[str, Any] = {
    "claim_assessments": [],
    "overall_reasoning": "sufficiency judge unavailable (LLM error)",
}

# Placeholder used when the judge returned a response that failed schema
# validation — distinct from LLM_UNAVAILABLE (transport/JSON parse errors).
_SCHEMA_VALIDATION_FAILED_SUFFICIENCY: dict[str, Any] = {
    "claim_assessments": [],
    "overall_reasoning": "sufficiency judge response failed schema validation",
}

# Placeholder used when retrieval returned no chunks — the judge is skipped
# (no claims to assess) but downstream still expects the key to be present.
_EMPTY_RETRIEVAL_SUFFICIENCY: dict[str, Any] = {
    "claim_assessments": [],
    "overall_reasoning": "retrieval empty; judge skipped",
}


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

    ``cited_text`` prefers the verbatim ``source_quote`` (the literal source
    span returned by Anthropic's Citations API) and falls back to the Claim's
    block-level ``text`` only when the extractor could not capture a quote
    (older records or malformed API entries).
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
                cited_text=claim.source_quote if claim.source_quote is not None else claim.text,
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
    judge: SufficiencyJudge,
    nonce_secret: bytes | None,
) -> Callable[[WorkingState], dict[str, Any]]:
    """Build the grounded heartbeat node."""
    if nonce_secret is None:
        # Fail fast at factory time: a missing secret in the grounded path would
        # otherwise either propagate as a ValueError from sign_nonce at the
        # first heartbeat or — if defaulted — silently ship every Decision
        # with an HMAC over a known key.  Force callers to inject explicitly.
        raise ValueError(
            "nonce_secret is required for the grounded research path"
        )
    nonce_key: bytes = nonce_secret  # narrow Optional for mypy --strict inside closure.

    # Pull each LLM call's model id directly off the collaborators so the
    # ``llm_span`` attribute is the actual model that handled the request.
    # ``_model`` is a leading-underscore implementation attribute; treating it
    # as read-only from the agent layer keeps the LLM client interfaces
    # unchanged (T03 must not restructure them).
    extractor_model: str = getattr(extractor, "_model", "unknown")
    judge_model: str = getattr(judge, "_model", "unknown")

    def research(state: WorkingState) -> dict[str, Any]:  # noqa: ARG001 -- reads clock, not heartbeat
        # T03: CM form (not decorator) so REFUSE branches can stamp
        # ``failure_mode`` and ``decision_id`` onto the agent span before
        # returning.  Mirrors the pattern used by ``firm/cli.py`` risk_node.
        with agent_span("research") as span:
            # Step 1: deterministic ticker selection. Simplest stable rule.
            ticker = universe.tickers[0]
            question = _format_question(ticker)
            now = clock.now()

            # Step 2: retrieve. Empty → REFUSE / INSUFFICIENT_EVIDENCE.
            # The retriever is a ``GroundedRetriever`` (hybrid + rerank); the
            # ``retrieval.hybrid`` operation name is the rollup the spec asks for
            # ("1 span per retrieval stage").  Per-sub-stage spans (BM25, dense,
            # rerank) belong inside the retriever implementation, not here.
            with retrieval_span("hybrid"):
                retrieved = retriever.retrieve(question, as_of=now)
            chunks: list[Chunk] = [rc.chunk for rc in retrieved]
            chunks_dump: list[dict[str, Any]] = [c.model_dump() for c in chunks]

            decision_id = ulid_new()
            nonce = sign_nonce(
                nonce_key, decision_id=decision_id, timestamp=int(now.timestamp())
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
                stamp_decision(
                    span, refuse_decision.id, refuse_decision.failure_mode
                )
                return {
                    "research_decision": refuse_decision,
                    "retrieved_chunks": chunks_dump,
                    "claims": [],
                    "sufficiency_result": copy.deepcopy(_EMPTY_RETRIEVAL_SUFFICIENCY),
                    "tool_call_ids": [],
                }

            # Step 3: extract cited claims.
            with llm_span(_PROVIDER_ANTHROPIC, extractor_model):
                claims = extractor.extract(query=question, chunks=chunks, as_of=now)
            claims_dump: list[dict[str, Any]] = [c.model_dump() for c in claims]
            # Surface tool_call_ids from the extractor (T24). The Protocol
            # guarantees the attribute exists; copy defensively so downstream
            # mutation cannot leak back into the extractor's state.
            tool_call_ids: list[str] = list(extractor.last_tool_call_ids)

            # Step 4: oldest-filing-age metadata (shared across branches).
            metadata: dict[str, Any] = {"agent": "research", "ticker": ticker}
            oldest_age = _compute_oldest_filing_age_days(chunks, now=now)
            if oldest_age is not None:
                metadata["oldest_filing_age_days"] = oldest_age

            # Step 5: sufficiency gate.
            # JudgeSchemaError (subclass) → REFUSE SCHEMA_VALIDATION_FAILED.
            # JudgeResponseError           → REFUSE LLM_UNAVAILABLE.
            # Catch ONLY these two; other exceptions propagate so a real
            # bug is not silently masked. JudgeSchemaError must come first
            # because it is a subclass of JudgeResponseError.
            try:
                with llm_span(_PROVIDER_ANTHROPIC, judge_model):
                    sufficiency: SufficiencyResult = judge.assess(
                        question=question, claims=claims
                    )
            except JudgeSchemaError as exc:
                schema_validation_failed_decision = Decision(
                    id=decision_id,
                    decision_id_chain=[],
                    action=ActionEnum.REFUSE,
                    payload=RefusePayload(reason="sufficiency:schema_validation_failed"),
                    rationale=f"sufficiency judge response failed schema validation: {exc!s}",
                    confidence=0.0,
                    citations=_build_citations(claims, chunks),
                    falsification_condition=(
                        f"sufficiency judge returns a conforming response for {ticker} at a later heartbeat"
                    ),
                    escalation_reason=None,
                    failure_mode=FailureMode.SCHEMA_VALIDATION_FAILED,
                    metadata=metadata,
                    nonce=nonce,
                )
                stamp_decision(
                    span,
                    schema_validation_failed_decision.id,
                    schema_validation_failed_decision.failure_mode,
                )
                return {
                    "research_decision": schema_validation_failed_decision,
                    "retrieved_chunks": chunks_dump,
                    "claims": claims_dump,
                    "sufficiency_result": copy.deepcopy(_SCHEMA_VALIDATION_FAILED_SUFFICIENCY),
                    "tool_call_ids": tool_call_ids,
                }
            except JudgeResponseError as exc:
                llm_unavailable_decision = Decision(
                    id=decision_id,
                    decision_id_chain=[],
                    action=ActionEnum.REFUSE,
                    payload=RefusePayload(reason="sufficiency:llm_unavailable"),
                    rationale=f"sufficiency judge unavailable: {exc!s}",
                    confidence=0.0,
                    citations=_build_citations(claims, chunks),
                    falsification_condition=(
                        f"sufficiency judge succeeds for {ticker} at a later heartbeat"
                    ),
                    escalation_reason=None,
                    failure_mode=FailureMode.LLM_UNAVAILABLE,
                    metadata=metadata,
                    nonce=nonce,
                )
                stamp_decision(
                    span,
                    llm_unavailable_decision.id,
                    llm_unavailable_decision.failure_mode,
                )
                return {
                    "research_decision": llm_unavailable_decision,
                    "retrieved_chunks": chunks_dump,
                    "claims": claims_dump,
                    "sufficiency_result": copy.deepcopy(_LLM_UNAVAILABLE_SUFFICIENCY),
                    "tool_call_ids": tool_call_ids,
                }

            sufficiency_dump: dict[str, Any] = sufficiency.model_dump(mode="json")
            status = sufficiency.aggregate_status()
            citations = _build_citations(claims, chunks)

            # Step 6: branch on aggregate sufficiency status.
            if status == "insufficient":
                insufficient_decision = Decision(
                    id=decision_id,
                    decision_id_chain=[],
                    action=ActionEnum.REFUSE,
                    payload=RefusePayload(reason="sufficiency:insufficient"),
                    rationale=(
                        "sufficiency judge marked at least one claim UNSUPPORTED"
                    ),
                    confidence=0.0,
                    citations=citations,
                    falsification_condition=(
                        f"{ticker} produces fully-supported claims at a later heartbeat"
                    ),
                    escalation_reason=None,
                    failure_mode=FailureMode.INSUFFICIENT_EVIDENCE,
                    metadata=metadata,
                    nonce=nonce,
                )
                stamp_decision(
                    span,
                    insufficient_decision.id,
                    insufficient_decision.failure_mode,
                )
                return {
                    "research_decision": insufficient_decision,
                    "retrieved_chunks": chunks_dump,
                    "claims": claims_dump,
                    "sufficiency_result": sufficiency_dump,
                    "tool_call_ids": tool_call_ids,
                }

            if status == "partial":
                # ESCALATE requires a proposed Buy/Sell payload (the HITL reviewer
                # needs the action they would be approving).  Use the same default
                # BUY(10 shares) shape as the happy path so the proposed action is
                # consistent across both branches.
                # TODO(T27): partial-evidence ESCALATE currently proposes the same default
                # BUY shape as the happy path; when PM voters wire real sizing, this branch
                # should either skip the proposed payload or compute a reduced size.
                escalate_decision = Decision(
                    id=decision_id,
                    decision_id_chain=[],
                    action=ActionEnum.ESCALATE,
                    payload=EscalatePayload(
                        proposed=BuyPayload(ticker=ticker, shares=Decimal("10")),
                        reason="sufficiency:partial",
                    ),
                    rationale=(
                        "sufficiency judge marked at least one claim PARTIAL; "
                        "escalating to HITL review"
                    ),
                    confidence=0.4,
                    citations=citations,
                    falsification_condition=(
                        f"{ticker} produces fully-supported claims at a later heartbeat"
                    ),
                    escalation_reason="sufficiency:partial",
                    failure_mode=None,
                    metadata=metadata,
                    nonce=nonce,
                )
                stamp_decision(
                    span, escalate_decision.id, escalate_decision.failure_mode
                )
                return {
                    "research_decision": escalate_decision,
                    "retrieved_chunks": chunks_dump,
                    "claims": claims_dump,
                    "sufficiency_result": sufficiency_dump,
                    "tool_call_ids": tool_call_ids,
                }

            # status == "ok" → proceed with the original BUY / HOLD decision.
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

            stamp_decision(span, decision.id, decision.failure_mode)
            return {
                "research_decision": decision,
                "retrieved_chunks": chunks_dump,
                "claims": claims_dump,
                "sufficiency_result": sufficiency_dump,
                "tool_call_ids": tool_call_ids,
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
        # T03: CM form so failure_mode/decision_id can be set on the span
        # (legacy stub never produces a failure_mode, but decision_id is set
        # for parity with the grounded path).
        with agent_span("research") as span:
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
            stamp_decision(span, decision.id, decision.failure_mode)
            return {"research_decision": decision}

    return research


def make_research(
    *,
    clock: Clock,
    broker: Broker,
    universe: UniverseConfig,
    retriever: GroundedRetriever | None = None,
    extractor: CitedClaimExtractor | None = None,
    judge: SufficiencyJudge | None = None,
    nonce_secret: bytes | None = None,
) -> Callable[[WorkingState], dict[str, Any]]:
    """Build a research node callable.

    When ``retriever``, ``extractor``, AND ``judge`` are all provided,
    returns the grounded heartbeat (Plan 2 §T19 + §T21) and
    ``nonce_secret`` is REQUIRED — leaving it ``None`` raises rather
    than letting the agent ship Decisions signed with a zero key.
    Otherwise returns the Plan 1 deterministic stub (which uses a
    literal nonce and ignores ``nonce_secret`` entirely).  This dual
    signature lets T29 swap in the real RAG stack while keeping
    existing Plan 1 tests + CLI working today.
    """
    if retriever is not None and extractor is not None and judge is not None:
        return _make_grounded_research(
            clock=clock,
            broker=broker,
            universe=universe,
            retriever=retriever,
            extractor=extractor,
            judge=judge,
            nonce_secret=nonce_secret,
        )
    return _make_legacy_stub_research(clock=clock, broker=broker, universe=universe)
