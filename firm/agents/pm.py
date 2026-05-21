"""PM agent — vote-of-3 self-consistency over single-lens LLM rationales.

Plan 2 §T27: ``make_pm(voter)`` returns a node callable that runs three
single-lens PM voters (quality / valuation / catalyst) over the research
agent's extracted Claims, then aggregates the three votes via
:func:`aggregate_votes` into one PM Decision.

T25 contributed:  PmLens, PmVote, PmVoteSchemaError, PmVoter.
T26 contributed:  aggregate_votes (deterministic Python).
T27 rewires make_pm() to call them in sequence.

PM does NOT call retrieval or tools — Chinese-wall constraint from spec §3.2.
It reasons only over claims produced by Research.
"""
from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal
from enum import StrEnum
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError, model_validator

from firm.core.ids import ulid_new
from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Claim,
    Decision,
    EscalatePayload,
    FailureMode,
    HoldPayload,
    ProfileName,
    RefusePayload,
    SellPayload,
    TypedPayload,
)
from firm.llm.citations import AnthropicMessagesClient
from firm.llm.messages_client import RouterBackedMessagesClient
from firm.llm.prompts import pm_voter_system
from firm.llm.router import CostRouter, LLMUnavailableError
from firm.obs import agent_span, llm_span, stamp_decision
from firm.orchestrator.state import WorkingState


# Provider literal used on every llm_span emitted from this module — lifted
# to a constant so a future provider rename only changes one site.
_PROVIDER_ANTHROPIC = "anthropic"

# T08 escalation hook: PM default profile, and the profile used when the
# sufficiency judge returned PARTIAL but a human ack overrode (so PM is
# voting on a HITL-approved-but-marginal idea, where opus's deeper reasoning
# is worth the cost).
_PM_DEFAULT_PROFILE: ProfileName = "sonnet"
_PM_ESCALATED_PROFILE: ProfileName = "opus"

# Agent name stamped on router-managed cost-ledger rows (T09).
_PM_AGENT_NAME = "pm"


# ---------------------------------------------------------------------------
# T25 — PM voter (single-lens)
# ---------------------------------------------------------------------------


class PmLens(StrEnum):
    QUALITY = "quality"
    VALUATION = "valuation"
    CATALYST = "catalyst"


class PmVote(BaseModel):
    lens: PmLens
    vote: ActionEnum  # constrained to BUY|HOLD|SELL via validator below
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    cited_claim_ids: list[str]

    @model_validator(mode="after")
    def _vote_is_buy_hold_or_sell(self) -> "PmVote":
        if self.vote not in (ActionEnum.BUY, ActionEnum.HOLD, ActionEnum.SELL):
            raise ValueError(
                f"PmVote.vote must be BUY, HOLD, or SELL; got {self.vote!r}"
            )
        return self


class PmVoteSchemaError(Exception):
    """Raised when the Sonnet response cannot be parsed into a PmVote."""


def _strip_markdown_fences(text: str) -> str:
    """Strip a single leading ```...``` markdown fence if present.

    Conservative: only removes fences when the text starts with three
    backticks and ends with three backticks.  Leaves everything else
    untouched so JSON values that embed ``` are not corrupted.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    if not stripped.endswith("```"):
        return stripped
    body = stripped[3:]
    newline_at = body.find("\n")
    if newline_at != -1 and body[:newline_at].strip().isalpha():
        body = body[newline_at + 1:]
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


class PmVoter:
    """Single-lens PM voter backed by Sonnet.

    Each ``vote()`` call:
    1. Renders a lens-specific system prompt via ``pm_voter_system``.
    2. Builds a user message with the research question, rationale, and
       cited claims wrapped in ``<retrieved_content>`` tags.
    3. Calls the Anthropic ``messages_create`` API.
    4. Parses and validates the JSON response into a :class:`PmVote`.
    5. Filters ``cited_claim_ids`` to the subset of provided ids.
    """

    def __init__(
        self,
        *,
        client: AnthropicMessagesClient,
        model: str,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def vote(
        self,
        *,
        lens: PmLens,
        question: str,
        claims: list[Claim],
        research_rationale: str,
    ) -> PmVote:
        """Cast a single-lens vote on the trade idea.

        Parameters
        ----------
        lens:
            Which analytical lens to apply.
        question:
            The original research question / trade idea.
        claims:
            Cited claims produced by the Research Extractor.  Each claim is
            assigned a positional id ``c1``, ``c2``, ... (1-indexed).
        research_rationale:
            The research agent's summary rationale for the proposed action.

        Returns
        -------
        PmVote
            Validated vote with ``cited_claim_ids`` filtered to the subset
            of provided ids.

        Raises
        ------
        PmVoteSchemaError
            If the LLM response cannot be parsed into a valid :class:`PmVote`.
        """
        system = pm_voter_system(lens.value)

        # Build the positional id mapping: c1, c2, ...
        claim_ids = [f"c{i + 1}" for i in range(len(claims))]
        valid_id_set: set[str] = set(claim_ids)

        claim_lines = "\n".join(
            f"[{cid}] {claim.text}"
            for cid, claim in zip(claim_ids, claims)
        )
        user_text = (
            f"Research question: {question}\n\n"
            f"Research rationale: {research_rationale}\n\n"
            "<retrieved_content>\n"
            f"{claim_lines}\n"
            "</retrieved_content>"
        )

        messages: list[dict[str, object]] = [
            {"role": "user", "content": user_text},
        ]

        response = self._client.messages_create(
            model=self._model,
            system=system,
            messages=messages,
            tools=None,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

        # Concatenate all text-type content blocks.
        text_parts: list[str] = []
        content = response.get("content", [])
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "text":
                    continue
                text_val = block.get("text", "")
                if isinstance(text_val, str):
                    text_parts.append(text_val)
        raw_text = _strip_markdown_fences("".join(text_parts))

        # Parse JSON.
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise PmVoteSchemaError(
                f"PM voter returned non-JSON text: {raw_text[:120]!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise PmVoteSchemaError(
                f"PM voter JSON must be an object, got {type(payload).__name__}"
            )

        # Build and validate PmVote.
        try:
            vote_obj = PmVote(
                lens=lens,
                vote=ActionEnum(payload["vote"]),
                confidence=payload["confidence"],
                rationale=payload["rationale"],
                cited_claim_ids=payload["cited_claim_ids"],
            )
        except (ValidationError, KeyError, ValueError) as exc:
            raise PmVoteSchemaError(
                f"PM voter JSON failed schema validation: {exc!s}"
            ) from exc

        # Server-side filter: keep only ids that were actually provided.
        filtered_ids = [cid for cid in vote_obj.cited_claim_ids if cid in valid_id_set]
        # Re-instantiate via model_validate so all validators (e.g. the
        # BUY/HOLD/SELL invariant) re-run; model_copy(update=...) would
        # silently bypass them.
        try:
            vote_obj = PmVote.model_validate(
                {**vote_obj.model_dump(), "cited_claim_ids": filtered_ids}
            )
        except (ValidationError, KeyError, ValueError) as exc:
            raise PmVoteSchemaError(
                f"PM voter JSON failed schema validation: {exc!s}"
            ) from exc

        return vote_obj


# ---------------------------------------------------------------------------
# T26 — deterministic vote aggregation
# ---------------------------------------------------------------------------


# Discount applied to the majority confidence when one voter dissents.
# Hard-coded constant (rather than a parameter) so the aggregation function
# stays pure and deterministic; documented in aggregate_votes' docstring.
_MAJORITY_DISSENT_DISCOUNT = 0.8

# Canonical lens ordering for rationale concatenation. T27's caller will lift
# the per-lens rationales out of this combined string into Decision.metadata.
_LENS_ORDER: tuple[PmLens, ...] = (PmLens.QUALITY, PmLens.VALUATION, PmLens.CATALYST)


def aggregate_votes(
    votes: list[PmVote],
) -> tuple[ActionEnum, float, str, FailureMode | None]:
    """Combine three single-lens PmVotes into a committee decision.

    Aggregation rules (locked, see plan §T26):

    1. **3 of the same vote** (unanimous BUY/HOLD/SELL) → that vote.
    2. **2 BUY + 1 HOLD** → BUY, with reservation (confidence discounted).
    3. **2 BUY + 1 SELL** → ESCALATE (informative directional split).
    4. **1 BUY + 2 SELL** → SELL, with reservation.
    5. **3 HOLD** → HOLD (subsumed by rule 1).
    6. **1 BUY + 1 HOLD + 1 SELL** → ESCALATE (full disagreement).

    Consistent extensions for the remaining 2-1 multisets:

    * 2 SELL + 1 HOLD → SELL, with reservation (parallel to rule 2).
    * 2 HOLD + 1 BUY  → HOLD (HOLD majority dominates a minority directional).
    * 2 HOLD + 1 SELL → HOLD (HOLD majority dominates a minority directional).

    Confidence formula:

    * **Unanimous:** ``mean(confidence)`` of all three voters.
    * **2-1 majority (incl. with-reservation cases):**
      ``mean(confidence of the 2 majority voters) * 0.8``.  The lone
      dissenter's confidence is not counted; its rationale is still preserved
      in the combined rationale string for downstream review.
    * **ESCALATE outcomes:** ``mean(confidence)`` of all three voters, used as
      an alignment signal; the action is ESCALATE regardless.

    Rationale: per-lens rationales are joined with lens labels, ordered
    QUALITY → VALUATION → CATALYST, so T27 can both store the combined
    string on ``Decision.rationale`` and split it out into
    ``Decision.metadata`` if desired.

    The returned ``FailureMode | None`` slot is always ``None`` in the
    deterministic happy path.  It exists so T27 can propagate failure
    detections (e.g., empty/malformed inputs surfaced by upstream voters)
    through the same return contract.

    Parameters
    ----------
    votes:
        Exactly three :class:`PmVote` objects, one per :class:`PmLens`.

    Returns
    -------
    tuple
        ``(action, confidence, rationale, failure_mode)`` — see formulas above.

    Raises
    ------
    ValueError
        If ``len(votes) != 3``.  Duplicate-lens guards are the caller's
        responsibility.
    """
    if len(votes) != 3:
        raise ValueError(
            f"aggregate_votes requires exactly 3 votes (one per lens); got {len(votes)}"
        )

    # ----- Tally the multiset (BUY, HOLD, SELL). ---------------------------
    by_action: dict[ActionEnum, list[PmVote]] = {
        ActionEnum.BUY: [],
        ActionEnum.HOLD: [],
        ActionEnum.SELL: [],
    }
    for v in votes:
        by_action[v.vote].append(v)

    buy_count = len(by_action[ActionEnum.BUY])
    hold_count = len(by_action[ActionEnum.HOLD])
    sell_count = len(by_action[ActionEnum.SELL])
    counts = (buy_count, hold_count, sell_count)

    # ----- Dispatch by multiset.  Table is 10 cases, fully enumerated. -----
    # Each entry decides which voter-subset's confidences average into the
    # final score; the discount is applied for 2-1 majorities only.
    mean_all = sum(v.confidence for v in votes) / 3.0

    def _mean(subset: list[PmVote]) -> float:
        return sum(v.confidence for v in subset) / len(subset)

    action: ActionEnum
    confidence: float
    if counts == (3, 0, 0):  # Rule 1: unanimous BUY
        action, confidence = ActionEnum.BUY, mean_all
    elif counts == (0, 3, 0):  # Rule 1 / 5: unanimous HOLD
        action, confidence = ActionEnum.HOLD, mean_all
    elif counts == (0, 0, 3):  # Rule 1: unanimous SELL
        action, confidence = ActionEnum.SELL, mean_all
    elif counts == (2, 1, 0):  # Rule 2: 2 BUY + 1 HOLD → BUY w/ reservation
        action = ActionEnum.BUY
        confidence = _mean(by_action[ActionEnum.BUY]) * _MAJORITY_DISSENT_DISCOUNT
    elif counts == (0, 1, 2):  # Extension: 2 SELL + 1 HOLD → SELL w/ reservation
        action = ActionEnum.SELL
        confidence = _mean(by_action[ActionEnum.SELL]) * _MAJORITY_DISSENT_DISCOUNT
    elif counts == (1, 0, 2):  # Rule 4: 1 BUY + 2 SELL → SELL w/ reservation
        action = ActionEnum.SELL
        confidence = _mean(by_action[ActionEnum.SELL]) * _MAJORITY_DISSENT_DISCOUNT
    elif counts == (1, 2, 0):  # Extension: 2 HOLD + 1 BUY → HOLD
        action = ActionEnum.HOLD
        confidence = _mean(by_action[ActionEnum.HOLD]) * _MAJORITY_DISSENT_DISCOUNT
    elif counts == (0, 2, 1):  # Extension: 2 HOLD + 1 SELL → HOLD
        action = ActionEnum.HOLD
        confidence = _mean(by_action[ActionEnum.HOLD]) * _MAJORITY_DISSENT_DISCOUNT
    elif counts == (2, 0, 1):  # Rule 3: 2 BUY + 1 SELL → ESCALATE
        action, confidence = ActionEnum.ESCALATE, mean_all
    elif counts == (1, 1, 1):  # Rule 6: full disagreement → ESCALATE
        action, confidence = ActionEnum.ESCALATE, mean_all
    else:  # pragma: no cover — exhaustive over multisets of size 3 over {B,H,S}.
        raise AssertionError(f"unreachable: vote multiset {counts}")

    # ----- Build the combined rationale, ordered QUALITY → VAL → CATALYST. -
    by_lens: dict[PmLens, PmVote] = {v.lens: v for v in votes}
    rationale = "\n".join(
        f"[{lens.value}] {by_lens[lens].rationale}"
        for lens in _LENS_ORDER
        if lens in by_lens
    )

    return action, confidence, rationale, None


# ---------------------------------------------------------------------------
# T27 — make_pm: vote-of-3 + aggregation node
# ---------------------------------------------------------------------------


# Default share quantity used when the committee flips into a SELL whose
# research counterpart did not carry a shares figure (e.g. research voted
# HOLD).  Plan 2 defers real position sizing to Risk in Plan 3; the value
# here is just a non-zero placeholder that satisfies the Decimal-typed
# Sell/Buy payload schema.
_DEFAULT_SHARES_PLACEHOLDER = Decimal("1")


def _ticker_from_payload(payload: TypedPayload) -> str | None:
    """Extract the ticker carried by a typed payload, if any.

    BuyPayload and SellPayload carry ``ticker`` directly.  EscalatePayload
    carries a nested proposed Buy/Sell.  HoldPayload and RefusePayload have
    no ticker — return ``None`` for those.
    """
    if isinstance(payload, (BuyPayload, SellPayload)):
        return payload.ticker
    if isinstance(payload, EscalatePayload):
        return payload.proposed.ticker
    return None


def _shares_from_payload(payload: TypedPayload) -> Decimal:
    """Pull a share count off a payload, falling back to the placeholder."""
    if isinstance(payload, (BuyPayload, SellPayload)):
        return payload.shares
    if isinstance(payload, EscalatePayload):
        return payload.proposed.shares
    return _DEFAULT_SHARES_PLACEHOLDER


def _payload_for(action: ActionEnum, research: Decision) -> TypedPayload:
    """Build the PM Decision's payload for the aggregated ``action``.

    Plan 2 simplification: when the aggregated action matches the research
    action, reuse the research payload verbatim.  When the committee flips
    direction (e.g. research BUY but committee aggregates to HOLD), build a
    fresh payload of the matching type carrying the same ticker / shares.
    Real position sizing is deferred to Risk in Plan 3.
    """
    if action == research.action and not isinstance(research.payload, EscalatePayload):
        # Reuse research's typed payload directly for matching directional cases.
        return research.payload

    ticker = _ticker_from_payload(research.payload) or "<unknown>"
    shares = _shares_from_payload(research.payload)

    if action == ActionEnum.BUY:
        return BuyPayload(ticker=ticker, shares=shares)
    if action == ActionEnum.SELL:
        return SellPayload(ticker=ticker, shares=shares)
    if action == ActionEnum.HOLD:
        return HoldPayload(reason="PM committee aggregated to HOLD")
    if action == ActionEnum.ESCALATE:
        return EscalatePayload(
            proposed=BuyPayload(ticker=ticker, shares=shares),
            reason="PM committee directional disagreement",
        )
    # REFUSE is not produced by aggregate_votes; pass-through path handles it.
    raise AssertionError(f"unexpected aggregated action: {action}")  # pragma: no cover


async def _vote_parallel(
    voter: PmVoter,
    *,
    question: str,
    claims: list[Claim],
    research_rationale: str,
    voter_model: str,
) -> tuple[list[PmVote], list[float]]:
    """Run three single-lens voters concurrently via asyncio.gather + to_thread.

    Concurrency cap = 3 (one task per lens). Each task wraps voter.vote in its
    own llm_span so per-call cost/timing attribution survives.  OTel ContextVars
    are propagated by asyncio.to_thread so each llm_span nests under the caller's
    agent_span("pm") context.

    Returns
    -------
    tuple
        ``(votes, per_voter_ms)``.  ``per_voter_ms[i]`` is the wall-clock
        cost of the i-th voter's call (in the same order as ``votes``) so
        the caller can stamp the OTel parent span with a measurement-based
        sequential-estimate instead of a multiplicative guess.

        Order guarantee: ``asyncio.gather`` preserves submission order, so
        ``results[i]`` always corresponds to ``coros[i]`` — and therefore
        ``votes[i]`` and ``per_voter_ms[i]`` align with the lens at index
        ``i`` in the QUALITY / VALUATION / CATALYST list below.
    """

    def _one(lens: PmLens) -> tuple[PmVote, float]:
        t0 = time.perf_counter()
        with llm_span(_PROVIDER_ANTHROPIC, voter_model):
            vote = voter.vote(
                lens=lens,
                question=question,
                claims=claims,
                research_rationale=research_rationale,
            )
        return vote, (time.perf_counter() - t0) * 1000.0

    coros = [
        asyncio.to_thread(_one, lens)
        for lens in (PmLens.QUALITY, PmLens.VALUATION, PmLens.CATALYST)
    ]
    results = list(await asyncio.gather(*coros))
    votes = [r[0] for r in results]
    per_voter_ms = [r[1] for r in results]
    return votes, per_voter_ms


def make_pm(
    voter: PmVoter,
    *,
    router: CostRouter | None = None,
) -> Callable[[WorkingState], dict[str, Any]]:
    """Build the PM node callable.

    For each heartbeat the node:

    1. Inspects ``state["research_decision"]``.  If the research action is
       REFUSE or ESCALATE, the PM does not vote — it emits a pass-through
       Decision chaining the research id and tagged with ``passthrough=True``
       so downstream nodes can distinguish.
    2. Otherwise, deserializes ``state["claims"]`` back into a
       ``list[Claim]`` and runs three sequential single-lens votes
       (quality, valuation, catalyst) through ``voter``.  Plan 2 keeps the
       calls sequential; Plan 3 introduces concurrency.
    3. Aggregates via :func:`aggregate_votes` into ``(action, confidence,
       combined_rationale, failure_mode)``.
    4. Builds a typed payload matching the aggregated action via
       :func:`_payload_for`, propagates ``oldest_filing_age_days`` from
       research's metadata when present, and emits the PM Decision chained
       to ``research.id`` while copying ``research.citations`` and
       ``research.falsification_condition``.
    5. Writes ``pm_votes`` to state as a list of dump dicts (pass-through
       branches write an empty list).

    The ``question`` argument required by :meth:`PmVoter.vote` is derived
    from ``research.rationale`` — the research Decision does not carry the
    original question as its own field, and rationale is the closest
    available proxy.  Plan 3 may surface the question on working state
    explicitly so the PM voter receives it verbatim.

    ``router`` (T08): when provided AND the voter's client is a
    :class:`RouterBackedMessagesClient`, each heartbeat binds the voter's
    client to a profile (sonnet by default; opus only when the sufficiency
    judge returned ``partial`` AND ``state["human_override_ack"]`` is truthy
    — see TODO below). When ``router`` is ``None``, the voter calls its
    client directly with no routing layer (pre-T08 behavior).

    Catches :class:`LLMUnavailableError` raised by the voter loop and emits
    a REFUSE Decision with ``failure_mode=LLM_UNAVAILABLE`` and a
    conservative "all-models-exhausted" payload (T08 spec).
    """

    # Read the model id off the voter so each ``llm.call`` span records the
    # actual model that handled the request.  ``_model`` is a leading-
    # underscore implementation attribute; treating it as read-only from
    # the agent layer keeps PmVoter's public interface unchanged (T03 must
    # not restructure the LLM client interface).
    voter_model: str = getattr(voter, "_model", "unknown")

    def pm(state: WorkingState) -> dict[str, Any]:
        # T03: CM form (not decorator) so each return branch can stamp
        # ``failure_mode`` and ``decision_id`` onto the agent span.  Mirrors
        # the pattern used by ``firm/cli.py`` risk_node.
        with agent_span("pm") as span:
            research: Decision = state["research_decision"]

            # ---- Pass-through path: research already terminated the heartbeat.
            if research.action in (ActionEnum.REFUSE, ActionEnum.ESCALATE):
                passthrough_metadata: dict[str, Any] = {
                    "agent": "pm",
                    "passthrough": True,
                }
                oldest_age = research.metadata.get("oldest_filing_age_days")
                if oldest_age is not None:
                    passthrough_metadata["oldest_filing_age_days"] = oldest_age
                passthrough = Decision(
                    id=ulid_new(),
                    decision_id_chain=[research.id],
                    action=research.action,
                    payload=research.payload,
                    rationale=research.rationale,
                    confidence=research.confidence,
                    citations=research.citations,
                    falsification_condition=research.falsification_condition,
                    escalation_reason=research.escalation_reason,
                    failure_mode=research.failure_mode,
                    metadata=passthrough_metadata,
                    nonce="pm",
                )
                stamp_decision(span, passthrough.id, passthrough.failure_mode)
                return {"pm_decision": passthrough, "pm_votes": []}

            # ---- Vote path: three sequential single-lens votes, then aggregate.
            claims_dicts: list[dict[str, Any]] = list(state.get("claims", []))
            claims: list[Claim] = [Claim.model_validate(c) for c in claims_dicts]
            # Use research.rationale as the question proxy (see docstring).
            question = research.rationale

            # T08 wiring: hoisted ahead of the voter loop so the same id is
            # used to bind the router client AND to stamp the emitted PM
            # Decision below. Ledger rows written by the router during
            # voting then attribute back to this exact PM Decision id.
            pm_decision_id = ulid_new()

            if router is not None:
                # Opus escalation hook: PARTIAL sufficiency + human-ack
                # override means we're voting on a HITL-approved marginal
                # idea, where opus's deeper reasoning is worth the cost.
                # TODO(T13): ``human_override_ack`` is populated by the
                # HITL approval flow (Section C T13/T14). The branch is
                # dormant today; the synthetic-state test in
                # tests/integration/test_router_wired_e2e.py exercises it.
                sufficiency_status = state.get("sufficiency_status")
                human_ack = state.get("human_override_ack")
                profile: ProfileName = (
                    _PM_ESCALATED_PROFILE
                    if sufficiency_status == "partial" and bool(human_ack)
                    else _PM_DEFAULT_PROFILE
                )
                voter_client = getattr(voter, "_client", None)
                if isinstance(voter_client, RouterBackedMessagesClient):
                    voter_client.bind(
                        profile=profile,
                        decision_id=pm_decision_id,
                        agent=_PM_AGENT_NAME,
                    )

            t_start = time.perf_counter()
            votes: list[PmVote] = []
            per_voter_ms: list[float] = []
            try:
                votes, per_voter_ms = asyncio.run(
                    _vote_parallel(
                        voter,
                        question=question,
                        claims=claims,
                        research_rationale=research.rationale,
                        voter_model=voter_model,
                    )
                )
            except LLMUnavailableError as exc:
                # T08 spec: PM REFUSEs with LLM_UNAVAILABLE + the conservative
                # "all-models-exhausted" payload when the router ladder is
                # exhausted. Distinct rationale string so dashboards can
                # split router exhaustion from PmVoteSchemaError-driven
                # SCHEMA_VALIDATION_FAILED REFUSEs.
                exhausted_metadata: dict[str, Any] = {"agent": "pm"}
                oldest_age = research.metadata.get("oldest_filing_age_days")
                if oldest_age is not None:
                    exhausted_metadata["oldest_filing_age_days"] = oldest_age
                router_refuse = Decision(
                    id=pm_decision_id,
                    decision_id_chain=[research.id],
                    action=ActionEnum.REFUSE,
                    payload=RefusePayload(reason="all-models-exhausted"),
                    rationale=f"all model profiles exhausted: {exc!s}",
                    confidence=0.0,
                    citations=list(research.citations),
                    falsification_condition=research.falsification_condition,
                    escalation_reason=None,
                    failure_mode=FailureMode.LLM_UNAVAILABLE,
                    metadata=exhausted_metadata,
                    nonce="pm",
                )
                stamp_decision(span, router_refuse.id, router_refuse.failure_mode)
                return {
                    "pm_decision": router_refuse,
                    "pm_votes": [v.model_dump(mode="json") for v in votes],
                }
            except PmVoteSchemaError as exc:
                # Mirror research.py's JudgeSchemaError path: surface as a REFUSE
                # Decision with SCHEMA_VALIDATION_FAILED so a single malformed
                # voter response does not crash the heartbeat.
                schema_metadata: dict[str, Any] = {"agent": "pm"}
                oldest_age = research.metadata.get("oldest_filing_age_days")
                if oldest_age is not None:
                    schema_metadata["oldest_filing_age_days"] = oldest_age
                schema_refuse = Decision(
                    id=pm_decision_id,
                    decision_id_chain=[research.id],
                    action=ActionEnum.REFUSE,
                    payload=RefusePayload(reason="pm:schema_validation_failed"),
                    rationale=f"PM voter response failed schema validation: {exc!s}",
                    confidence=0.0,
                    citations=list(research.citations),
                    falsification_condition=research.falsification_condition,
                    escalation_reason=None,
                    failure_mode=FailureMode.SCHEMA_VALIDATION_FAILED,
                    metadata=schema_metadata,
                    nonce="pm",
                )
                stamp_decision(span, schema_refuse.id, schema_refuse.failure_mode)
                return {
                    "pm_decision": schema_refuse,
                    "pm_votes": [v.model_dump(mode="json") for v in votes],
                }

            # Spec T24: stamp parent span with parallel timing so the OTel
            # collector surfaces the wall-clock savings.  sequential_estimate
            # is the sum of per-voter elapsed times (measured, not multiplied)
            # so cached/live mixes report the honest delta.
            parallel_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            sequential_estimate_ms = sum(per_voter_ms)
            span.set_attribute("pm.voter_count", 3)
            span.set_attribute("pm.parallel_ms", round(parallel_elapsed_ms, 2))
            span.set_attribute(
                "pm.sequential_estimate_ms", round(sequential_estimate_ms, 2)
            )
            span.set_attribute(
                "pm.latency_delta_ms",
                round(sequential_estimate_ms - parallel_elapsed_ms, 2),
            )

            action, confidence, combined_rationale, fmode = aggregate_votes(votes)
            payload = _payload_for(action, research)

            pm_metadata: dict[str, Any] = {"agent": "pm"}
            oldest_age = research.metadata.get("oldest_filing_age_days")
            if oldest_age is not None:
                pm_metadata["oldest_filing_age_days"] = oldest_age

            escalation_reason = (
                "PM committee aggregated to ESCALATE"
                if action == ActionEnum.ESCALATE
                else None
            )

            decision = Decision(
                id=pm_decision_id,
                decision_id_chain=[research.id],
                action=action,
                payload=payload,
                rationale=combined_rationale,
                confidence=confidence,
                citations=list(research.citations),
                falsification_condition=research.falsification_condition,
                escalation_reason=escalation_reason,
                failure_mode=fmode,
                metadata=pm_metadata,
                nonce="pm",
            )
            stamp_decision(span, decision.id, decision.failure_mode)
            return {
                "pm_decision": decision,
                "pm_votes": [v.model_dump(mode="json") for v in votes],
            }

    return pm


__all__ = [
    "PmLens",
    "PmVote",
    "PmVoteSchemaError",
    "PmVoter",
    "aggregate_votes",
    "make_pm",
]
