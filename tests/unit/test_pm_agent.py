"""Tests for PmVoter, PmVote, PmLens (Plan 2 §T25) and make_pm (Plan 2 §T27).

These tests follow TDD: written before the implementation.  All network calls
are intercepted by a recording stub satisfying ``AnthropicMessagesClient``.
"""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from firm.agents.pm import PmLens, PmVote, PmVoteSchemaError, PmVoter, make_pm
from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Claim,
    Decision,
    EscalatePayload,
    HoldPayload,
    RefusePayload,
)
from firm.orchestrator.state import WorkingState


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _StubClient:
    """Recording stub satisfying the AnthropicMessagesClient Protocol."""

    def __init__(self, raw_text: str) -> None:
        self._raw_text = raw_text
        self.last_kwargs: dict[str, object] | None = None

    def messages_create(self, **kwargs: object) -> dict[str, object]:
        self.last_kwargs = kwargs
        return {
            "content": [
                {"type": "text", "text": self._raw_text},
            ]
        }


def _make_claims(n: int) -> list[Claim]:
    """Return n minimal Claim objects."""
    return [Claim(text=f"Claim {i + 1} text.", source_chunk_id=f"chunk-{i}") for i in range(n)]


def _make_voter(raw_text: str) -> tuple[PmVoter, _StubClient]:
    stub = _StubClient(raw_text)
    voter = PmVoter(client=stub, model="claude-sonnet-4-6")
    return voter, stub


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_voter_quality_lens_produces_vote_with_rationale_and_cited_claim_ids() -> None:
    """Happy path: quality lens BUY with two cited claims."""
    raw = json.dumps(
        {
            "vote": "BUY",
            "confidence": 0.8,
            "rationale": "Wide moat plus high ROIC.",
            "cited_claim_ids": ["c1", "c2"],
        }
    )
    voter, _ = _make_voter(raw)
    claims = _make_claims(2)

    result = voter.vote(
        lens=PmLens.QUALITY,
        question="Is AAPL a quality business?",
        claims=claims,
        research_rationale="Strong financials across periods.",
    )

    assert isinstance(result, PmVote)
    assert result.lens == PmLens.QUALITY
    assert result.vote == ActionEnum.BUY
    assert result.confidence == 0.8
    assert len(result.rationale) > 0
    assert result.cited_claim_ids == ["c1", "c2"]


def test_voter_returns_buy_hold_sell_only() -> None:
    """ESCALATE is not a valid vote; PmVoteSchemaError must be raised."""
    raw = json.dumps(
        {
            "vote": "ESCALATE",
            "confidence": 0.5,
            "rationale": "Unclear picture.",
            "cited_claim_ids": [],
        }
    )
    voter, _ = _make_voter(raw)
    claims = _make_claims(1)

    with pytest.raises(PmVoteSchemaError):
        voter.vote(
            lens=PmLens.VALUATION,
            question="Is AAPL cheap?",
            claims=claims,
            research_rationale="Mixed valuation signals.",
        )


def test_voter_rejects_claim_ids_not_in_input_set() -> None:
    """cited_claim_ids outside the input set (c1, c2) are silently dropped."""
    raw = json.dumps(
        {
            "vote": "HOLD",
            "confidence": 0.5,
            "rationale": "Uncertain.",
            "cited_claim_ids": ["c1", "c99", "phantom"],
        }
    )
    voter, _ = _make_voter(raw)
    claims = _make_claims(2)  # ids: c1, c2

    result = voter.vote(
        lens=PmLens.CATALYST,
        question="Is there a catalyst for AAPL?",
        claims=claims,
        research_rationale="No clear near-term trigger.",
    )

    assert result.cited_claim_ids == ["c1"]


def test_voter_uses_correct_prompt_per_lens() -> None:
    """Each lens name appears as a substring in the system prompt sent to the LLM."""
    raw = json.dumps(
        {
            "vote": "HOLD",
            "confidence": 0.4,
            "rationale": "Lens-specific reasoning.",
            "cited_claim_ids": [],
        }
    )
    claims = _make_claims(1)

    for lens in PmLens:
        stub = _StubClient(raw)
        voter = PmVoter(client=stub, model="claude-sonnet-4-6")
        voter.vote(
            lens=lens,
            question="Test question.",
            claims=claims,
            research_rationale="Test rationale.",
        )
        assert stub.last_kwargs is not None
        system_prompt = stub.last_kwargs.get("system")
        assert isinstance(system_prompt, str), f"system must be str for {lens}"
        assert lens.value in system_prompt, (
            f"Lens name '{lens.value}' not found in system prompt for {lens}"
        )


# ---------------------------------------------------------------------------
# Hygiene tests
# ---------------------------------------------------------------------------


def test_voter_raises_on_malformed_json() -> None:
    """Non-JSON response must raise PmVoteSchemaError."""
    voter, _ = _make_voter("not json at all")
    claims = _make_claims(1)

    with pytest.raises(PmVoteSchemaError):
        voter.vote(
            lens=PmLens.QUALITY,
            question="q",
            claims=claims,
            research_rationale="r",
        )


def test_voter_strips_markdown_fences() -> None:
    """Markdown-fenced JSON must be parsed successfully."""
    inner = json.dumps(
        {
            "vote": "SELL",
            "confidence": 0.7,
            "rationale": "Overvalued per DCF.",
            "cited_claim_ids": ["c1"],
        }
    )
    raw = f"```json\n{inner}\n```"
    voter, _ = _make_voter(raw)
    claims = _make_claims(1)

    result = voter.vote(
        lens=PmLens.VALUATION,
        question="Is AAPL overvalued?",
        claims=claims,
        research_rationale="High multiples.",
    )

    assert result.vote == ActionEnum.SELL
    assert result.confidence == pytest.approx(0.7)


def test_pm_vote_re_instantiation_runs_validator() -> None:
    """Regression: filtering cited_claim_ids must not bypass the BUY/HOLD/SELL validator.

    model_copy(update=...) silently bypasses validators; model_validate({**dump, ...})
    does not. This guards against a future maintainer regressing the path.
    """
    base = PmVote(
        lens=PmLens.QUALITY,
        vote=ActionEnum.BUY,
        confidence=0.7,
        rationale="ok",
        cited_claim_ids=["c1"],
    )
    # The path used inside vote(): if we dump+validate with a bad vote we MUST get a ValidationError.
    with pytest.raises(ValidationError):
        PmVote.model_validate({**base.model_dump(), "vote": ActionEnum.ESCALATE})


# ---------------------------------------------------------------------------
# T27 — make_pm(voter) vote-of-3 + aggregation
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Recording AnthropicMessagesClient stub that returns a queue of responses.

    Each ``messages_create`` call pops the next response from ``responses``
    (text strings) and records the full kwargs in ``calls``.  Falls back to
    the last response if the queue is exhausted, so a single fixed response
    can be supplied for all calls.
    """

    def __init__(self, responses: list[str]) -> None:
        if not responses:
            raise ValueError("at least one response required")
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self._last_text: str = responses[-1]

    def messages_create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        text = self._responses.pop(0) if self._responses else self._last_text
        self._last_text = text
        return {"content": [{"type": "text", "text": text}]}


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
            "cited_claim_ids": cited_claim_ids if cited_claim_ids is not None else ["c1"],
        }
    )


def _research_buy(
    *,
    metadata: dict[str, Any] | None = None,
) -> Decision:
    """Build a default BUY research Decision with one citation-able claim."""
    md = {"agent": "research", "ticker": "AAPL"}
    if metadata is not None:
        md.update(metadata)
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
        metadata=md,
        nonce="research-nonce",
    )


def _research_refuse() -> Decision:
    return Decision(
        id="res-refuse-1",
        decision_id_chain=[],
        action=ActionEnum.REFUSE,
        payload=RefusePayload(reason="sufficiency:insufficient"),
        rationale="judge marked claims UNSUPPORTED",
        confidence=0.0,
        citations=[],
        falsification_condition="claims become supported later",
        escalation_reason=None,
        failure_mode=None,
        metadata={"agent": "research", "ticker": "AAPL"},
        nonce="research-nonce",
    )


def _research_escalate() -> Decision:
    return Decision(
        id="res-esc-1",
        decision_id_chain=[],
        action=ActionEnum.ESCALATE,
        payload=EscalatePayload(
            proposed=BuyPayload(ticker="AAPL", shares=Decimal("10")),
            reason="sufficiency:partial",
        ),
        rationale="partial sufficiency; escalating",
        confidence=0.4,
        citations=[],
        falsification_condition="claims become fully supported",
        escalation_reason="sufficiency:partial",
        failure_mode=None,
        metadata={"agent": "research", "ticker": "AAPL"},
        nonce="research-nonce",
    )


def _research_hold() -> Decision:
    return Decision(
        id="res-hold-1",
        decision_id_chain=[],
        action=ActionEnum.HOLD,
        payload=HoldPayload(reason="research holds"),
        rationale="research holds; no clear directional signal",
        confidence=0.5,
        citations=[],
        falsification_condition="signal clarifies at a later heartbeat",
        escalation_reason=None,
        failure_mode=None,
        metadata={"agent": "research"},
        nonce="research-nonce",
    )


def _claim_dicts(n: int) -> list[dict[str, Any]]:
    return [Claim(text=f"Claim {i+1}.", source_chunk_id=f"chunk-{i}").model_dump() for i in range(n)]


def test_pm_runs_three_voters_in_parallel_and_aggregates() -> None:
    """Three voters fire (sequentially in Plan 2) with three distinct lens prompts.

    Plan 2 calls voters sequentially; the "in_parallel" name is a holdover from the
    spec — what's asserted is the call count and that each system prompt is lens-
    specific.
    """
    responses = [
        _vote_json("BUY", 0.8, "quality says BUY"),
        _vote_json("BUY", 0.7, "valuation says BUY"),
        _vote_json("BUY", 0.6, "catalyst says BUY"),
    ]
    client = _RecordingClient(responses)
    voter = PmVoter(client=client, model="claude-sonnet-4-6")
    pm = make_pm(voter)

    state: WorkingState = {
        "research_decision": _research_buy(),
        "claims": _claim_dicts(2),
    }
    out = pm(state)

    assert len(client.calls) == 3
    systems = [call["system"] for call in client.calls]
    assert len(set(systems)) == 3, "each lens must produce a distinct system prompt"
    assert any("quality" in s for s in systems)
    assert any("valuation" in s for s in systems)
    assert any("catalyst" in s for s in systems)
    assert out["pm_decision"].action == ActionEnum.BUY


def test_pm_emits_decision_with_aggregated_action_and_combined_rationale() -> None:
    """Aggregated action plus a combined rationale that names every lens contribution."""
    responses = [
        _vote_json("BUY", 0.9, "q-rationale-text"),
        _vote_json("BUY", 0.8, "v-rationale-text"),
        _vote_json("BUY", 0.7, "c-rationale-text"),
    ]
    client = _RecordingClient(responses)
    voter = PmVoter(client=client, model="claude-sonnet-4-6")
    pm = make_pm(voter)

    state: WorkingState = {
        "research_decision": _research_buy(),
        "claims": _claim_dicts(2),
    }
    out = pm(state)
    decision: Decision = out["pm_decision"]

    assert decision.action == ActionEnum.BUY
    # Unanimous BUY -> mean confidence
    assert decision.confidence == pytest.approx((0.9 + 0.8 + 0.7) / 3.0)
    assert "q-rationale-text" in decision.rationale
    assert "v-rationale-text" in decision.rationale
    assert "c-rationale-text" in decision.rationale
    assert "res-1" in decision.decision_id_chain
    assert decision.metadata["agent"] == "pm"


def test_pm_state_carries_pm_votes_list() -> None:
    """pm_votes is a length-3 list of dump dicts, one per lens."""
    responses = [
        _vote_json("BUY", 0.8, "q"),
        _vote_json("HOLD", 0.5, "v"),
        _vote_json("BUY", 0.6, "c"),
    ]
    client = _RecordingClient(responses)
    voter = PmVoter(client=client, model="claude-sonnet-4-6")
    pm = make_pm(voter)

    state: WorkingState = {
        "research_decision": _research_buy(),
        "claims": _claim_dicts(1),
    }
    out = pm(state)

    votes = out["pm_votes"]
    assert isinstance(votes, list)
    assert len(votes) == 3
    required_keys = {"lens", "vote", "confidence", "rationale", "cited_claim_ids"}
    for v in votes:
        assert required_keys.issubset(v.keys())
    lenses = {v["lens"] for v in votes}
    assert lenses == {PmLens.QUALITY.value, PmLens.VALUATION.value, PmLens.CATALYST.value}


def test_pm_falls_through_when_research_action_is_refuse() -> None:
    """REFUSE input short-circuits voting; PM emits REFUSE pass-through, no voter calls."""
    client = _RecordingClient([_vote_json()])
    voter = PmVoter(client=client, model="claude-sonnet-4-6")
    pm = make_pm(voter)

    research = _research_refuse()
    state: WorkingState = {"research_decision": research, "claims": []}
    out = pm(state)

    assert len(client.calls) == 0, "REFUSE input must not invoke voters"
    decision: Decision = out["pm_decision"]
    assert decision.action == ActionEnum.REFUSE
    assert "res-refuse-1" in decision.decision_id_chain
    assert decision.metadata["agent"] == "pm"
    # pm_votes should be present but empty (or omitted); we accept empty list.
    assert out.get("pm_votes", []) == []


def test_pm_handles_escalate_research_input() -> None:
    """ESCALATE input also short-circuits voting; PM passes ESCALATE through."""
    client = _RecordingClient([_vote_json()])
    voter = PmVoter(client=client, model="claude-sonnet-4-6")
    pm = make_pm(voter)

    research = _research_escalate()
    state: WorkingState = {"research_decision": research, "claims": _claim_dicts(1)}
    out = pm(state)

    assert len(client.calls) == 0, "ESCALATE input must not invoke voters"
    decision: Decision = out["pm_decision"]
    assert decision.action == ActionEnum.ESCALATE
    assert "res-esc-1" in decision.decision_id_chain
    assert decision.metadata["agent"] == "pm"
    assert out.get("pm_votes", []) == []


def test_pm_propagates_oldest_filing_age_days() -> None:
    """PM Decision metadata mirrors research's oldest_filing_age_days."""
    responses = [
        _vote_json("BUY", 0.8, "q"),
        _vote_json("BUY", 0.7, "v"),
        _vote_json("BUY", 0.6, "c"),
    ]
    client = _RecordingClient(responses)
    voter = PmVoter(client=client, model="claude-sonnet-4-6")
    pm = make_pm(voter)

    research = _research_buy(metadata={"oldest_filing_age_days": 7})
    state: WorkingState = {
        "research_decision": research,
        "claims": _claim_dicts(1),
    }
    out = pm(state)
    decision: Decision = out["pm_decision"]
    assert decision.metadata.get("oldest_filing_age_days") == 7


def test_pm_builds_fresh_payload_when_aggregated_differs_from_research() -> None:
    """If aggregate flips BUY → HOLD, PM rebuilds the payload to match (Plan 2 simplification)."""
    # 2 HOLD + 1 BUY → HOLD per the extension rule.
    responses = [
        _vote_json("HOLD", 0.6, "q-hold"),
        _vote_json("HOLD", 0.5, "v-hold"),
        _vote_json("BUY", 0.7, "c-buy"),
    ]
    client = _RecordingClient(responses)
    voter = PmVoter(client=client, model="claude-sonnet-4-6")
    pm = make_pm(voter)

    state: WorkingState = {
        "research_decision": _research_buy(),
        "claims": _claim_dicts(1),
    }
    out = pm(state)
    decision: Decision = out["pm_decision"]
    assert decision.action == ActionEnum.HOLD
    assert isinstance(decision.payload, HoldPayload)


def test_pm_uses_unknown_ticker_when_hold_research_aggregates_to_buy() -> None:
    """Research HOLD has no ticker on payload; committee BUY → ticker=<unknown>, shares=1."""
    responses = [
        _vote_json("BUY", 0.8, "q-buy"),
        _vote_json("BUY", 0.7, "v-buy"),
        _vote_json("HOLD", 0.5, "c-hold"),
    ]
    client = _RecordingClient(responses)
    voter = PmVoter(client=client, model="claude-sonnet-4-6")
    pm = make_pm(voter)

    state: WorkingState = {
        "research_decision": _research_hold(),
        "claims": _claim_dicts(1),
    }
    out = pm(state)
    decision: Decision = out["pm_decision"]
    assert decision.action == ActionEnum.BUY
    assert isinstance(decision.payload, BuyPayload)
    assert decision.payload.ticker == "<unknown>"
    assert decision.payload.shares == Decimal("1")


def test_pm_escalates_with_escalate_payload_on_committee_disagreement() -> None:
    """1B/1H/1S → ESCALATE; payload is an EscalatePayload, escalation_reason is set."""
    responses = [
        _vote_json("BUY", 0.8, "q-buy"),
        _vote_json("HOLD", 0.6, "v-hold"),
        _vote_json("SELL", 0.7, "c-sell"),
    ]
    client = _RecordingClient(responses)
    voter = PmVoter(client=client, model="claude-sonnet-4-6")
    pm = make_pm(voter)

    state: WorkingState = {
        "research_decision": _research_buy(),
        "claims": _claim_dicts(1),
    }
    out = pm(state)
    decision: Decision = out["pm_decision"]
    assert decision.action == ActionEnum.ESCALATE
    assert isinstance(decision.payload, EscalatePayload)
    assert decision.escalation_reason is not None
    assert "PM committee" in decision.escalation_reason
