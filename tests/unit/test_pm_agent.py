"""Tests for PmVoter, PmVote, PmLens (Plan 2 §T25).

These tests follow TDD: written before the implementation.  All network calls
are intercepted by a recording stub satisfying ``AnthropicMessagesClient``.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from firm.agents.pm import PmLens, PmVote, PmVoteSchemaError, PmVoter
from firm.core.models import ActionEnum, Claim


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
