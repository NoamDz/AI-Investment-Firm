"""Tests for the Haiku-backed sufficiency judge (Plan 2 §T20).

The judge takes a question + a list of cited claims and returns a
:class:`firm.grounding.schema.SufficiencyResult` whose
:meth:`SufficiencyResult.aggregate_status` summarises the worst-case label.

These tests use a recording stub client (no real SDK) and inline canned
response dicts in the shape produced by
:meth:`firm.llm.anthropic_client.CachedAnthropicClient.messages_create`.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from firm.core.models import Claim
from firm.grounding.judge import JudgeResponseError, SufficiencyJudge
from firm.grounding.schema import ClaimSupport, SufficiencyResult
from firm.llm.prompts import SUFFICIENCY_SYSTEM


class _StubClient:
    """Recording stub satisfying the ``AnthropicMessagesClient`` Protocol."""

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.last_kwargs: dict[str, object] | None = None
        self.call_count = 0

    def messages_create(self, **kwargs: object) -> dict[str, object]:
        self.last_kwargs = kwargs
        self.call_count += 1
        return self.response


def _text_response(payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap a JSON payload as a single text-block Anthropic response."""
    return {
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


def _claim(text: str, chunk_id: str) -> Claim:
    return Claim(text=text, source_chunk_id=chunk_id, source_span=(0, len(text)))


def test_judge_returns_all_supported_for_strong_claims() -> None:
    claims = [
        _claim("Apple FY2023 revenue was $383.3B.", "AAPL-10K::0001"),
        _claim("Apple FY2023 gross margin was 44.1%.", "AAPL-10K::0002"),
    ]
    payload = {
        "assessments": [
            {"claim_id": "c1", "status": "SUPPORTED", "rationale": "Direct figure."},
            {"claim_id": "c2", "status": "SUPPORTED", "rationale": "Direct figure."},
        ],
        "overall_reasoning": "All claims directly cited from filings.",
    }
    stub = _StubClient(_text_response(payload))
    judge = SufficiencyJudge(client=stub, model="claude-haiku-4-5", max_tokens=1024)

    result = judge.assess(question="What were Apple's FY2023 financials?", claims=claims)

    assert isinstance(result, SufficiencyResult)
    assert len(result.claim_assessments) == 2
    assert all(a.support == ClaimSupport.SUPPORTED for a in result.claim_assessments)
    assert result.aggregate_status() == "ok"
    assert result.overall_reasoning == "All claims directly cited from filings."


def test_judge_returns_partial_when_some_claims_partial() -> None:
    claims = [
        _claim("Apple FY2023 revenue was $383.3B.", "AAPL-10K::0001"),
        _claim("Apple's services business is growing.", "AAPL-10K::0003"),
    ]
    payload = {
        "assessments": [
            {"claim_id": "c1", "status": "SUPPORTED", "rationale": "Verbatim."},
            {"claim_id": "c2", "status": "PARTIAL", "rationale": "Qualitative only."},
        ],
        "overall_reasoning": "Second claim lacks quantitative backing.",
    }
    stub = _StubClient(_text_response(payload))
    judge = SufficiencyJudge(client=stub, model="claude-haiku-4-5")

    result = judge.assess(question="Summarise Apple's FY2023.", claims=claims)

    assert result.aggregate_status() == "partial"
    statuses = [a.support for a in result.claim_assessments]
    assert ClaimSupport.SUPPORTED in statuses
    assert ClaimSupport.PARTIAL in statuses


def test_judge_returns_unsupported_when_evidence_missing() -> None:
    claims = [
        _claim("Apple FY2023 revenue was $383.3B.", "AAPL-10K::0001"),
        _claim("Apple plans to acquire a major studio.", "AAPL-news::0004"),
    ]
    payload = {
        "assessments": [
            {"claim_id": "c1", "status": "SUPPORTED", "rationale": "Verbatim."},
            {
                "claim_id": "c2",
                "status": "UNSUPPORTED",
                "rationale": "Cited text is speculative blog post, not a filing.",
            },
        ],
        "overall_reasoning": "Second claim not backed by primary sources.",
    }
    stub = _StubClient(_text_response(payload))
    judge = SufficiencyJudge(client=stub, model="claude-haiku-4-5")

    result = judge.assess(question="What is Apple planning?", claims=claims)

    assert result.aggregate_status() == "insufficient"
    assert any(a.support == ClaimSupport.UNSUPPORTED for a in result.claim_assessments)


def test_judge_response_schema_validation_failure_raises() -> None:
    """Malformed JSON from Haiku must surface as :class:`JudgeResponseError`."""
    claims = [_claim("Apple FY2023 revenue was $383.3B.", "AAPL-10K::0001")]
    bad_response: dict[str, Any] = {
        "content": [{"type": "text", "text": "this is not json"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    stub = _StubClient(bad_response)
    judge = SufficiencyJudge(client=stub, model="claude-haiku-4-5")

    with pytest.raises(JudgeResponseError, match="JSON"):
        judge.assess(question="q", claims=claims)


def test_judge_passes_correct_system_prompt_and_kwargs() -> None:
    """Smoke-test the wire-up: system prompt, model, temperature 0.0, user msg."""
    claims = [_claim("Apple FY2023 revenue was $383.3B.", "AAPL-10K::0001")]
    payload = {
        "assessments": [
            {"claim_id": "c1", "status": "SUPPORTED", "rationale": "ok"},
        ],
        "overall_reasoning": "fine",
    }
    stub = _StubClient(_text_response(payload))
    judge = SufficiencyJudge(client=stub, model="claude-haiku-4-5", max_tokens=512)

    judge.assess(question="What were Apple's FY2023 financials?", claims=claims)

    assert stub.last_kwargs is not None
    kwargs = stub.last_kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert kwargs["system"] == SUFFICIENCY_SYSTEM
    assert kwargs["max_tokens"] == 512
    assert kwargs["temperature"] == 0.0

    messages = kwargs["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, dict)
    assert msg["role"] == "user"
    content = msg["content"]
    assert isinstance(content, str)
    # Synthetic claim IDs are 1-indexed and reference the claim text.
    assert "c1" in content
    assert "Apple FY2023 revenue was $383.3B." in content
    assert "What were Apple's FY2023 financials?" in content


def test_judge_empty_claims_short_circuits_without_llm_call() -> None:
    """Zero claims should not waste an LLM call; return an empty result."""
    stub = _StubClient(_text_response({"assessments": [], "overall_reasoning": ""}))
    judge = SufficiencyJudge(client=stub, model="claude-haiku-4-5")

    result = judge.assess(question="q", claims=[])

    assert stub.call_count == 0
    assert result.claim_assessments == []
    # aggregate_status of empty == "ok" per grounding/schema.py contract.
    assert result.aggregate_status() == "ok"


def test_judge_strips_markdown_fences_around_json() -> None:
    """Defensive: Haiku occasionally wraps JSON in ```json fences."""
    claims = [_claim("Apple FY2023 revenue was $383.3B.", "AAPL-10K::0001")]
    fenced = (
        "```json\n"
        + json.dumps(
            {
                "assessments": [
                    {"claim_id": "c1", "status": "SUPPORTED", "rationale": "ok"},
                ],
                "overall_reasoning": "fine",
            }
        )
        + "\n```"
    )
    response: dict[str, Any] = {
        "content": [{"type": "text", "text": fenced}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    stub = _StubClient(response)
    judge = SufficiencyJudge(client=stub, model="claude-haiku-4-5")

    result = judge.assess(question="q", claims=claims)

    assert len(result.claim_assessments) == 1
    assert result.claim_assessments[0].support == ClaimSupport.SUPPORTED


def test_judge_missing_keys_in_response_raises() -> None:
    """Well-formed JSON but missing the ``assessments`` key must raise."""
    claims = [_claim("Apple FY2023 revenue was $383.3B.", "AAPL-10K::0001")]
    payload: dict[str, Any] = {"overall_reasoning": "no assessments"}
    stub = _StubClient(_text_response(payload))
    judge = SufficiencyJudge(client=stub, model="claude-haiku-4-5")

    with pytest.raises(JudgeResponseError):
        judge.assess(question="q", claims=claims)
