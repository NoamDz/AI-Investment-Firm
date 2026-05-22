"""Tests for firm.hitl.notify — SlackNotifier posts ESCALATE decisions to Slack.

T12 spec: ESCALATE Decision in unit fixture → asserts one chat.postMessage mock
call with two buttons.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision, EscalatePayload
from firm.hitl.notify import SlackNotifier
from firm.hitl.signing import verify as verify_sig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHANNEL = "#trading-hitl"
_APPROVER_ID = "U12345"
_SECRET = b"test-internal-secret"
_FIXED_DT = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_FIXED_TS = int(_FIXED_DT.timestamp())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clock() -> ReplayClock:
    return ReplayClock(_FIXED_DT)


def _make_escalate_decision(
    *,
    dec_id: str = "dec-test-1",
    ticker: str = "AAPL",
    shares: str = "100",
    reason: str = "trade > HITL threshold",
    rationale: str = "requires human review",
) -> Decision:
    return Decision(
        id=dec_id,
        decision_id_chain=["pm-1"],
        action=ActionEnum.ESCALATE,
        payload=EscalatePayload(
            proposed=BuyPayload(ticker=ticker, shares=Decimal(shares)),
            reason=reason,
        ),
        rationale=rationale,
        confidence=0.9,
        citations=[],
        falsification_condition="if cap raised",
        escalation_reason=reason,
        failure_mode=None,
        metadata={},
        nonce="nonce-test",
    )


def _make_notifier(web_client: object) -> SlackNotifier:
    return SlackNotifier(
        web_client=web_client,
        channel=_CHANNEL,
        approver_id=_APPROVER_ID,
        clock=_make_clock(),
        internal_secret=_SECRET,
    )


def _get_blocks_from_call(mock_client: MagicMock) -> list:
    """Extract `blocks` kwarg from the single chat.postMessage call."""
    mock_client.chat_postMessage.assert_called_once()
    call_kwargs = mock_client.chat_postMessage.call_args[1]
    return call_kwargs["blocks"]


# ---------------------------------------------------------------------------
# Test 1 — spec-required: one postMessage call with two buttons
# ---------------------------------------------------------------------------


def test_notify_calls_chat_postMessage_once_with_two_buttons() -> None:
    mock_client = MagicMock()
    notifier = _make_notifier(mock_client)
    decision = _make_escalate_decision()

    notifier.notify(decision=decision)

    mock_client.chat_postMessage.assert_called_once()
    call_kwargs = mock_client.chat_postMessage.call_args[1]

    assert call_kwargs["channel"] == _CHANNEL
    blocks = call_kwargs["blocks"]
    assert len(blocks) == 2, f"Expected 2 blocks, got {len(blocks)}: {blocks}"

    actions_block = blocks[1]
    elements = actions_block["elements"]
    assert len(elements) == 2, f"Expected 2 button elements, got {len(elements)}"

    action_ids = [el["action_id"] for el in elements]
    assert "approve" in action_ids
    assert "reject" in action_ids


# ---------------------------------------------------------------------------
# Test 2 — buttons carry valid signed value (round-trip with verifier)
# ---------------------------------------------------------------------------


def test_notify_buttons_carry_valid_signed_value() -> None:
    mock_client = MagicMock()
    notifier = _make_notifier(mock_client)
    decision = _make_escalate_decision(dec_id="dec-sign-test")

    notifier.notify(decision=decision)

    blocks = _get_blocks_from_call(mock_client)
    elements = blocks[1]["elements"]

    for element in elements:
        value_json = element["value"]
        payload = json.loads(value_json)

        # Must have all 5 required keys
        for key in ("decision_id", "approver_id", "ts", "action", "sig"):
            assert key in payload, f"Missing key {key!r} in button value"

        # Round-trip: verify sig with the same secret
        ok = verify_sig(
            payload={
                "decision_id": payload["decision_id"],
                "approver_id": payload["approver_id"],
                "ts": payload["ts"],
            },
            signature=payload["sig"],
            secret=_SECRET,
            now=_FIXED_TS,
        )
        assert ok, f"Signature verification failed for action={payload['action']!r}"


# ---------------------------------------------------------------------------
# Test 3 — button ts equals clock.now() unix timestamp
# ---------------------------------------------------------------------------


def test_notify_uses_clock_for_signing_ts() -> None:
    mock_client = MagicMock()
    clock = ReplayClock(_FIXED_DT)
    notifier = SlackNotifier(
        web_client=mock_client,
        channel=_CHANNEL,
        approver_id=_APPROVER_ID,
        clock=clock,
        internal_secret=_SECRET,
    )
    decision = _make_escalate_decision()

    notifier.notify(decision=decision)

    blocks = _get_blocks_from_call(mock_client)
    elements = blocks[1]["elements"]

    expected_ts = int(clock.now().timestamp())
    for element in elements:
        payload = json.loads(element["value"])
        assert payload["ts"] == expected_ts, (
            f"Expected ts={expected_ts}, got ts={payload['ts']}"
        )


# ---------------------------------------------------------------------------
# Test 4 — summary text includes decision fields
# ---------------------------------------------------------------------------


def test_notify_summary_includes_decision_fields() -> None:
    mock_client = MagicMock()
    notifier = _make_notifier(mock_client)
    decision = _make_escalate_decision(
        dec_id="dec-summary-1",
        ticker="AAPL",
        shares="100",
        reason="trade > HITL threshold",
        rationale="short rationale",
    )

    notifier.notify(decision=decision)

    blocks = _get_blocks_from_call(mock_client)
    section_block = blocks[0]
    text = section_block["text"]["text"]

    assert "dec-summary-1" in text, f"decision_id not in text: {text!r}"
    assert "AAPL" in text, f"ticker not in text: {text!r}"
    assert "100" in text, f"qty not in text: {text!r}"
    assert "trade > HITL threshold" in text, f"reason not in text: {text!r}"


# ---------------------------------------------------------------------------
# Test 5 — rationale truncated at 500 chars
# ---------------------------------------------------------------------------


def test_notify_rationale_truncated_at_500_chars() -> None:
    mock_client = MagicMock()
    notifier = _make_notifier(mock_client)
    long_rationale = "A" * 1500
    decision = _make_escalate_decision(rationale=long_rationale)

    notifier.notify(decision=decision)

    blocks = _get_blocks_from_call(mock_client)
    section_text = blocks[0]["text"]["text"]

    # The entire section text must be bounded well below 1500 chars
    assert len(section_text) < 1000, (
        f"section text too long ({len(section_text)} chars); rationale not truncated"
    )
    # Should not contain the full 1500-char string
    assert "A" * 501 not in section_text, "Rationale not truncated at 500 chars"


# ---------------------------------------------------------------------------
# Test 6 — Slack exception PROPAGATES (orchestrator owns audit-on-failure)
# ---------------------------------------------------------------------------


def test_notify_propagates_slack_exception() -> None:
    """SlackNotifier.notify() must NOT swallow exceptions.

    The orchestrator (firm.agents.hitl.make_hitl) is the single owner of
    audit-on-failure: it catches the exception and writes the
    hitl.slack_notify_failed audit row.  If notify() swallowed the exception
    here, that audit path would be unreachable in production.
    """
    mock_client = MagicMock()
    mock_client.chat_postMessage.side_effect = RuntimeError("Slack is down")
    notifier = _make_notifier(mock_client)
    decision = _make_escalate_decision()

    with pytest.raises(RuntimeError, match="Slack is down"):
        notifier.notify(decision=decision)
