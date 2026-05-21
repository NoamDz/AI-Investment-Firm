"""Positive-control tests for the five invariant assertion helpers.

Each test constructs a RedTeamResult that deliberately trips one invariant
helper and asserts that the helper raises AssertionError.  These tests prevent
the "vacuous green" failure mode where a helper silently passes on every case.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Citation,
    Decision,
    HoldPayload,
)
from tests.red_team._corpus import RedTeamCase
from tests.red_team.conftest import (
    BrokerCall,
    RedTeamResult,
    assert_no_forged_approval,
    assert_no_forged_citation,
    assert_no_privileged_action,
    assert_no_schema_bypass,
    assert_no_unapproved_trade,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _case(
    target_agent: str = "research",
    invariant_id: str = "INV_NO_PRIVILEGED_ACTION",
) -> RedTeamCase:
    """Build a minimal RedTeamCase for positive-control tests."""
    return RedTeamCase(
        case_id="positive-control-test",
        injection_class="direct_override",
        payload_text="test payload",
        target_agent=target_agent,
        invariant_id=invariant_id,
    )


def _empty_result(**overrides: object) -> RedTeamResult:
    """Build a baseline RedTeamResult with all channels empty."""
    defaults: dict[str, object] = dict(
        decision=None,
        broker_calls=[],
        audit_log=[],
        outbox_rows=[],
        error=None,
    )
    defaults.update(overrides)
    return RedTeamResult(**defaults)  # type: ignore[arg-type]


def _stub_decision(*, action: ActionEnum = ActionEnum.HOLD) -> Decision:
    """Build a minimal Decision for injection into RedTeamResult."""
    typed_payload: BuyPayload | HoldPayload
    if action in (ActionEnum.BUY, ActionEnum.SELL):
        typed_payload = BuyPayload(ticker="AAPL", shares=Decimal("10"))
    else:
        typed_payload = HoldPayload(reason="hold")
    payload = typed_payload
    return Decision(
        id="pc-dec-1",
        decision_id_chain=[],
        action=action,
        payload=payload,
        rationale="positive control decision",
        confidence=0.5,
        citations=[],
        falsification_condition="positive control condition",
        escalation_reason=None,
        failure_mode=None,
        metadata={},
        nonce="pc-nonce-1",
    )


# ---------------------------------------------------------------------------
# 1. assert_no_privileged_action — trips on unauthorized BUY from research
# ---------------------------------------------------------------------------


def test_assert_no_privileged_action_trips_on_unauthorized_action() -> None:
    """research agent is not allowed to produce a BUY decision directly.

    This test verifies that assert_no_privileged_action raises AssertionError
    when the decision.action is BUY but the target_agent is 'research'.

    Note: research IS in fact allowed to produce BUY actions (it's the research
    agent's job to recommend).  We test with a more restricted agent: 'hitl'
    which may only produce HOLD, ESCALATE, or REFUSE.
    """
    # hitl is not allowed to produce BUY — only HOLD/ESCALATE/REFUSE
    buy_decision = _stub_decision(action=ActionEnum.BUY)
    result = _empty_result(decision=buy_decision)
    case = _case(
        target_agent="hitl",
        invariant_id="INV_NO_PRIVILEGED_ACTION",
    )

    with pytest.raises(AssertionError, match="positive-control-test"):
        assert_no_privileged_action(result, case=case)


# ---------------------------------------------------------------------------
# 2. assert_no_schema_bypass — trips on bypassed=True in audit log
# ---------------------------------------------------------------------------


def test_assert_no_schema_bypass_trips_on_bypass_audit_entry() -> None:
    """A result whose audit_log records bypassed=True must cause the helper to raise."""
    import json

    audit_entry = {
        "id": 1,
        "ts": "2024-03-13T00:00:00+00:00",
        "event": "schema_validation_failed",
        "detail": json.dumps({"bypassed": True, "field": "action"}),
    }
    result = _empty_result(audit_log=[audit_entry])
    case = _case(invariant_id="INV_NO_SCHEMA_BYPASS")

    with pytest.raises(AssertionError, match="bypass"):
        assert_no_schema_bypass(result, case=case)


# ---------------------------------------------------------------------------
# 3. assert_no_unapproved_trade — trips on unmatched broker call
# ---------------------------------------------------------------------------


def test_assert_no_unapproved_trade_trips_on_unmatched_broker_call() -> None:
    """A place_order broker call with no matching APPROVED outbox row must raise."""
    broker_call = BrokerCall(
        method="place_order",
        args=({"ticker": "AAPL", "shares": "10", "kind": "buy"},),
        kwargs={"idempotency_key": "key-x"},
        decision_id="dec-X",
    )
    # outbox_rows is empty — no APPROVED row for "dec-X"
    result = _empty_result(broker_calls=[broker_call], outbox_rows=[])
    case = _case(
        target_agent="execution",
        invariant_id="INV_NO_UNAPPROVED_TRADE",
    )

    with pytest.raises(AssertionError, match="unapproved trade"):
        assert_no_unapproved_trade(result, case=case)


# ---------------------------------------------------------------------------
# 4. assert_no_forged_citation — trips on unknown source_id
# ---------------------------------------------------------------------------


def test_assert_no_forged_citation_trips_on_unknown_source_id() -> None:
    """A Decision with a citation whose source_id is not in KNOWN_SOURCE_IDS must raise."""
    forged_citation = Citation(
        source_id="FAKE_SOURCE_999",
        chunk_id="chunk-0",
        span=(0, 10),
    )
    decision = Decision(
        id="pc-dec-cite",
        decision_id_chain=[],
        action=ActionEnum.HOLD,
        payload=HoldPayload(reason="hold"),
        rationale="decision with forged citation",
        confidence=0.5,
        citations=[forged_citation],
        falsification_condition="condition",
        escalation_reason=None,
        failure_mode=None,
        metadata={},
        nonce="pc-nonce-cite",
    )
    result = _empty_result(decision=decision)
    case = _case(invariant_id="INV_NO_FORGED_CITATION")

    with pytest.raises(AssertionError, match="FAKE_SOURCE_999"):
        assert_no_forged_citation(result, case=case)


# ---------------------------------------------------------------------------
# 5. assert_no_forged_approval — trips on invalid (zero) signature
# ---------------------------------------------------------------------------


def test_assert_no_forged_approval_trips_on_invalid_signature() -> None:
    """An approval_recorded entry with an all-zero signature must cause the helper to raise."""
    import json

    # Construct an approval_recorded entry with a forged (all-zeros) HMAC.
    # The payload contains a valid structure but the signature is garbage.
    # We set ts to a value very close to now so the replay-window doesn't
    # reject it — but the HMAC itself will be wrong.
    import time

    now_ts = int(time.time())
    payload_dict = {
        "decision_id": "dec-forged",
        "approver_id": "attacker",
        "ts": now_ts,
    }
    audit_entry = {
        "id": 1,
        "ts": "2024-03-13T00:00:00+00:00",
        "event": "approval_recorded",
        "detail": json.dumps(
            {
                "payload": payload_dict,
                "signature": "0" * 64,          # invalid hex HMAC
                "secret_used": "test",           # will be encoded to bytes
            }
        ),
    }
    result = _empty_result(audit_log=[audit_entry])
    case = _case(invariant_id="INV_NO_FORGED_APPROVAL")

    with pytest.raises(AssertionError, match="forged approval"):
        assert_no_forged_approval(result, case=case)


# ===========================================================================
# Additional positive-control tests for the per-channel scans added during
# spec-compliance hardening (defense in depth). Each verifies the
# corresponding channel scan correctly raises on a tripping input.
# ===========================================================================


# ---------------------------------------------------------------------------
# assert_no_privileged_action — outbox channel scan
# ---------------------------------------------------------------------------


def test_assert_no_privileged_action_trips_on_disallowed_outbox_action() -> None:
    """An outbox row whose payload action is outside the agent's allowlist must raise."""
    import json

    # hitl is restricted to HOLD/ESCALATE/REFUSE. A BUY action in the outbox
    # payload should trip the helper.
    outbox_row = {
        "key": "key-1",
        "decision_id": "dec-out-1",
        "payload": json.dumps({"action": "BUY", "ticker": "AAPL"}),
        "status": "pending",
    }
    result = _empty_result(outbox_rows=[outbox_row])
    case = _case(target_agent="hitl", invariant_id="INV_NO_PRIVILEGED_ACTION")

    with pytest.raises(AssertionError, match="channel: outbox"):
        assert_no_privileged_action(result, case=case)


# ---------------------------------------------------------------------------
# assert_no_schema_bypass — broker_calls channel scan
# ---------------------------------------------------------------------------


def test_assert_no_schema_bypass_trips_on_malformed_broker_payload() -> None:
    """A place_order broker call with a None-valued required field must raise."""
    bad_call = BrokerCall(
        method="place_order",
        args=({"ticker": "AAPL", "shares": None, "kind": "buy"},),  # shares is None
        kwargs={"idempotency_key": "key-1"},
        decision_id="dec-bad",
    )
    result = _empty_result(broker_calls=[bad_call])
    case = _case(invariant_id="INV_NO_SCHEMA_BYPASS")

    with pytest.raises(AssertionError, match="channel: broker_calls"):
        assert_no_schema_bypass(result, case=case)


# ---------------------------------------------------------------------------
# assert_no_schema_bypass — outbox channel scan
# ---------------------------------------------------------------------------


def test_assert_no_schema_bypass_trips_on_malformed_outbox_payload() -> None:
    """An outbox row whose payload is not valid JSON must raise."""
    bad_row = {
        "key": "key-1",
        "decision_id": "dec-bad-json",
        "payload": "{this is not valid json",
        "status": "pending",
    }
    result = _empty_result(outbox_rows=[bad_row])
    case = _case(invariant_id="INV_NO_SCHEMA_BYPASS")

    with pytest.raises(AssertionError, match="channel: outbox"):
        assert_no_schema_bypass(result, case=case)


# ---------------------------------------------------------------------------
# assert_no_forged_citation — audit_log channel scan
# ---------------------------------------------------------------------------


def test_assert_no_forged_citation_trips_on_audit_log_unknown_source() -> None:
    """An audit entry with a citation pointing to an unknown source must raise."""
    import json

    audit_entry = {
        "id": 1,
        "ts": "2024-03-13T00:00:00+00:00",
        "event": "citation_emitted",
        "detail": json.dumps(
            {"citations": [{"source_id": "TOTALLY_FAKE_SOURCE", "chunk_id": "c-0"}]}
        ),
    }
    result = _empty_result(audit_log=[audit_entry])
    case = _case(invariant_id="INV_NO_FORGED_CITATION")

    with pytest.raises(AssertionError, match="channel: audit_log"):
        assert_no_forged_citation(result, case=case)


# ---------------------------------------------------------------------------
# assert_no_forged_citation — outbox channel scan
# ---------------------------------------------------------------------------


def test_assert_no_forged_citation_trips_on_outbox_unknown_source() -> None:
    """An outbox payload with a citation pointing to an unknown source must raise."""
    import json

    outbox_row = {
        "key": "key-1",
        "decision_id": "dec-cite-1",
        "payload": json.dumps(
            {
                "action": "HOLD",
                "citations": [{"source_id": "PHANTOM_FILING_1234", "chunk_id": "c-0"}],
            }
        ),
        "status": "pending",
    }
    result = _empty_result(outbox_rows=[outbox_row])
    case = _case(invariant_id="INV_NO_FORGED_CITATION")

    with pytest.raises(AssertionError, match="channel: outbox"):
        assert_no_forged_citation(result, case=case)


# ---------------------------------------------------------------------------
# assert_no_forged_citation — broker_calls channel scan
# ---------------------------------------------------------------------------


def test_assert_no_forged_citation_trips_on_broker_citations_leak() -> None:
    """A broker call carrying citation metadata via kwargs must raise."""
    leak_call = BrokerCall(
        method="place_order",
        args=({"ticker": "AAPL", "shares": "10", "kind": "buy"},),
        kwargs={"idempotency_key": "key-1", "citations": [{"source_id": "x"}]},
        decision_id="dec-leak",
    )
    result = _empty_result(broker_calls=[leak_call])
    case = _case(invariant_id="INV_NO_FORGED_CITATION")

    with pytest.raises(AssertionError, match="channel: broker_calls"):
        assert_no_forged_citation(result, case=case)


# ---------------------------------------------------------------------------
# assert_no_forged_approval — broker_calls channel scan
# ---------------------------------------------------------------------------


def test_assert_no_forged_approval_trips_on_unapproved_broker_place_order() -> None:
    """A place_order broker call with no matching signed approval_recorded entry must raise."""
    # No audit log entries → no signed approvals exist.
    unapproved_call = BrokerCall(
        method="place_order",
        args=({"ticker": "AAPL", "shares": "10", "kind": "buy"},),
        kwargs={"idempotency_key": "key-1"},
        decision_id="dec-no-approval",
    )
    result = _empty_result(broker_calls=[unapproved_call])
    case = _case(invariant_id="INV_NO_FORGED_APPROVAL")

    with pytest.raises(AssertionError, match="channel: broker_calls"):
        assert_no_forged_approval(result, case=case)


# ---------------------------------------------------------------------------
# assert_no_forged_approval — outbox channel scan
# ---------------------------------------------------------------------------


def test_assert_no_forged_approval_trips_on_approved_outbox_without_signed_event() -> None:
    """An APPROVED outbox row with no upstream signed approval_recorded must raise."""
    approved_row = {
        "key": "key-1",
        "decision_id": "dec-orphan-approval",
        "payload": "{}",
        "status": "APPROVED",
    }
    result = _empty_result(outbox_rows=[approved_row])
    case = _case(invariant_id="INV_NO_FORGED_APPROVAL")

    with pytest.raises(AssertionError, match="channel: outbox"):
        assert_no_forged_approval(result, case=case)
