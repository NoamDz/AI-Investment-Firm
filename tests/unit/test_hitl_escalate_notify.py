"""Tests for make_hitl notifier wiring — T12 ESCALATE node integration.

Covers:
7. notifier called once on new ESCALATE entry
8. notifier NOT called on re-traversal (INSERT OR IGNORE no-op)
9. notifier=None preserves existing behavior (no exception)
10. notifier exception is swallowed and audit-logged
"""
from __future__ import annotations

import json
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

from firm.agents.hitl import make_hitl
from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision, EscalatePayload
from firm.db.connection import get_conn
from firm.db.migrations import init_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clock() -> ReplayClock:
    return ReplayClock(datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc))


def _make_escalate_decision(dec_id: str = "dec-hitl-1") -> Decision:
    return Decision(
        id=dec_id,
        decision_id_chain=["pm-1"],
        action=ActionEnum.ESCALATE,
        payload=EscalatePayload(
            proposed=BuyPayload(ticker="AAPL", shares=Decimal("50")),
            reason="trade > HITL threshold",
        ),
        rationale="requires human review",
        confidence=0.9,
        citations=[],
        falsification_condition="if cap raised",
        escalation_reason="trade > HITL threshold",
        failure_mode=None,
        metadata={},
        nonce="nonce-1",
    )


def _make_mock_notifier() -> MagicMock:
    notifier = MagicMock()
    notifier.notify = MagicMock()
    return notifier


# ---------------------------------------------------------------------------
# Test 7 — notifier called once on new ESCALATE entry
# ---------------------------------------------------------------------------


def test_escalate_node_calls_notifier_once_on_new_queue(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    clock = _clock()
    decision = _make_escalate_decision()
    mock_notifier = _make_mock_notifier()

    hitl = make_hitl(db_path=db, clock=clock, notifier=mock_notifier)
    result = hitl({"risk_decision": decision})

    # Notifier called exactly once with the decision
    mock_notifier.notify.assert_called_once_with(decision=decision)

    # hitl_queue row must be present
    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", (decision.id,)
        ).fetchone()
    assert row is not None
    assert row["status"] == "pending"

    assert result["hitl_required"] is True


# ---------------------------------------------------------------------------
# Test 8 — notifier NOT called on re-traversal (idempotency)
# ---------------------------------------------------------------------------


def test_escalate_node_skips_notifier_on_re_traversal(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    clock = _clock()
    decision = _make_escalate_decision()
    mock_notifier = _make_mock_notifier()

    hitl = make_hitl(db_path=db, clock=clock, notifier=mock_notifier)

    # First traversal — inserts row, notifier called
    hitl({"risk_decision": decision})
    assert mock_notifier.notify.call_count == 1

    # Second traversal — INSERT OR IGNORE no-ops, notifier NOT called again
    hitl({"risk_decision": decision})
    assert mock_notifier.notify.call_count == 1, (
        "Notifier should not be called on re-traversal"
    )


# ---------------------------------------------------------------------------
# Test 9 — notifier=None preserves existing behavior (no exception)
# ---------------------------------------------------------------------------


def test_escalate_node_works_with_none_notifier(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    clock = _clock()
    decision = _make_escalate_decision(dec_id="dec-none-notifier")

    hitl = make_hitl(db_path=db, clock=clock, notifier=None)

    # Must not raise; row must be inserted
    result = hitl({"risk_decision": decision})

    assert result["hitl_required"] is True
    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", (decision.id,)
        ).fetchone()
    assert row is not None
    assert row["status"] == "pending"


# ---------------------------------------------------------------------------
# Test 10 — notifier failure is swallowed and audit-logged
# ---------------------------------------------------------------------------


def test_escalate_node_swallows_notifier_failure_and_audits(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    clock = _clock()
    decision = _make_escalate_decision(dec_id="dec-notify-fail")

    mock_notifier = _make_mock_notifier()
    mock_notifier.notify.side_effect = RuntimeError("Slack is down")

    hitl = make_hitl(db_path=db, clock=clock, notifier=mock_notifier)

    # (a) hitl returns normally — exception must not propagate
    result = hitl({"risk_decision": decision})
    assert result["hitl_required"] is True
    assert result["hitl_approved"] is False

    # (b) audit_log has a hitl.slack_notify_failed row with the decision_id
    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT detail FROM audit_log WHERE event='hitl.slack_notify_failed' AND detail LIKE ?",
            (f"%{decision.id}%",),
        ).fetchone()

    assert row is not None, "Expected hitl.slack_notify_failed audit row"
    detail = json.loads(row["detail"])
    assert detail.get("decision_id") == decision.id
    assert "error" in detail
