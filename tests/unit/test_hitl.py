from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.agents.hitl import make_hitl, mark_approved
from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision, EscalatePayload
from firm.db.connection import get_conn
from firm.db.migrations import init_db


def _persist_decision(db: Path, d: Decision, clock):
    import json
    with closing(get_conn(db)) as conn:
        conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                d.id, json.dumps(d.decision_id_chain), d.action.value,
                d.payload.model_dump_json(), d.rationale, d.confidence,
                json.dumps([c.model_dump(mode="json") for c in d.citations]),
                d.falsification_condition, d.escalation_reason,
                d.failure_mode.value if d.failure_mode else None,
                json.dumps(d.metadata), d.nonce, clock.now().isoformat(),
            ),
        )


def _risk_escalation() -> Decision:
    return Decision(
        id="risk-1", decision_id_chain=["pm-1"], action=ActionEnum.ESCALATE,
        payload=EscalatePayload(
            proposed=BuyPayload(ticker="AAPL", shares=Decimal("100")),
            reason="trade > HITL threshold",
        ),
        rationale="hitl required", confidence=1.0, citations=[],
        falsification_condition="timeout", escalation_reason="trade > HITL threshold",
        failure_mode=None, metadata={}, nonce="n",
    )


def test_hitl_queues_decision(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    d = _risk_escalation()
    _persist_decision(db, d, clock)

    hitl = make_hitl(db_path=db, clock=clock)
    state = hitl({"risk_decision": d})
    assert state.get("hitl_required") is True

    row = get_conn(db).execute("SELECT status FROM hitl_queue WHERE decision_id=?", (d.id,)).fetchone()
    assert row["status"] == "pending"


def test_mark_approved_updates_queue(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    d = _risk_escalation()
    _persist_decision(db, d, clock)
    make_hitl(db_path=db, clock=clock)({"risk_decision": d})

    mark_approved(db_path=db, decision_id=d.id, approver="cli-user", clock=clock)
    row = get_conn(db).execute("SELECT status, approver FROM hitl_queue WHERE decision_id=?", (d.id,)).fetchone()
    assert row["status"] == "approved"
    assert row["approver"] == "cli-user"


def test_hitl_fast_path_when_not_escalate(tmp_path: Path):
    """Non-ESCALATE decisions pass through with hitl_required=False, hitl_approved=True."""
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    pm_decision = Decision(
        id="pm-1", decision_id_chain=[], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="r", confidence=0.5, citations=[],
        falsification_condition="f", escalation_reason=None,
        failure_mode=None, metadata={}, nonce="n",
    )
    out = make_hitl(db_path=db, clock=clock)({"risk_decision": pm_decision})
    assert out == {"hitl_required": False, "hitl_approved": True}


def test_hitl_requeue_is_idempotent(tmp_path: Path):
    """Re-entering the node for the same decision produces exactly one queue row and one audit event."""
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    d = _risk_escalation()
    _persist_decision(db, d, clock)
    hitl = make_hitl(db_path=db, clock=clock)
    hitl({"risk_decision": d})
    hitl({"risk_decision": d})
    with closing(get_conn(db)) as conn:
        count = conn.execute("SELECT COUNT(*) AS n FROM hitl_queue WHERE decision_id=?", (d.id,)).fetchone()["n"]
        audit_count = conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE event='hitl.queued' AND detail LIKE ?",
            (f"%{d.id}%",),
        ).fetchone()["n"]
    assert count == 1
    assert audit_count == 1


def test_hitl_resume_after_approval_sees_approved_status(tmp_path: Path):
    """The core LangGraph re-entry contract: queue, ack, re-enter, hitl_approved=True."""
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    d = _risk_escalation()
    _persist_decision(db, d, clock)
    hitl = make_hitl(db_path=db, clock=clock)

    first = hitl({"risk_decision": d})
    assert first == {"hitl_required": True, "hitl_approved": False}

    mark_approved(db_path=db, decision_id=d.id, approver="cli-user", clock=clock)

    resumed = hitl({"risk_decision": d})
    assert resumed == {"hitl_required": True, "hitl_approved": True}


def test_mark_approved_pre_queue_emits_audit_event(tmp_path: Path):
    """Pre-approve path (no prior 'pending' row) must still write hitl.approved."""
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    d = _risk_escalation()
    _persist_decision(db, d, clock)

    # Call mark_approved WITHOUT first invoking hitl — exercises the
    # INSERT OR IGNORE 'approved' pre-queue path used by T31.
    mark_approved(db_path=db, decision_id=d.id, approver="t31-test", clock=clock)

    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", (d.id,)
        ).fetchone()
        audit = conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE event='hitl.approved' AND detail LIKE ?",
            (f"%{d.id}%",),
        ).fetchone()["n"]
    assert row["status"] == "approved"
    assert audit == 1, "hitl.approved audit event missing on pre-approve path"
