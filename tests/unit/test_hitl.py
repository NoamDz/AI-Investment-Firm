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
    conn = get_conn(db)
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
