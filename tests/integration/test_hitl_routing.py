"""Integration test for HITL routing via ESCALATE action.

Covers:
- No FK violation when hitl node runs (C1 fix).
- Full ESCALATE → interrupt → resume → approve → re-run → confirmed outbox.

The LangGraph interrupt_before=["hitl"] pattern produces a 4-invoke sequence:
  Invoke 1 ({}):   monitor→research→pm→risk; interrupt before hitl.
  Invoke 2 (None): hitl runs, inserts pending queue row; execution skips.
  mark_approved(): updates hitl_queue to 'approved'.
  Invoke 3 ({}):   monitor→research→pm→risk again; interrupt before hitl.
  Invoke 4 (None): hitl reads 'approved'; execution proceeds; confirmed outbox row.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from firm.agents.execution import make_execution
from firm.agents.hitl import make_hitl, mark_approved
from firm.agents.reporter import make_reporter
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision, EscalatePayload
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.orchestrator.graph import build_graph
from firm.orchestrator.state import WorkingState


def _make_escalate_decision() -> Decision:
    return Decision(
        id="dec-escalate-1",
        decision_id_chain=["dec-research-1", "dec-pm-1"],
        action=ActionEnum.ESCALATE,
        payload=EscalatePayload(
            proposed=BuyPayload(ticker="AAPL", shares=Decimal("5")),
            reason="position size exceeds single-stock cap",
        ),
        rationale="Position would breach 10% single-stock cap; escalate for human review.",
        confidence=0.9,
        citations=[],
        falsification_condition="If cap raised above 15% this can proceed.",
        escalation_reason="position size exceeds single-stock cap",
        failure_mode=None,
        metadata={},
        nonce="nonce-escalate-1",
    )


def _make_buy_decision() -> Decision:
    return Decision(
        id="dec-research-1",
        decision_id_chain=[],
        action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("5")),
        rationale="Strong fundamentals.",
        confidence=0.8,
        citations=[],
        falsification_condition="If revenue declines.",
        escalation_reason=None,
        failure_mode=None,
        metadata={},
        nonce="nonce-research-1",
    )


def _build_graph(db: Path, clock: ReplayClock, broker: FakeBroker, tmp_path: Path):
    escalate_decision = _make_escalate_decision()

    def monitor_node(state: WorkingState) -> dict[str, Any]:
        return {"heartbeat_at": clock.now().isoformat()}

    def research_node(state: WorkingState) -> dict[str, Any]:
        return {"research_decision": _make_buy_decision()}

    def pm_node(state: WorkingState) -> dict[str, Any]:
        return {"pm_decision": state["research_decision"]}

    def risk_node(state: WorkingState) -> dict[str, Any]:
        return {"risk_decision": escalate_decision}

    hitl = make_hitl(db_path=db, clock=clock)
    execution = make_execution(db_path=db, broker=broker, clock=clock)
    reporter = make_reporter(reports_root=tmp_path / "reports", clock=clock, db_path=db)

    return (
        build_graph(
            db_path=db,
            monitor_node=monitor_node,
            research_node=research_node,
            pm_node=pm_node,
            risk_node=risk_node,
            hitl_node=hitl,
            execution_node=execution,
            reporter_node=reporter,
        ),
        escalate_decision,
    )


def test_hitl_no_fk_violation_on_escalate(tmp_path: Path) -> None:
    """Regression: hitl node must not raise FK violation for the risk decision (C1).

    Before the fix, hitl.py would INSERT into hitl_queue (which has a FK to decisions)
    before the risk Decision had been persisted — causing sqlite3.IntegrityError.
    """
    db = tmp_path / "firm.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))

    with get_conn(db) as conn:
        conn.execute(
            "INSERT INTO cash (id, amount, updated_at) VALUES (1, ?, ?)",
            (str(broker.get_cash()), clock.now().isoformat()),
        )

    graph, escalate_decision = _build_graph(db, clock, broker, tmp_path)
    config = {"configurable": {"thread_id": "test-hitl-nofk"}}

    # Invoke 1: graph interrupts before hitl, no FK error yet
    graph.invoke({}, config=config)

    # Invoke 2 (resume): hitl runs — must NOT raise FK violation (C1 fix).
    # hitl persists the risk decision before inserting into hitl_queue.
    result = graph.invoke(None, config=config)

    conn = get_conn(db)

    # hitl_queue should have a pending row (hitl ran without FK error)
    hq_row = conn.execute(
        "SELECT status FROM hitl_queue WHERE decision_id=?", (escalate_decision.id,)
    ).fetchone()
    assert hq_row is not None, "hitl_queue must have a row after hitl node runs"
    assert hq_row["status"] == "pending"

    # decisions table must have the risk decision persisted (FK parent satisfied)
    dec_row = conn.execute(
        "SELECT id FROM decisions WHERE id=?", (escalate_decision.id,)
    ).fetchone()
    assert dec_row is not None, "risk decision must be persisted before hitl_queue insert"

    # execution skipped because pending approval
    assert result.get("execution_result", {}).get("skipped") is True


def test_hitl_escalate_interrupt_approve_resume_confirms_outbox(tmp_path: Path) -> None:
    """Full 4-invoke ESCALATE approval flow results in a confirmed outbox row."""
    db = tmp_path / "firm.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))

    with get_conn(db) as conn:
        conn.execute(
            "INSERT INTO cash (id, amount, updated_at) VALUES (1, ?, ?)",
            (str(broker.get_cash()), clock.now().isoformat()),
        )

    graph, escalate_decision = _build_graph(db, clock, broker, tmp_path)
    config = {"configurable": {"thread_id": "test-hitl-full"}}

    # Invoke 1: interrupts before hitl
    graph.invoke({}, config=config)

    # Invoke 2: hitl runs, inserts pending row, execution skips
    result2 = graph.invoke(None, config=config)
    assert result2.get("execution_result", {}).get("skipped") is True

    conn = get_conn(db)
    hq = conn.execute(
        "SELECT status FROM hitl_queue WHERE decision_id=?", (escalate_decision.id,)
    ).fetchone()
    assert hq is not None and hq["status"] == "pending"

    # Approve
    mark_approved(
        db_path=db,
        decision_id=escalate_decision.id,
        approver="test-approver",
        clock=clock,
    )
    hq2 = conn.execute(
        "SELECT status FROM hitl_queue WHERE decision_id=?", (escalate_decision.id,)
    ).fetchone()
    assert hq2 is not None and hq2["status"] == "approved"

    # Invoke 3: new run on same thread — interrupts before hitl again
    graph.invoke({}, config=config)

    # Invoke 4: hitl reads 'approved', execution proceeds
    result4 = graph.invoke(None, config=config)
    assert result4.get("execution_result", {}).get("skipped") is None or result4.get("execution_result", {}).get("skipped") is not True

    # outbox should have a confirmed row
    rows = conn.execute("SELECT status FROM outbox WHERE status='confirmed'").fetchall()
    assert len(rows) >= 1, f"Expected at least 1 confirmed outbox row, got {len(rows)}"
