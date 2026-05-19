"""HITL gate. Plan 1 = CLI ack; Plan 3 = Slack signed approvals. See spec §3, §8.4."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
from typing import Any, Callable

from firm.audit.log import AuditLog
from firm.agents.reporter import _persist_decisions_from_state
from firm.core.clock import Clock
from firm.core.models import ActionEnum, Decision
from firm.db.connection import get_conn
from firm.orchestrator.state import WorkingState


def make_hitl(*, db_path: Path, clock: Clock) -> Callable[[WorkingState], dict[str, Any]]:
    def hitl(state: WorkingState) -> dict[str, Any]:
        risk: Decision = state["risk_decision"]
        if risk.action != ActionEnum.ESCALATE:
            return {"hitl_required": False, "hitl_approved": True}

        # Persist the risk decision before writing to hitl_queue (hitl_queue has FK → decisions).
        _persist_decisions_from_state({"risk_decision": risk}, db_path, clock)

        with closing(get_conn(db_path)) as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO hitl_queue (decision_id, queued_at, status) "
                "VALUES (?, ?, 'pending')",
                (risk.id, clock.now().isoformat()),
            )
            inserted = cur.rowcount == 1
            row = conn.execute(
                "SELECT status FROM hitl_queue WHERE decision_id=?", (risk.id,)
            ).fetchone()
        if inserted:
            AuditLog(db_path, clock).append("hitl.queued", {"decision_id": risk.id})
        approved = row and row["status"] == "approved"
        return {"hitl_required": True, "hitl_approved": bool(approved)}
    return hitl


def mark_approved(*, db_path: Path, decision_id: str, approver: str, clock: Clock) -> None:
    with closing(get_conn(db_path)) as conn:
        cur = conn.execute(
            "UPDATE hitl_queue SET status='approved', approver=?, decided_at=? "
            "WHERE decision_id=? AND status='pending'",
            (approver, clock.now().isoformat(), decision_id),
        )
        mutated = cur.rowcount == 1
    if mutated:
        AuditLog(db_path, clock).append("hitl.approved", {"decision_id": decision_id, "approver": approver})


def mark_rejected(*, db_path: Path, decision_id: str, approver: str, clock: Clock) -> None:
    with closing(get_conn(db_path)) as conn:
        cur = conn.execute(
            "UPDATE hitl_queue SET status='rejected', approver=?, decided_at=? "
            "WHERE decision_id=? AND status='pending'",
            (approver, clock.now().isoformat(), decision_id),
        )
        mutated = cur.rowcount == 1
    if mutated:
        AuditLog(db_path, clock).append("hitl.rejected", {"decision_id": decision_id, "approver": approver})
