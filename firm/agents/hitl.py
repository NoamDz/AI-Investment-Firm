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
    """Approve a queued HITL decision.

    Supports two modes:

    * **Post-queue**: when ``hitl`` has already run and inserted a ``'pending'``
      row, the ``UPDATE … WHERE status='pending'`` flips it to ``'approved'``.

    * **Pre-queue** (T31 pattern): when the graph has been interrupted
      *before* the ``hitl`` node executes (``interrupt_before=["hitl"]``),
      the ``hitl_queue`` row does not exist yet.  A preceding
      ``INSERT OR IGNORE`` creates a pre-approved row so that when the
      graph resumes and ``hitl`` runs its own ``INSERT OR IGNORE``, that
      insert is a no-op and the ``SELECT status`` immediately returns
      ``'approved'``.

    The caller is responsible for ensuring the ``decisions`` row for
    ``decision_id`` already exists (FK constraint on ``hitl_queue``).
    """
    now_iso = clock.now().isoformat()
    with closing(get_conn(db_path)) as conn:
        # Pre-approve path: insert an 'approved' row if none exists yet.
        ins = conn.execute(
            "INSERT OR IGNORE INTO hitl_queue "
            "(decision_id, queued_at, status, approver, decided_at) "
            "VALUES (?, ?, 'approved', ?, ?)",
            (decision_id, now_iso, approver, now_iso),
        )
        pre_approved = ins.rowcount == 1
        # Post-queue path: flip an existing 'pending' row to 'approved'.
        cur = conn.execute(
            "UPDATE hitl_queue SET status='approved', approver=?, decided_at=? "
            "WHERE decision_id=? AND status='pending'",
            (approver, now_iso, decision_id),
        )
        mutated = cur.rowcount == 1
    # Emit the audit event on either path: a fresh pre-approve INSERT or a
    # pending → approved UPDATE.  Without this, pre-approved decisions slipped
    # through with zero entries in audit_log — a real audit trail gap.
    if pre_approved or mutated:
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
