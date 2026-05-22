"""HITL gate. Plan 1 = CLI ack; Plan 3 = Slack signed approvals. See spec §3, §8.4."""
from __future__ import annotations

import logging
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from firm.audit.log import AuditLog
from firm.agents.reporter import _persist_decisions_from_state
from firm.core.clock import Clock
from firm.core.ids import sign_nonce, ulid_new
from firm.core.models import (
    ActionEnum,
    Decision,
    FailureMode,
    RefusePayload,
)
from firm.db.connection import get_conn
from firm.orchestrator.state import WorkingState

if TYPE_CHECKING:
    from firm.hitl.notify import SlackNotifier

logger = logging.getLogger(__name__)


# Spec §8.5 default: HITL approvals must arrive within 30 minutes or the
# pending row is treated as unapproved-high-risk.  Lifted to a module
# constant so callers (and tests) can reference the same canonical value.
DEFAULT_HITL_TIMEOUT_SECONDS: int = 1800


def make_hitl(
    *,
    db_path: Path,
    clock: Clock,
    notifier: "SlackNotifier | None" = None,
) -> Callable[[WorkingState], dict[str, Any]]:
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
            # Notify via Slack on first ESCALATE entry only (idempotency: not on
            # re-traversal).  DB insert happens before Slack call (audit ordering).
            if notifier is not None:
                try:
                    notifier.notify(decision=risk)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "hitl.slack_notify_failed: decision_id=%s error=%s",
                        risk.id,
                        exc,
                    )
                    AuditLog(db_path, clock).append(
                        "hitl.slack_notify_failed",
                        {"decision_id": risk.id, "error": str(exc)},
                    )
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


# ---------------------------------------------------------------------------
# Plan 4 T24 — HITL aging/reaper policy (UNAPPROVED_HIGH_RISK)
# ---------------------------------------------------------------------------


def reap_expired_hitl_entries(
    *,
    db_path: Path,
    clock: Clock,
    deadline_seconds: int = DEFAULT_HITL_TIMEOUT_SECONDS,
    nonce_secret: bytes,
) -> list[Decision]:
    """Sweep ``hitl_queue`` for aged-out pending rows and emit REFUSE Decisions.

    For every row where ``status='pending'`` and ``clock.now() - queued_at``
    exceeds ``deadline_seconds``, construct a REFUSE :class:`Decision`
    with ``failure_mode=UNAPPROVED_HIGH_RISK`` and a ``decision_id_chain``
    pointing back to the expired ESCALATE.  Persist each new disposition
    Decision via :func:`_persist_decisions_from_state` and return the list.

    Conservative-default policy: when a high-risk trade ages out without
    an explicit human approval, the disposition is NOT a bare
    ``HITL_TIMEOUT`` (which would be a transport/UX failure mode); it is
    ``UNAPPROVED_HIGH_RISK`` — the trade is refused on the substantive
    grounds that no authorised human signed off, which is materially
    different from "the message did not arrive".

    Intentional non-decision: the reaper does NOT mutate the ``hitl_queue``
    row (it remains ``'pending'``).  Whether the queue-row lifecycle
    should transition (e.g., ``'pending'`` → ``'timed_out'``) is a
    separate question deferred to a future iteration.

    Parameters
    ----------
    db_path:
        Path to the firm SQLite DB.
    clock:
        Clock providing ``now()`` for age comparison and Decision timestamping.
    deadline_seconds:
        Age threshold beyond which a pending row is considered aged-out.
        Defaults to :data:`DEFAULT_HITL_TIMEOUT_SECONDS` (1800s = 30 min,
        matching spec §8.5).
    nonce_secret:
        HMAC secret used to sign the emitted REFUSE Decisions' nonces.

    Returns
    -------
    list[Decision]
        The REFUSE Decisions emitted this sweep (empty if no rows aged out).
    """
    now = clock.now()
    refused: list[Decision] = []
    with closing(get_conn(db_path)) as conn:
        rows = conn.execute(
            "SELECT decision_id, queued_at FROM hitl_queue WHERE status='pending'"
        ).fetchall()
    for row in rows:
        queued_at = datetime.fromisoformat(row["queued_at"])
        age_seconds = (now - queued_at).total_seconds()
        if age_seconds < deadline_seconds:
            continue
        refuse_id = ulid_new()
        nonce = sign_nonce(
            nonce_secret,
            decision_id=refuse_id,
            timestamp=int(now.timestamp()),
        )
        refuse = Decision(
            id=refuse_id,
            decision_id_chain=[row["decision_id"]],
            action=ActionEnum.REFUSE,
            payload=RefusePayload(reason="hitl:unapproved_high_risk"),
            rationale=(
                f"hitl_queue entry {row['decision_id']!r} aged "
                f"{int(age_seconds)}s past the {deadline_seconds}s deadline "
                f"without an approval; conservative default applied"
            ),
            confidence=0.0,
            citations=[],
            falsification_condition=(
                "an authorised human posts an approval within the deadline"
            ),
            escalation_reason=None,
            failure_mode=FailureMode.UNAPPROVED_HIGH_RISK,
            metadata={"agent": "hitl", "expired_decision_id": row["decision_id"]},
            nonce=nonce,
        )
        _persist_decisions_from_state({"risk_decision": refuse}, db_path, clock)
        refused.append(refuse)
    return refused
