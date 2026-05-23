"""Plan 4 T26 — HITL_TIMEOUT end-to-end fixture.

Scenario
--------
A heartbeat produces a trade that the Risk agent decides to ESCALATE
to a human.  The HITL gate persists the ESCALATE Decision, inserts a
``hitl_queue`` row with status ``'pending'`` keyed by the ESCALATE
Decision id, and then attempts to notify the operator via Slack.  In
this fixture the Slack notifier raises mid-call (the wire is broken /
the Slack API is returning 5xx); the production catch at
``firm/agents/hitl.py:50-62`` swallows the exception, writes a
``hitl.slack_notify_failed`` event to ``audit_log``, and continues so
the HITL queue is never blocked by a Slack outage.

But no human ever sees the prompt — the message never arrived — and
so no CLI acknowledgement is ever posted either.  The pending
``hitl_queue`` row would otherwise sit there forever.  A future
production reaper (sketch in this fixture's test-scoped helper)
detects this transport failure by scanning ``audit_log`` for
``hitl.slack_notify_failed`` events whose corresponding
``hitl_queue`` row is still ``'pending'``, and emits a REFUSE
Decision with ``failure_mode=HITL_TIMEOUT``.

Transport vs substantive grounds: HITL_TIMEOUT vs UNAPPROVED_HIGH_RISK
---------------------------------------------------------------------
:attr:`FailureMode.HITL_TIMEOUT` is a **transport-grade** failure
mode — "the message did not arrive".  Its trigger is a *delivery
failure*: the notifier raised, the audit log proves the message was
never sent, and therefore no human can have acknowledged it.  The
disposition Decision refuses on those transport grounds.

This contrasts with :attr:`FailureMode.UNAPPROVED_HIGH_RISK` (see
``tests/integration/test_failuremode_unapproved_high_risk.py``),
which is a *substantive-grounds* refusal: the Slack message DID
arrive, a human MIGHT have seen it, but the deadline elapsed without
an explicit approval and the conservative-default policy refuses the
trade for not carrying an authorised signature.  Both failure modes
can land a high-risk trade in REFUSE, but the rationale, the
``failure_mode`` enum, and the operator playbook differ — hence two
fixtures.

Test mechanics
--------------
* :class:`_FailingNotifier` is a minimal stub implementing the
  ``SlackNotifier`` surface used by ``firm/agents/hitl.py``: a single
  ``notify(*, decision)`` method that raises
  ``RuntimeError("503 Slack unavailable")`` unconditionally and
  tracks ``notify_call_count``.
* The fixture wires this stub into ``make_hitl`` and drives the gate
  with an ESCALATE Decision.  The existing production catch at
  ``firm/agents/hitl.py:50-62`` writes ``hitl.slack_notify_failed``
  to ``audit_log``.
* The test-scoped helper
  :func:`_emit_hitl_timeout_for_unreached_escalations` then scans
  ``audit_log`` for those events, joins back to ``hitl_queue`` rows
  still ``'pending'``, and emits a REFUSE Decision per failure with
  ``failure_mode=HITL_TIMEOUT`` and a ``decision_id_chain`` pointing
  back to the un-notified ESCALATE.

Intentional non-decisions
-------------------------
The reaper helper does **not** mutate the ``hitl_queue`` row (it
remains ``'pending'``).  An operator who later discovers the outage
via the audit log can still CLI-ack the original ESCALATE through an
alternate channel; the HITL_TIMEOUT REFUSE is a "best-current-
disposition" emit, not a terminal lifecycle transition for the queue
row.  Whether the queue-row lifecycle should transition
(``'pending'`` → ``'notify_failed'``, for instance) is an open
question the production reaper will resolve; this fixture
deliberately leaves it alone.

The helper also does **not** modify ``firm/agents/hitl.py`` — T26 is
a pure registry-flip + one-new-fixture task and the production
reaper is out of scope.  A future production reaper would implement
this same transport-grade disposition policy in
``firm/agents/hitl.py`` (or a sibling reaper module); this fixture
is the test-side specification of what that production reaper must
do.

The FAILURE_MODE_FIXTURES registry entry for HITL_TIMEOUT points at
this fixture.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.agents.hitl import make_hitl
from firm.agents.reporter import _persist_decisions_from_state
from firm.core.clock import Clock, ReplayClock
from firm.core.ids import sign_nonce, ulid_new
from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Decision,
    EscalatePayload,
    FailureMode,
    RefusePayload,
)
from firm.db.connection import get_conn
from firm.db.migrations import init_db


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 3, 13, tzinfo=timezone.utc)
_NONCE_SECRET = b"x" * 32


# ---------------------------------------------------------------------------
# Failing-notifier stub
# ---------------------------------------------------------------------------


class _FailingNotifier:
    """Minimal ``SlackNotifier``-shaped stub whose ``notify`` always raises.

    Implements the same kwargs-only ``notify(*, decision)`` surface that
    ``firm/agents/hitl.py`` calls.  Every invocation raises
    ``RuntimeError("503 Slack unavailable")`` (representative of an HTTP
    5xx from the real Slack SDK) and increments ``notify_call_count`` so
    the test can assert the production code actually attempted the
    notification.
    """

    def __init__(self) -> None:
        self.notify_call_count = 0

    def notify(self, *, decision: Decision) -> None:  # noqa: ARG002
        self.notify_call_count += 1
        raise RuntimeError("503 Slack unavailable")


# ---------------------------------------------------------------------------
# Test-scoped reaper (a future production reaper would implement this policy)
# ---------------------------------------------------------------------------


def _emit_hitl_timeout_for_unreached_escalations(
    *,
    db_path: Path,
    clock: Clock,
    nonce_secret: bytes,
) -> list[Decision]:
    """Sweep ``audit_log`` for ``hitl.slack_notify_failed`` events whose
    corresponding ``hitl_queue`` row is still ``'pending'``, and emit one
    REFUSE :class:`Decision` per such row with
    ``failure_mode=HITL_TIMEOUT``.

    Transport-grade policy: when ``notifier.notify(...)`` raises, the
    Slack message never arrived and so no human can have acknowledged
    it.  The disposition Decision refuses on those *transport* grounds —
    distinct from :attr:`FailureMode.UNAPPROVED_HIGH_RISK`, which
    encodes the substantive-grounds refusal when a delivered message
    aged out without explicit approval (see
    ``tests/integration/test_failuremode_unapproved_high_risk.py``).

    This helper is test-scoped (leading underscore).  A future
    production reaper would implement this same transport-grade
    disposition policy in ``firm/agents/hitl.py`` (or a sibling reaper
    module); this fixture is the test-side specification of what that
    production reaper must do.  The helper deliberately does NOT mutate
    the ``hitl_queue`` row — an operator who later discovers the
    outage via the audit log can still CLI-ack the original ESCALATE
    through an alternate channel.  Whether the queue-row lifecycle
    should transition (e.g., ``'pending'`` → ``'notify_failed'``) is an
    open question the production reaper will resolve.
    """
    now = clock.now()
    refused: list[Decision] = []
    with closing(get_conn(db_path)) as conn:
        # Find all ``hitl.slack_notify_failed`` audit events; the JSON
        # ``detail`` payload carries the ``decision_id`` of the ESCALATE
        # whose notification failed.
        audit_rows = conn.execute(
            "SELECT detail FROM audit_log WHERE event = ?",
            ("hitl.slack_notify_failed",),
        ).fetchall()
        for audit_row in audit_rows:
            detail = json.loads(audit_row["detail"])
            escalate_id = detail["decision_id"]
            # Only emit a HITL_TIMEOUT REFUSE if the queue row is still
            # 'pending' — an operator may have CLI-acked through an
            # alternate channel since the notify_failed event was logged.
            # (N+1 fine here because audit_log has <=O(escalations)
            # notify-failed rows; production reaper should rewrite as a
            # single JOIN.)
            queue_row = conn.execute(
                "SELECT status FROM hitl_queue WHERE decision_id = ?",
                (escalate_id,),
            ).fetchone()
            if queue_row is None or queue_row["status"] != "pending":
                continue
            refuse_id = ulid_new()
            nonce = sign_nonce(
                nonce_secret,
                decision_id=refuse_id,
                timestamp=int(now.timestamp()),
            )
            refuse = Decision(
                id=refuse_id,
                decision_id_chain=[escalate_id],
                action=ActionEnum.REFUSE,
                payload=RefusePayload(reason="hitl:notifier_unavailable"),
                rationale=(
                    f"Slack notifier failed to deliver ESCALATE "
                    f"{escalate_id!r} (transport-grade timeout — no human "
                    f"can have acknowledged a message that never arrived)"
                ),
                confidence=0.0,
                citations=[],
                falsification_condition=(
                    "operator CLI-acks the escalation through an alternate "
                    "channel"
                ),
                escalation_reason=None,
                failure_mode=FailureMode.HITL_TIMEOUT,
                metadata={
                    "agent": "hitl",
                    "expired_decision_id": escalate_id,
                    "transport_failure": "slack",
                },
                nonce=nonce,
            )
            _persist_decisions_from_state(
                {"risk_decision": refuse}, db_path, clock
            )
            refused.append(refuse)
    return refused


# ---------------------------------------------------------------------------
# Primary fixture — HITL_TIMEOUT end-to-end
# ---------------------------------------------------------------------------


def test_failed_slack_notify_on_pending_hitl_row_emits_refuse_with_hitl_timeout(
    tmp_path: Path,
) -> None:
    """A failing Slack ``notify`` on a ``'pending'`` ``hitl_queue`` row
    produces a REFUSE Decision with ``failure_mode=HITL_TIMEOUT`` and a
    ``decision_id_chain`` back to the un-notified ESCALATE; the original
    ESCALATE row is not mutated; the ``hitl_queue`` row remains
    ``'pending'`` (intentional non-decision so the operator can still
    CLI-ack through an alternate channel).
    """
    # --- infrastructure setup -----------------------------------------------
    db_path = tmp_path / "firm.db"
    init_db(db_path)

    clock = ReplayClock(_T0)

    # --- Step 1: construct upstream "proposal" Decision (PM/Research BUY) ----
    # The trade need not be high-NAV; HITL_TIMEOUT is a transport failure
    # mode and applies to any ESCALATE whose Slack notification fails,
    # not just to high-risk trades.  We synthesise a modest BUY so the
    # fixture remains focused on the Slack-transport path.
    proposal_id = ulid_new()
    proposal_nonce = sign_nonce(
        _NONCE_SECRET,
        decision_id=proposal_id,
        timestamp=int(clock.now().timestamp()),
    )
    proposal = Decision(
        id=proposal_id,
        decision_id_chain=[],
        action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("100")),
        rationale="AAPL strong FCF; integration fixture for HITL_TIMEOUT.",
        confidence=0.7,
        citations=[],
        falsification_condition="AAPL FCF falls below $50B in FY2025.",
        escalation_reason=None,
        failure_mode=None,
        metadata={"agent": "pm", "ticker": "AAPL"},
        nonce=proposal_nonce,
    )

    # --- Step 2: the Risk ESCALATE Decision (what Risk would have produced) --
    # Any ESCALATE reason works; the HITL_TIMEOUT scenario is about the
    # notification failing, not about why the trade was escalated.
    escalate_id = ulid_new()
    escalate_nonce = sign_nonce(
        _NONCE_SECRET,
        decision_id=escalate_id,
        timestamp=int(clock.now().timestamp()),
    )
    escalate = Decision(
        id=escalate_id,
        decision_id_chain=[proposal_id],
        action=ActionEnum.ESCALATE,
        payload=EscalatePayload(
            proposed=BuyPayload(ticker="AAPL", shares=Decimal("100")),
            reason="HITL required for fixture (transport-failure path)",
        ),
        rationale="HITL required: fixture exercises Slack-notify failure path",
        confidence=1.0,
        citations=[],
        falsification_condition="HITL Slack notification delivery",
        escalation_reason="fixture: notifier will raise",
        failure_mode=None,
        metadata={"agent": "risk", "ticker": "AAPL"},
        nonce=escalate_nonce,
    )

    # Persist the upstream proposal so the FK on the ESCALATE row (via
    # decision_id_chain) is satisfiable; ``make_hitl`` will persist the
    # ESCALATE itself.
    _persist_decisions_from_state(
        {"pm_decision": proposal}, db_path, clock
    )

    # --- Step 3: drive the HITL gate with a failing notifier -----------------
    notifier = _FailingNotifier()
    hitl = make_hitl(db_path=db_path, clock=clock, notifier=notifier)
    state: dict[str, Decision] = {"risk_decision": escalate}
    result = hitl(state)

    # The gate must report HITL required + not approved; the production
    # catch at firm/agents/hitl.py:50-62 swallowed the notifier exception
    # and wrote ``hitl.slack_notify_failed`` to audit_log.
    assert result == {"hitl_required": True, "hitl_approved": False}, (
        f"expected HITL-required + not-approved, got {result!r}"
    )
    assert notifier.notify_call_count == 1, (
        f"notifier.notify must have been called exactly once; "
        f"got {notifier.notify_call_count}"
    )

    # Confirm the production catch wrote the expected audit entry.
    with closing(get_conn(db_path)) as conn:
        notify_failed_rows = conn.execute(
            "SELECT detail FROM audit_log WHERE event = ?",
            ("hitl.slack_notify_failed",),
        ).fetchall()
    assert len(notify_failed_rows) == 1, (
        f"expected exactly one hitl.slack_notify_failed audit entry; "
        f"got {len(notify_failed_rows)}"
    )

    # --- Step 4: invoke the test-scoped reaper -------------------------------
    refused = _emit_hitl_timeout_for_unreached_escalations(
        db_path=db_path,
        clock=clock,
        nonce_secret=_NONCE_SECRET,
    )

    # --- Step 5: assertions on the returned Decision -------------------------
    assert len(refused) == 1, (
        f"expected exactly one transport-failed escalation, got "
        f"{len(refused)}: {refused!r}"
    )
    refuse = refused[0]
    assert refuse.action == ActionEnum.REFUSE, (
        f"expected REFUSE, got {refuse.action}"
    )
    assert refuse.failure_mode == FailureMode.HITL_TIMEOUT, (
        f"expected HITL_TIMEOUT, got {refuse.failure_mode}"
    )
    assert refuse.decision_id_chain == [escalate_id], (
        f"refuse.decision_id_chain must point back to the un-notified "
        f"ESCALATE; expected [{escalate_id!r}], got {refuse.decision_id_chain!r}"
    )

    # --- Step 6: DB assertions — REFUSE row written with HITL_TIMEOUT --------
    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = conn.execute(
            "SELECT id, failure_mode FROM decisions WHERE failure_mode = ?",
            ("hitl_timeout",),
        ).fetchall()
    assert len(rows) >= 1, (
        "decisions table must contain at least one row with "
        "failure_mode='hitl_timeout'"
    )

    # --- Step 7: original ESCALATE row not retroactively mutated -------------
    with closing(sqlite3.connect(str(db_path))) as conn:
        escalate_row = conn.execute(
            "SELECT id, failure_mode FROM decisions WHERE id = ?",
            (escalate_id,),
        ).fetchone()
    assert escalate_row is not None, (
        f"original ESCALATE row {escalate_id!r} missing from decisions table"
    )
    assert escalate_row[1] is None, (
        f"original ESCALATE row must keep failure_mode IS NULL "
        f"(reaper must NOT mutate the upstream Decision); got {escalate_row[1]!r}"
    )

    # --- Step 8: hitl_queue row remains 'pending' (intentional non-decision) -
    # The reaper emits a disposition Decision but does NOT mutate the
    # hitl_queue row.  An operator can still CLI-ack the original
    # ESCALATE through an alternate channel; the HITL_TIMEOUT REFUSE is
    # a "best-current-disposition" emit, not a terminal lifecycle
    # transition.
    with closing(get_conn(db_path)) as conn:
        hitl_row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id = ?",
            (escalate_id,),
        ).fetchone()
    assert hitl_row is not None, (
        f"hitl_queue row for {escalate_id!r} missing"
    )
    assert hitl_row["status"] == "pending", (
        f"reaper must NOT mutate hitl_queue row status (open question for "
        f"future production reaper); expected 'pending', got {hitl_row['status']!r}"
    )
