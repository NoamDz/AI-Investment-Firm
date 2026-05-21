"""Plan 4 T24 — UNAPPROVED_HIGH_RISK end-to-end fixture.

Scenario
--------
A heartbeat produces a trade that would consume more than 3% of NAV.
The Risk agent ESCALATEs the proposal to a human; the HITL gate
persists the ESCALATE Decision, inserts a ``hitl_queue`` row with
status ``'pending'`` keyed by the ESCALATE Decision id, and waits.
In this fixture, **no approval is ever posted**.  A configurable
timeout deadline (30 minutes by default, matching spec §8.5) then
elapses.

Conservative-default policy: UNAPPROVED_HIGH_RISK vs HITL_TIMEOUT
-----------------------------------------------------------------
When a high-risk trade ages out of the HITL queue without an explicit
human approval, the disposition is **not** a bare ``HITL_TIMEOUT``
(which would be a transport/UX failure mode — "the message did not
arrive"); it is :attr:`FailureMode.UNAPPROVED_HIGH_RISK` — the trade
is refused on substantive grounds that no authorised human signed
off.  The two failure modes have the same trigger (deadline elapsed
on a pending HITL row) but materially different post-timeout
dispositions, and the distinction is encoded in the ``failure_mode``
stamped on the emitted REFUSE Decision, in the rationale text, and
in the conservative-default policy documented below.

Test mechanics
--------------
The test mocks the timer via the module-level sentinel
``_TIMEOUT_DEADLINE_SECONDS = 1800`` (30 min, matching spec §8.5)
and the :class:`ReplayClock.advance` API in
``firm/core/clock.py``.

Because production has no HITL timeout reaper yet (the production
``firm/agents/hitl.py`` only inserts pending rows; it never sweeps
them), the test inlines a test-scoped helper
:func:`_reap_expired_high_risk_hitl_entries` that scans
``hitl_queue`` for rows whose age has exceeded the deadline and, for
each, constructs and persists a REFUSE Decision with
``failure_mode=UNAPPROVED_HIGH_RISK`` and a
``decision_id_chain`` pointing back to the aged-out ESCALATE.  The
helper docstring explicitly says **a future production reaper would
implement this same policy** so a reader does not mistake the
helper for a stub patching over missing production code; rather,
this fixture is the test-side specification of what that production
reaper must do.

Intentional non-decisions
-------------------------
The reaper helper does **not** mutate the ``hitl_queue`` row (it
remains ``'pending'``).  Whether the queue-row lifecycle should
transition (``'pending'`` → ``'timed_out'``, for instance) is a
separate question the production reaper will resolve; this fixture
exercises only the disposition-Decision emission and leaves the
queue lifecycle untouched as an explicit, documented non-decision.

Complements (does not replace)
``tests/integration/test_failuremode_stale_data.py``; the
FAILURE_MODE_FIXTURES registry entry for UNAPPROVED_HIGH_RISK
points at this fixture.  The closely related ``HITL_TIMEOUT`` mode
remains in ALLOWED_GAPS for now — its triggering site is out of
scope for T24.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

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

# Spec §8.5 default: HITL approvals must arrive within 30 minutes or the
# pending row is treated as unapproved-high-risk.  Encoded here as a
# module-level sentinel so the test mocks the "timer" by advancing the
# ReplayClock past this many seconds.
_TIMEOUT_DEADLINE_SECONDS = 1800


# ---------------------------------------------------------------------------
# Test-scoped reaper (a future production reaper would implement this policy)
# ---------------------------------------------------------------------------


def _reap_expired_high_risk_hitl_entries(
    *,
    db_path: Path,
    clock: Clock,
    deadline_seconds: int,
    nonce_secret: bytes,
) -> list[Decision]:
    """Sweep ``hitl_queue`` for aged-out pending rows and emit refusals.

    For every row where ``status='pending'`` and ``clock.now() - queued_at``
    exceeds ``deadline_seconds``, construct a REFUSE :class:`Decision` with
    ``failure_mode=UNAPPROVED_HIGH_RISK`` and a ``decision_id_chain``
    pointing back to the expired ESCALATE.  Persist each new disposition
    Decision via :func:`_persist_decisions_from_state` and return the
    list.

    Conservative-default policy: when a high-risk trade ages out without
    an explicit human approval, the disposition is NOT a bare
    ``HITL_TIMEOUT`` (which would be a transport/UX failure mode); it is
    ``UNAPPROVED_HIGH_RISK`` — the trade is refused on the substantive
    grounds that no authorised human signed off, which is materially
    different from "the message didn't arrive".

    This helper is test-scoped (leading underscore).  A future production
    reaper would implement this same policy in
    ``firm/agents/hitl.py`` (or a sibling reaper module); this fixture
    is the test-side specification of what that production reaper must
    do.  The helper deliberately does NOT mutate the ``hitl_queue`` row
    — whether the queue-row lifecycle should transition (e.g.,
    ``'pending'`` → ``'timed_out'``) is an open question the production
    reaper will resolve.
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


# ---------------------------------------------------------------------------
# Primary fixture — UNAPPROVED_HIGH_RISK end-to-end
# ---------------------------------------------------------------------------


def test_aged_pending_high_risk_hitl_row_emits_refuse_with_unapproved_high_risk_failure_mode(
    tmp_path: Path,
) -> None:
    """Aged-out HITL row keyed to a >3% NAV ESCALATE => REFUSE Decision
    with ``failure_mode=UNAPPROVED_HIGH_RISK`` and a ``decision_id_chain``
    back to the expired ESCALATE; original ESCALATE row is not mutated;
    ``hitl_queue`` row is left ``'pending'`` (lifecycle non-decision).
    """
    # --- infrastructure setup -----------------------------------------------
    db_path = tmp_path / "firm.db"
    init_db(db_path)

    clock = ReplayClock(_T0)

    # --- Step 1: construct upstream "proposal" Decision (PM/Research BUY) ----
    # This represents the upstream agent's wish to BUY a deliberately-large
    # quantity that would exceed 3% NAV of a $100k portfolio.  In production
    # Risk would compute the proposed share value × quote price against NAV
    # and decide to ESCALATE; we synthesise both the proposal and the
    # downstream ESCALATE directly so this test can focus on the post-HITL
    # disposition rather than re-exercising the Risk evaluator (which is
    # already covered by tests/integration/test_failuremode_stale_data.py
    # and tests/unit/test_risk_limits.py).
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
        payload=BuyPayload(ticker="AAPL", shares=Decimal("5000")),
        rationale="AAPL strong FCF; integration fixture for UNAPPROVED_HIGH_RISK.",
        confidence=0.7,
        citations=[],
        falsification_condition="AAPL FCF falls below $50B in FY2025.",
        escalation_reason=None,
        failure_mode=None,
        metadata={"agent": "pm", "ticker": "AAPL"},
        nonce=proposal_nonce,
    )

    # --- Step 2: the Risk ESCALATE Decision (what Risk would have produced) --
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
            proposed=BuyPayload(ticker="AAPL", shares=Decimal("5000")),
            reason="position pct exceeds max_position_pct",
        ),
        rationale="HITL required: position pct exceeds max_position_pct",
        confidence=1.0,
        citations=[],
        falsification_condition="HITL approval timeout",
        escalation_reason="position 0.500 > 0.03",
        failure_mode=None,
        metadata={"agent": "risk", "ticker": "AAPL"},
        nonce=escalate_nonce,
    )

    # Persist both — mirrors what the reporter / hitl gate would do.
    _persist_decisions_from_state(
        {"pm_decision": proposal, "risk_decision": escalate}, db_path, clock
    )

    # --- Step 3: simulate the HITL gate inserting a pending row --------------
    # Mirrors the INSERT in firm/agents/hitl.py:36-41.  No notifier is
    # involved because this fixture is about post-timeout disposition, not
    # the Slack/CLI notification path.
    with closing(get_conn(db_path)) as conn:
        conn.execute(
            "INSERT INTO hitl_queue (decision_id, queued_at, status) "
            "VALUES (?, ?, 'pending')",
            (escalate_id, clock.now().isoformat()),
        )

    # --- Step 4: mock the timer — advance past the deadline ------------------
    # No approval is ever posted.  Advance the clock by 31 minutes (just
    # over the 30-minute deadline encoded in _TIMEOUT_DEADLINE_SECONDS).
    clock.advance(31 * 60)

    # --- Step 5: invoke the test-scoped reaper -------------------------------
    refused = _reap_expired_high_risk_hitl_entries(
        db_path=db_path,
        clock=clock,
        deadline_seconds=_TIMEOUT_DEADLINE_SECONDS,
        nonce_secret=_NONCE_SECRET,
    )

    # --- Step 6: assertions on the returned Decision -------------------------
    assert len(refused) == 1, (
        f"expected exactly one aged-out row, got {len(refused)}: {refused!r}"
    )
    refuse = refused[0]
    assert refuse.action == ActionEnum.REFUSE, (
        f"expected REFUSE, got {refuse.action}"
    )
    assert refuse.failure_mode == FailureMode.UNAPPROVED_HIGH_RISK, (
        f"expected UNAPPROVED_HIGH_RISK, got {refuse.failure_mode}"
    )
    assert refuse.decision_id_chain == [escalate_id], (
        f"refuse.decision_id_chain must point back to the expired ESCALATE; "
        f"expected [{escalate_id!r}], got {refuse.decision_id_chain!r}"
    )

    # --- Step 7: DB assertions — audit row written ---------------------------
    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = conn.execute(
            "SELECT id, failure_mode FROM decisions WHERE failure_mode = ?",
            ("unapproved_high_risk",),
        ).fetchall()
    assert len(rows) >= 1, (
        "decisions table must contain at least one row with "
        "failure_mode='unapproved_high_risk'"
    )

    # --- Step 8: original ESCALATE row not retroactively mutated -------------
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

    # --- Step 9: hitl_queue row remains 'pending' (intentional non-decision) -
    # The reaper emits a disposition Decision but does NOT mutate the
    # hitl_queue row.  Whether the queue row should transition to
    # 'timed_out' is an open question for the production reaper; this
    # test pins that the current test-scoped reaper leaves it alone.
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
