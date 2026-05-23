"""Plan 4 T23 — TOOL_PERMISSION_DENIED end-to-end fixture.

Wires a :class:`firm.broker.capability.RestrictedBroker` with
``agent_role="research"`` around an inner :class:`FakeBroker`, hands it
to the production execution node via :func:`make_execution`, and runs a
BUY risk Decision through the node.  The outbox's bounded-retry loop
re-raises :class:`ToolPermissionDeniedError` (permission errors are
not transient), and the execution agent stamps a REFUSE Decision with
``failure_mode=TOOL_PERMISSION_DENIED`` via
:func:`_build_tool_permission_denied_refuse`.

The primary test demonstrates the failure mode is reachable end-to-end
via the production code path:

  1. Construct ``restricted = RestrictedBroker(inner=FakeBroker(...),
     agent_role="research")`` — the wrong role for placing orders.
  2. Build a BUY risk Decision and invoke the execution node directly
     (this is the production code path; no test-local stamping).
  3. Assert the returned Decision (persisted via the agent's own
     ``_persist_decisions_from_state`` call) has
     ``failure_mode=TOOL_PERMISSION_DENIED``.
  4. Assert the audit row is queryable by
     ``WHERE failure_mode = 'tool_permission_denied'``.
  5. Assert the inner broker was NOT invoked (capability check rejects
     before delegation; no order leaked through).

Two additional unit-level tests pin down that the gate is SELECTIVE —
not a blanket block:

  * ``test_execution_role_broker_call_passes_through`` — the privileged
    role can place orders normally; the inner broker is invoked.
  * ``test_research_role_read_only_methods_pass_through`` — read-only
    methods (``get_quote`` / ``get_cash`` / ``list_positions``) are
    unconditionally allowed for any role.

Complements (does not replace)
``tests/red_team/conftest.py::assert_no_privileged_action``; that helper
remains the test-time defense-in-depth invariant across four channels
(decision.action allowlist, broker_calls, audit_log, outbox).  The
capability layer is the production-time enforcement that this fixture
exercises end-to-end through the execution agent (Bundle E).
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.agents.execution import make_execution
from firm.broker.capability import RestrictedBroker
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.ids import sign_nonce, ulid_new
from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Decision,
)
from firm.db.migrations import init_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_T0 = datetime(2024, 3, 13, tzinfo=timezone.utc)
_NONCE_SECRET = b"x" * 32


def _build_buy_payload_dict() -> dict[str, object]:
    """Construct the dict payload that Broker.submit (alias broker.place_order) expects.

    Mirrors the shape that :class:`firm.broker.fake_broker.FakeBroker.submit`
    expects: ``{"ticker", "shares", "kind"}``.  We use this for both the
    rejection path and the pass-through positive case so the two tests
    exercise the same surface.
    """
    return {"ticker": "AAPL", "shares": "10", "kind": "buy"}


# ---------------------------------------------------------------------------
# Primary fixture — TOOL_PERMISSION_DENIED end-to-end
# ---------------------------------------------------------------------------


def test_research_role_broker_call_rejected_with_tool_permission_denied_failure_mode(
    tmp_path: Path,
) -> None:
    """Production path: BUY risk Decision + research-role RestrictedBroker
    => execution agent emits a REFUSE Decision with
    ``failure_mode=TOOL_PERMISSION_DENIED`` and the inner broker is never
    touched.

    Wired through:

      * :class:`RestrictedBroker` with ``agent_role="research"`` (wrong
        role for ``broker.place_order``).
      * :func:`firm.outbox.outbox.place_order_via_outbox` — the retry
        loop now re-raises ``ToolPermissionDeniedError`` immediately
        instead of misclassifying it as ``BrokerUnavailableError``.
      * :func:`firm.agents.execution.make_execution` — the typed
        handler stamps ``FailureMode.TOOL_PERMISSION_DENIED`` on a REFUSE
        Decision and persists it via the agent's own
        ``_persist_decisions_from_state`` call.
    """
    # --- infrastructure setup -----------------------------------------------
    db_path = tmp_path / "firm.db"
    init_db(db_path)

    clock = ReplayClock(_T0)
    inner = FakeBroker(initial_cash=Decimal("100000"))
    # Wrong role: ``research`` is NOT in the capability layer's privileged
    # allowlist, so any submit() through ``restricted`` must raise
    # ToolPermissionDeniedError before reaching ``inner``.
    restricted = RestrictedBroker(inner=inner, agent_role="research")

    # --- Step 1: build a BUY risk Decision the execution agent will act on --
    risk_decision_id = ulid_new()
    risk_nonce = sign_nonce(
        _NONCE_SECRET,
        decision_id=risk_decision_id,
        timestamp=int(clock.now().timestamp()),
    )
    risk_decision = Decision(
        id=risk_decision_id,
        decision_id_chain=[],
        action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="risk-approved BUY for T23 capability fixture",
        confidence=0.7,
        citations=[],
        falsification_condition="risk gate flips",
        escalation_reason=None,
        failure_mode=None,
        metadata={"agent": "risk", "ticker": "AAPL"},
        nonce=risk_nonce,
    )

    # --- Step 2: invoke the execution node directly (production code path) --
    execution = make_execution(
        db_path=db_path,
        broker=restricted,
        clock=clock,
        nonce_secret=_NONCE_SECRET,
    )
    result = execution(
        {"risk_decision": risk_decision, "hitl_required": False, "hitl_approved": True}
    )

    # --- Step 3: execution_result reports the production rejection ----------
    exec_result = result["execution_result"]
    assert exec_result.get("skipped") is True, (
        f"expected skipped=True (capability rejection), got {exec_result!r}"
    )
    assert exec_result.get("reason") == "tool_permission_denied", (
        f"expected reason='tool_permission_denied', got {exec_result.get('reason')!r}"
    )
    assert exec_result.get("role") == "research"
    assert exec_result.get("tool") == "broker.place_order"

    # --- Step 4: REFUSE Decision with TOOL_PERMISSION_DENIED persisted -------
    # The execution agent's own _persist_decisions_from_state call writes the
    # REFUSE Decision to the decisions table — no test-local persistence.
    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = conn.execute(
            "SELECT id, action, failure_mode FROM decisions WHERE failure_mode = ?",
            ("tool_permission_denied",),
        ).fetchall()
    assert len(rows) >= 1, (
        "decisions table must contain at least one row with "
        "failure_mode='tool_permission_denied' (stamped by execution agent)"
    )
    assert rows[0][1] == "REFUSE"

    # --- Step 5: inner broker was NOT invoked --------------------------------
    # FakeBroker exposes its placed-order cache as ``_order_cache``; the
    # capability layer must have rejected BEFORE submit reached the inner
    # broker, so this cache must remain empty.
    assert len(inner._order_cache) == 0, (  # noqa: SLF001 — assert on internals
        f"inner broker was reached despite capability rejection; cache={inner._order_cache!r}"
    )
    # Belt-and-suspenders: positions and cash must be unchanged.
    assert inner.get_cash() == Decimal("100000")
    assert inner.list_positions() == []


# ---------------------------------------------------------------------------
# Positive control — execution role passes through
# ---------------------------------------------------------------------------


def test_execution_role_broker_call_passes_through() -> None:
    """``agent_role="execution"`` may invoke broker.place_order without
    raising; the inner broker MUST see the call (positive control proving
    the gate is selective, not a blanket block).
    """
    inner = FakeBroker(initial_cash=Decimal("100000"))
    restricted = RestrictedBroker(inner=inner, agent_role="execution")

    result = restricted.submit(
        _build_buy_payload_dict(), idempotency_key="t23-allowed-1"
    )

    # The inner broker produced an OrderResult — proving delegation occurred.
    assert result.ticker == "AAPL"
    assert result.filled_shares == Decimal("10")
    # And the inner broker's order cache now has the entry.
    assert "t23-allowed-1" in inner._order_cache  # noqa: SLF001 — assert on internals


# ---------------------------------------------------------------------------
# Negative control — read-only methods always pass through
# ---------------------------------------------------------------------------


def test_research_role_read_only_methods_pass_through() -> None:
    """Read-only Broker methods (``get_quote`` / ``get_cash`` /
    ``list_positions``) are unconditionally allowed for any role.  Proves
    the capability layer gates ONLY the order-placing surface.
    """
    inner = FakeBroker(initial_cash=Decimal("100000"))
    restricted = RestrictedBroker(inner=inner, agent_role="research")

    # get_quote: no exception, returns a Quote with the expected ticker.
    quote = restricted.get_quote("AAPL")
    assert quote.ticker == "AAPL"
    assert quote.price > Decimal("0")

    # get_cash: no exception, matches inner.
    assert restricted.get_cash() == Decimal("100000")

    # list_positions: no exception, returns the inner's positions
    # (empty in this fixture's setup).
    assert restricted.list_positions() == []
