"""Plan 4 T23 — TOOL_PERMISSION_DENIED end-to-end fixture.

Wires a :class:`firm.broker.capability.RestrictedBroker` with
``agent_role="research"`` around an inner :class:`FakeBroker`.  Any code
path that attempts the order-placing surface (``broker.place_order``,
implemented at the protocol level as ``Broker.submit``) from a
non-execution role is rejected by the capability layer with
:class:`ToolPermissionDeniedError` BEFORE the inner broker is touched.

The primary test demonstrates the failure mode is reachable end-to-end:

  1. Construct ``restricted = RestrictedBroker(inner=FakeBroker(...),
     agent_role="research")``.
  2. Attempt the privileged call; assert the exception fires with the
     correct ``(role, tool)`` attributes.
  3. Build a REFUSE :class:`Decision` with ``failure_mode =
     TOOL_PERMISSION_DENIED`` mirroring what an agent would emit when the
     capability layer rejected its tool call.
  4. Persist that Decision via the reporter pattern (
     :func:`firm.agents.reporter._persist_decisions_from_state`) into a
     temp SQLite DB.
  5. Assert the audit row is queryable by
     ``WHERE failure_mode = 'tool_permission_denied'``.
  6. Assert the inner broker was NOT invoked (capability check rejects
     before delegation).

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
capability layer is the new production-time enforcement that this
fixture also exercises.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from firm.agents.reporter import _persist_decisions_from_state
from firm.broker.capability import RestrictedBroker, ToolPermissionDeniedError
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.ids import sign_nonce, ulid_new
from firm.core.models import (
    ActionEnum,
    Decision,
    FailureMode,
    RefusePayload,
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
    """Research-role broker.place_order => ToolPermissionDeniedError =>
    REFUSE / TOOL_PERMISSION_DENIED Decision persisted to audit table; inner
    broker is never touched.
    """
    # --- infrastructure setup -----------------------------------------------
    db_path = tmp_path / "firm.db"
    init_db(db_path)

    clock = ReplayClock(_T0)
    inner = FakeBroker(initial_cash=Decimal("100000"))
    restricted = RestrictedBroker(inner=inner, agent_role="research")

    # --- Step 1: capability layer rejects the privileged call ----------------
    with pytest.raises(ToolPermissionDeniedError) as exc_info:
        restricted.submit(_build_buy_payload_dict(), idempotency_key="t23-rejected-1")

    assert exc_info.value.role == "research", (
        f"expected role='research', got {exc_info.value.role!r}"
    )
    assert exc_info.value.tool == "broker.place_order", (
        f"expected tool='broker.place_order', got {exc_info.value.tool!r}"
    )

    # --- Step 2: build REFUSE / TOOL_PERMISSION_DENIED Decision --------------
    decision_id = ulid_new()
    nonce = sign_nonce(
        _NONCE_SECRET,
        decision_id=decision_id,
        timestamp=int(clock.now().timestamp()),
    )
    decision = Decision(
        id=decision_id,
        decision_id_chain=[],
        action=ActionEnum.REFUSE,
        payload=RefusePayload(reason="capability:tool_permission_denied"),
        rationale=(
            f"agent_role={exc_info.value.role!r} attempted to invoke "
            f"tool={exc_info.value.tool!r}; capability layer denied"
        ),
        confidence=0.0,
        citations=[],
        falsification_condition=(
            "research agent has its broker reference rotated to "
            "agent_role='execution' (out-of-spec configuration)"
        ),
        escalation_reason=None,
        failure_mode=FailureMode.TOOL_PERMISSION_DENIED,
        metadata={"agent": "research", "ticker": "AAPL"},
        nonce=nonce,
    )

    assert decision.action == ActionEnum.REFUSE
    assert decision.failure_mode == FailureMode.TOOL_PERMISSION_DENIED

    # --- Step 3: persist via the reporter helper -----------------------------
    _persist_decisions_from_state(
        {"research_decision": decision}, db_path, clock
    )

    # --- Step 4: audit row queryable -----------------------------------------
    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = conn.execute(
            "SELECT id, action, failure_mode FROM decisions WHERE failure_mode = ?",
            ("tool_permission_denied",),
        ).fetchall()
    assert len(rows) >= 1, (
        "decisions table must contain at least one row with "
        "failure_mode='tool_permission_denied'"
    )
    assert rows[0][0] == decision_id
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
