"""Plan 4 T25 — BROKER_UNAVAILABLE end-to-end fixture.

Scenario
--------
A heartbeat produces a normal-sized BUY Decision (well within risk
limits, so Risk waves it through to Execution).  The broker MCP is
down: every ``submit`` call raises a 503-like exception.  The
production execution agent (a) places the order via the transactional
outbox (which writes a ``'pending'`` row BEFORE calling the broker —
see spec §5.2 crash semantics), then (b) hits the bounded retry loop
inside :func:`place_order_via_outbox`, which retries ``N`` times with
the SAME idempotency key.  When all attempts are exhausted, the
outbox raises :class:`firm.outbox.outbox.BrokerUnavailableError`; the
execution agent catches it and emits a REFUSE :class:`Decision`
stamped with ``failure_mode=BROKER_UNAVAILABLE``, persists it, and
returns a ``skipped`` execution_result so the heartbeat reaches the
reporter without losing the unfilled order.

Production semantics
--------------------
Per spec §10.5 line 222:

    Broker API down / timeout | submit timeout > 10s or 5xx |
    Outbox stays 'pending'; retry with same idempotency key (§5.2).
    After N retries, abort decision and surface unfilled order in EOD
    report | BROKER_UNAVAILABLE.

The contract this fixture pins:

* The outbox row exists in ``'pending'`` state BEFORE any broker call
  (the INSERT is in a separate transaction that commits first), so a
  broker crash mid-loop cannot lose the row.
* Retries use the SAME idempotency key (sha256(decision.id:nonce)) so
  the broker can dedupe its own state on the next attempt — spec §5.2
  invariant.
* After ``N`` exhausted attempts the outbox row STAYS ``'pending'``;
  the disposition REFUSE Decision is the new audit-trail row.  The
  next heartbeat's :func:`recover_pending` will retry the same row
  with the same key when the broker comes back; **this fixture does
  NOT exercise** ``recover_pending`` (covered separately in
  ``tests/unit/test_outbox.py``).
* The execution agent emits a REFUSE Decision with
  ``decision_id_chain=[risk.id]`` so the audit trail points back to
  the executable risk decision whose submit failed; the original BUY
  Decision is NOT mutated (its ``failure_mode`` stays NULL).

Test mechanics
--------------
A ``_FailingBroker`` subclass conforms to the :class:`Broker`
protocol and unconditionally raises ``RuntimeError("503 Service
Unavailable")`` on every ``submit`` call, while delegating
``list_positions`` / ``get_cash`` / ``get_quote`` to a real
:class:`FakeBroker` so the rest of the agent graph (which queries
positions / cash for sizing) keeps working.  The test counts
``submit`` invocations to pin the retry budget.

The clock is a :class:`ReplayClock` so timestamps are deterministic;
no real sleeps occur (the retry loop is intentionally synchronous
and tight — see :func:`place_order_via_outbox` for the inline comment
explaining the no-backoff choice).

Intentional non-decisions
-------------------------
The fixture deliberately does NOT:

* Mutate the outbox row's ``status`` on broker exhaustion — leaving
  it ``'pending'`` is the entire point (it is what enables
  next-heartbeat recovery per spec §5.2).
* Touch the original BUY Decision row in the decisions table on the
  catch — that would corrupt the audit trail by retroactively
  overwriting an already-emitted decision's ``failure_mode``.
* Exercise the recovery path (``recover_pending``) — that's covered
  in ``tests/unit/test_outbox.py`` and is the responsibility of the
  next heartbeat, not the failing one.
* Sleep between attempts.  Backoff is intentionally NOT implemented
  here; if real ops want it, that's a separate task with its own
  policy decision about budget vs latency.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from firm.agents.execution import make_execution
from firm.agents.reporter import _persist_decisions_from_state
from firm.broker.fake_broker import FakeBroker
from firm.broker.protocol import OrderResult, Position, Quote
from firm.core.clock import ReplayClock
from firm.core.ids import sign_nonce, ulid_new
from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Decision,
)
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.outbox.outbox import _idempotency_key


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 3, 13, tzinfo=timezone.utc)
_NONCE_SECRET = b"x" * 32

# Mirrors ``place_order_via_outbox``'s default ``max_attempts`` keyword.  The
# fixture pins this so a future change to the default surfaces here.
_MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Brokers used by the fixture
# ---------------------------------------------------------------------------


class _FailingBroker:
    """:class:`Broker`-protocol broker whose ``submit`` always raises 503.

    Delegates ``list_positions`` / ``get_cash`` / ``get_quote`` to a wrapped
    :class:`FakeBroker` so upstream agents that query the broker for
    positions/cash sizing keep working — the only thing that fails is the
    submit path.
    """

    def __init__(self, *, inner: FakeBroker | None = None) -> None:
        self._inner = inner if inner is not None else FakeBroker(
            initial_cash=Decimal("100000")
        )
        self.submit_call_count: int = 0
        self.last_idempotency_keys: list[str] = []

    def list_positions(self) -> list[Position]:
        return self._inner.list_positions()

    def get_cash(self) -> Decimal:
        return self._inner.get_cash()

    def get_quote(self, ticker: str) -> Quote:
        return self._inner.get_quote(ticker)

    def submit(
        self, decision_payload: dict[str, Any], idempotency_key: str
    ) -> OrderResult:
        self.submit_call_count += 1
        self.last_idempotency_keys.append(idempotency_key)
        # 503-equivalent: the broker MCP is unreachable / returning 5xx.
        # Any opaque ``Exception`` subclass is fair game per the outbox's
        # broad-catch contract; ``RuntimeError`` mirrors what a generic
        # client lib's transport layer would raise on a 5xx.
        raise RuntimeError("503 Service Unavailable")


class _FlakyBroker:
    """:class:`Broker` that fails ``fail_first_n`` times then delegates to
    a real :class:`FakeBroker`.  Used by the recovery-path test below.
    """

    def __init__(
        self, *, fail_first_n: int, inner: FakeBroker | None = None
    ) -> None:
        self._inner = inner if inner is not None else FakeBroker(
            initial_cash=Decimal("100000")
        )
        self._fail_first_n = fail_first_n
        self.submit_call_count: int = 0

    def list_positions(self) -> list[Position]:
        return self._inner.list_positions()

    def get_cash(self) -> Decimal:
        return self._inner.get_cash()

    def get_quote(self, ticker: str) -> Quote:
        return self._inner.get_quote(ticker)

    def submit(
        self, decision_payload: dict[str, Any], idempotency_key: str
    ) -> OrderResult:
        self.submit_call_count += 1
        if self.submit_call_count <= self._fail_first_n:
            raise RuntimeError("503 Service Unavailable")
        return self._inner.submit(decision_payload, idempotency_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_buy_decision(*, clock: ReplayClock) -> Decision:
    """Construct a small BUY Decision well within risk limits."""
    decision_id = ulid_new()
    nonce = sign_nonce(
        _NONCE_SECRET,
        decision_id=decision_id,
        timestamp=int(clock.now().timestamp()),
    )
    return Decision(
        id=decision_id,
        decision_id_chain=[],
        action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="AAPL strong FCF; integration fixture for BROKER_UNAVAILABLE.",
        confidence=0.7,
        citations=[],
        falsification_condition="AAPL FCF falls below $50B in FY2025.",
        escalation_reason=None,
        failure_mode=None,
        metadata={"agent": "risk", "ticker": "AAPL"},
        nonce=nonce,
    )


# ---------------------------------------------------------------------------
# Primary fixture — BROKER_UNAVAILABLE end-to-end
# ---------------------------------------------------------------------------


def test_broker_503_emits_refuse_with_broker_unavailable_and_leaves_outbox_pending(
    tmp_path: Path,
) -> None:
    """503 on every ``broker.submit`` => REFUSE Decision with
    ``failure_mode=BROKER_UNAVAILABLE`` and ``decision_id_chain``
    pointing back to the BUY risk decision; outbox row remains
    ``'pending'`` (intentional, enables next-heartbeat recovery);
    original BUY row is not mutated; submit was attempted exactly
    ``_MAX_ATTEMPTS`` times with the SAME idempotency key.
    """
    # --- infrastructure setup -----------------------------------------------
    db_path = tmp_path / "firm.db"
    init_db(db_path)

    clock = ReplayClock(_T0)

    # --- Step 1: build a BUY risk decision (within limits) -------------------
    buy_decision = _make_buy_decision(clock=clock)

    # Persist the upstream Decision so the outbox INSERT's FK to decisions
    # is satisfied (mirrors what reporter / earlier agents would do).
    _persist_decisions_from_state(
        {"risk_decision": buy_decision}, db_path, clock
    )

    # --- Step 2: invoke the execution agent with a failing broker ------------
    failing = _FailingBroker()

    execution = make_execution(
        db_path=db_path,
        broker=failing,
        clock=clock,
        nonce_secret=_NONCE_SECRET,
    )

    # The execution agent MUST NOT raise; BrokerUnavailableError is caught.
    out = execution({"risk_decision": buy_decision, "hitl_required": False})

    # --- Step 3: assertion — submit was called exactly _MAX_ATTEMPTS times ---
    assert failing.submit_call_count == _MAX_ATTEMPTS, (
        f"expected exactly {_MAX_ATTEMPTS} broker.submit attempts "
        f"(bounded retry budget); got {failing.submit_call_count}"
    )

    # --- Step 4: assertion — every attempt used the SAME idempotency key -----
    # Spec §5.2 invariant: a retry must reuse the original key so the broker
    # can dedupe.  A unique key per attempt would defeat exactly-once.
    expected_key = _idempotency_key(buy_decision)
    assert failing.last_idempotency_keys == [expected_key] * _MAX_ATTEMPTS, (
        f"every retry must reuse the same idempotency_key "
        f"({expected_key!r}); got {failing.last_idempotency_keys!r}"
    )

    # --- Step 5: assertion — REFUSE Decision persisted with failure_mode -----
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, action, failure_mode, parent_chain "
            "FROM decisions WHERE failure_mode = ?",
            ("broker_unavailable",),
        ).fetchall()
    assert len(rows) >= 1, (
        "decisions table must contain at least one row with "
        "failure_mode='broker_unavailable'"
    )
    refuse_row = rows[0]
    assert refuse_row["action"] == ActionEnum.REFUSE.value, (
        f"expected REFUSE action on the BROKER_UNAVAILABLE row, "
        f"got {refuse_row['action']!r}"
    )

    # --- Step 6: assertion — outbox row remains 'pending' --------------------
    with closing(get_conn(db_path)) as conn:
        outbox_row = conn.execute(
            "SELECT status, result FROM outbox WHERE key = ?",
            (expected_key,),
        ).fetchone()
    assert outbox_row is not None, (
        f"outbox row for key {expected_key!r} missing; the INSERT must "
        f"happen BEFORE the retry loop so a broker crash cannot lose the row"
    )
    assert outbox_row["status"] == "pending", (
        f"outbox row must STAY 'pending' on broker exhaustion (enables "
        f"next-heartbeat recover_pending retry); got {outbox_row['status']!r}"
    )

    # --- Step 7: assertion — outbox.result is empty (no broker reply) --------
    # Nothing was ever returned by ``broker.submit`` so the result column
    # must be NULL (or empty / absent) — a populated result here would
    # signal that the retry loop swallowed a half-success.
    assert outbox_row["result"] in (None, ""), (
        f"outbox.result must be NULL/empty when every submit attempt failed; "
        f"got {outbox_row['result']!r}"
    )

    # --- Step 8: assertion — original BUY row not retroactively mutated ------
    with closing(sqlite3.connect(str(db_path))) as conn:
        buy_row = conn.execute(
            "SELECT id, failure_mode FROM decisions WHERE id = ?",
            (buy_decision.id,),
        ).fetchone()
    assert buy_row is not None, (
        f"original BUY row {buy_decision.id!r} missing from decisions table"
    )
    assert buy_row[1] is None, (
        f"original BUY row must keep failure_mode IS NULL (execution agent "
        f"must NOT mutate the upstream Decision); got {buy_row[1]!r}"
    )

    # --- Step 9: assertion — returned execution_result signals broker_unavailable
    exec_result = out["execution_result"]
    assert exec_result.get("skipped") is True, (
        f"execution_result must signal skipped=True; got {exec_result!r}"
    )
    assert exec_result.get("reason") == "broker_unavailable", (
        f"execution_result.reason must be 'broker_unavailable'; "
        f"got {exec_result.get('reason')!r}"
    )
    assert exec_result.get("attempts") == _MAX_ATTEMPTS, (
        f"execution_result.attempts must be {_MAX_ATTEMPTS}; "
        f"got {exec_result.get('attempts')!r}"
    )

    # --- Step 10: assertion — refuse row's parent_chain points to BUY --------
    # Parses the chain JSON to verify it points back to the executable BUY.
    # (Column is stored as ``parent_chain`` in the schema, which is the
    # serialised form of ``Decision.decision_id_chain``.)
    import json as _json
    chain = _json.loads(refuse_row["parent_chain"])
    assert chain == [buy_decision.id], (
        f"REFUSE decision_id_chain must point back to the failing BUY; "
        f"expected [{buy_decision.id!r}], got {chain!r}"
    )


# ---------------------------------------------------------------------------
# Secondary fixture — recovery within budget
# ---------------------------------------------------------------------------


def test_broker_recovers_within_retry_budget_returns_normally(
    tmp_path: Path,
) -> None:
    """Broker fails twice then succeeds: ``submit_call_count == 3``, no
    REFUSE row, outbox row reaches ``'confirmed'``, no exception bubbles.

    Pins that the retry loop's success path is *the same* as the no-fail
    happy path — recovery within budget MUST NOT emit a BROKER_UNAVAILABLE
    disposition.
    """
    db_path = tmp_path / "firm.db"
    init_db(db_path)
    clock = ReplayClock(_T0)

    buy_decision = _make_buy_decision(clock=clock)
    _persist_decisions_from_state(
        {"risk_decision": buy_decision}, db_path, clock
    )

    flaky = _FlakyBroker(fail_first_n=_MAX_ATTEMPTS - 1)
    execution = make_execution(
        db_path=db_path,
        broker=flaky,
        clock=clock,
        nonce_secret=_NONCE_SECRET,
    )

    out = execution({"risk_decision": buy_decision, "hitl_required": False})

    # Submit was tried exactly _MAX_ATTEMPTS times (fail, fail, success).
    assert flaky.submit_call_count == _MAX_ATTEMPTS, (
        f"expected {_MAX_ATTEMPTS} submit attempts (N-1 failures + 1 success); "
        f"got {flaky.submit_call_count}"
    )

    # No REFUSE BROKER_UNAVAILABLE row written.
    with closing(sqlite3.connect(str(db_path))) as conn:
        bu_rows = conn.execute(
            "SELECT id FROM decisions WHERE failure_mode = ?",
            ("broker_unavailable",),
        ).fetchall()
    assert len(bu_rows) == 0, (
        f"successful retry must NOT emit a BROKER_UNAVAILABLE disposition; "
        f"got {len(bu_rows)} rows"
    )

    # Outbox row flipped to 'confirmed'.
    expected_key = _idempotency_key(buy_decision)
    with closing(get_conn(db_path)) as conn:
        outbox_row = conn.execute(
            "SELECT status FROM outbox WHERE key = ?", (expected_key,)
        ).fetchone()
    assert outbox_row is not None
    assert outbox_row["status"] == "confirmed", (
        f"outbox row must reach 'confirmed' when broker eventually succeeds; "
        f"got {outbox_row['status']!r}"
    )

    # execution_result reflects a normal OrderResult (no skipped flag).
    exec_result = out["execution_result"]
    assert exec_result.get("skipped") is not True, (
        f"execution_result must NOT be skipped on successful recovery; "
        f"got {exec_result!r}"
    )
    assert exec_result.get("ticker") == "AAPL"


# ---------------------------------------------------------------------------
# Unit-level pin — the typed exception itself
# ---------------------------------------------------------------------------


def test_broker_unavailable_error_message_includes_attempts_and_key() -> None:
    """:class:`BrokerUnavailableError` must surface attempts + idempotency
    key + underlying cause so on-call engineers can grep the heartbeat log
    and immediately see the failed key (matches the outbox row to retry).
    """
    from firm.outbox.outbox import BrokerUnavailableError

    underlying = RuntimeError("503 Service Unavailable")
    exc = BrokerUnavailableError(
        idempotency_key="abc123",
        attempts=3,
        underlying=underlying,
    )
    msg = str(exc)
    assert "abc123" in msg
    assert "3" in msg
    assert "503 Service Unavailable" in msg
    assert exc.idempotency_key == "abc123"
    assert exc.attempts == 3
    assert exc.underlying is underlying


# Pin the integration test file is collectable by the failure-mode coverage
# registry — sanity check the function name matches the registry locator.
def test_failure_mode_registry_locator_matches() -> None:
    from tests.integration.test_failure_mode_coverage import FAILURE_MODE_FIXTURES
    from firm.core.models import FailureMode

    if FailureMode.BROKER_UNAVAILABLE not in FAILURE_MODE_FIXTURES:
        pytest.skip(
            "BROKER_UNAVAILABLE not yet migrated to FAILURE_MODE_FIXTURES; "
            "this assertion runs after Part C of T25."
        )
    locator = FAILURE_MODE_FIXTURES[FailureMode.BROKER_UNAVAILABLE]
    assert (
        locator
        == "tests/integration/test_failuremode_broker_unavailable.py"
        "::test_broker_503_emits_refuse_with_broker_unavailable_and_leaves_outbox_pending"
    )
