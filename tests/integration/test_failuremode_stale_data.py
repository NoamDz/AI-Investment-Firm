"""Plan 4 T21 — STALE_DATA end-to-end fixture.

Drives Risk → Execution → Reporter for a stale-quote scenario and asserts:

  (a) The risk node detects a quote whose timestamp is 90 s in the past
      (over the policy's 60 s staleness cap) and emits REFUSE / STALE_DATA.
  (b) Execution skips with no broker ``place_order`` call — the skip branch
      fires because ``action != BUY/SELL``.
  (c) The failing Decision lands in the ``decisions`` table as the durable
      audit row — ``failure_mode='stale_data'`` is queryable.

Complements (does not replace)
``tests/unit/test_risk_limits.py::test_blocks_stale_quote``; the
FAILURE_MODE_FIXTURES locator remains pointed at the unit test.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


from firm.agents.execution import make_execution
from firm.agents.reporter import make_reporter
from firm.agents.risk import RiskInput, evaluate_risk
from firm.broker.fake_broker import FakeBroker
from firm.broker.protocol import Quote
from firm.core.clock import ReplayClock
from firm.core.config import load_policy, load_universe
from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Decision,
    FailureMode,
)
from firm.db.migrations import init_db
from firm.obs import agent_span, stamp_decision
from firm.orchestrator.state import WorkingState
from tests.red_team.conftest import BrokerCall, CallLoggingBroker


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

_T0 = datetime(2024, 3, 13, tzinfo=timezone.utc)
_STALE_TIMESTAMP = (_T0 - timedelta(seconds=90)).isoformat()


class _StaleQuoteBroker(CallLoggingBroker):
    """Wraps CallLoggingBroker and overrides get_quote to return a stale Quote.

    The overridden ``get_quote`` returns a Quote whose ``timestamp`` is 90 s
    before T+0 — exceeding the policy's 60 s staleness cap.  All other
    methods delegate to ``CallLoggingBroker`` (which itself delegates to
    ``FakeBroker``), so call logging for ``place_order`` still works.
    """

    def get_quote(self, ticker: str) -> Quote:
        # Record the call first (mirrors CallLoggingBroker's contract).
        self.calls.append(BrokerCall(method="get_quote", args=(ticker,)))
        inner_quote = self._inner.get_quote(ticker)
        # Return a Quote with the same price but a stale timestamp.
        return Quote(
            ticker=inner_quote.ticker,
            price=inner_quote.price,
            timestamp=_STALE_TIMESTAMP,
        )


# ---------------------------------------------------------------------------
# Risk-node builder (mirrors firm/cli.py:329-358)
# ---------------------------------------------------------------------------


def _build_risk_node(
    broker: _StaleQuoteBroker,
    policy: Any,
    universe: Any,
    clock: ReplayClock,
) -> Any:
    """Close over broker/policy/universe/clock and return a risk node callable.

    Mirrors ``firm/cli.py:329-358`` verbatim with one local difference:
    ``quote_age_seconds`` is derived from the quote's timestamp instead of
    the production hard-coded 0, enabling the STALE_DATA branch to fire.
    """

    def risk_node(state: WorkingState) -> dict[str, Any]:
        with agent_span("risk") as span:
            proposal = state["pm_decision"]
            if not isinstance(proposal.payload, BuyPayload):
                stamp_decision(span, proposal.id, proposal.failure_mode)
                return {"risk_decision": proposal}
            ticker = proposal.payload.ticker
            quote = broker.get_quote(ticker)
            positions = {p.ticker: p.shares for p in broker.list_positions()}
            # Plan 4 T21 fixture: derive quote staleness from the quote's timestamp
            # instead of the production TODO at firm/cli.py:347's hard-coded 0;
            # the test's whole purpose is to exercise the STALE_DATA branch.
            quote_age_seconds = int(
                (clock.now() - datetime.fromisoformat(quote.timestamp)).total_seconds()
            )
            decision = evaluate_risk(
                RiskInput(
                    proposal=proposal,
                    quote_price=quote.price,
                    quote_age_seconds=quote_age_seconds,
                    cash=broker.get_cash(),
                    positions=positions,
                    sector_map=universe.sector_map,
                    trades_today=0,
                    nav=broker.get_cash()
                    + sum(
                        (
                            p.shares * broker.get_quote(p.ticker).price
                            for p in broker.list_positions()
                        ),
                        Decimal("0"),
                    ),
                    daily_pnl_pct=0.0,
                    policy=policy,
                )
            )
            stamp_decision(span, decision.id, decision.failure_mode)
            return {"risk_decision": decision}

    return risk_node


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_graph_propagates_refuse_stale_data_and_writes_no_broker_call(
    tmp_path: Path,
) -> None:
    """Stale quote (90 s > 60 s cap) → REFUSE / STALE_DATA propagates through
    Execution with no broker place_order call; audit row written to decisions table.
    """
    # --- infrastructure setup -----------------------------------------------
    db_path = tmp_path / "firm.db"
    init_db(db_path)

    clock = ReplayClock(_T0)
    inner_broker = FakeBroker(initial_cash=Decimal("100000"))
    stale_broker = _StaleQuoteBroker(inner_broker)

    policy = load_policy(Path("config/policy.yaml"))
    universe = load_universe(Path("config/universe.yaml"))

    # --- Clean PM Decision (synthetic — this test is about Risk → Exec → Reporter) ---
    clean_pm_decision = Decision(
        id="t21-pm-1",
        decision_id_chain=[],
        action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="AAPL strong FCF; integration fixture for STALE_DATA.",
        confidence=0.7,
        citations=[],
        falsification_condition="AAPL FCF falls below $50B in FY2025.",
        escalation_reason=None,
        failure_mode=None,
        metadata={},
        nonce="t21-nonce",
    )

    # --- Step 1: Risk node ---------------------------------------------------
    risk_node = _build_risk_node(stale_broker, policy, universe, clock)
    risk_out = risk_node({"pm_decision": clean_pm_decision})

    risk_decision: Decision = risk_out["risk_decision"]
    assert risk_decision.failure_mode == FailureMode.STALE_DATA, (
        f"expected STALE_DATA, got {risk_decision.failure_mode}"
    )
    assert risk_decision.action == ActionEnum.REFUSE, (
        f"expected REFUSE, got {risk_decision.action}"
    )

    # --- Step 2: Execution node ----------------------------------------------
    execution_node = make_execution(
        db_path=db_path, broker=stale_broker, clock=clock, nonce_secret=b"x" * 32
    )
    exe_out = execution_node(
        {"risk_decision": risk_out["risk_decision"], "hitl_required": False}
    )

    exec_result = exe_out["execution_result"]
    assert exec_result.get("skipped") is True, (
        f"execution must skip for REFUSE action, got {exec_result}"
    )
    reason = exec_result.get("reason", "")
    assert "refuse" in reason.lower(), (
        f"skip reason must mention 'refuse', got {reason!r}"
    )

    # --- Step 3: Reporter node -----------------------------------------------
    reporter_node = make_reporter(
        reports_root=tmp_path / "reports",
        clock=clock,
        db_path=db_path,
    )
    reporter_node({"risk_decision": risk_out["risk_decision"]})

    # --- Step 4: DB assertion — audit row written ----------------------------
    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = conn.execute(
            "SELECT failure_mode FROM decisions WHERE failure_mode = ?",
            ("stale_data",),
        ).fetchall()
    assert len(rows) >= 1, (
        "decisions table must contain at least one row with "
        "failure_mode='stale_data'"
    )

    # --- Step 5: Zero place_order broker calls -------------------------------
    place_order_calls = [c for c in stale_broker.calls if c.method == "place_order"]
    assert len(place_order_calls) == 0, (
        f"expected no place_order calls, got {place_order_calls}"
    )
