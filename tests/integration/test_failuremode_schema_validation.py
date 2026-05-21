"""Plan 4 T20 — SCHEMA_VALIDATION_FAILED end-to-end fixture.

Drives PM → Risk → Execution → Reporter for a malformed PM voter response
and asserts:

  (a) The REFUSE / SCHEMA_VALIDATION_FAILED Decision propagates through risk
      and execution unchanged — the graph slice passes the failing Decision
      through cleanly with no short-circuit to a different failure mode.
  (b) Execution skips with no broker ``place_order`` call — the skip branch
      at ``firm/agents/execution.py:45`` fires because ``action != BUY/SELL``.
  (c) The failing Decision lands in the ``decisions`` table as the durable
      audit row — ``failure_mode='schema_validation_failed'`` is queryable.

Complements (does not replace) the PM-unit-level fixture at
``tests/unit/test_pm_agent.py::test_pm_maps_pm_vote_schema_error_to_refuse_schema_validation_failed``;
the FAILURE_MODE_FIXTURES locator remains pointed at the unit test.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from threading import Lock
from typing import Any

import pytest

from firm.agents.execution import make_execution
from firm.agents.pm import PmVoter, make_pm
from firm.agents.reporter import make_reporter
from firm.agents.risk import RiskInput, evaluate_risk
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.config import load_policy, load_universe
from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Claim,
    Decision,
    FailureMode,
    RefusePayload,
    SellPayload,
)
from firm.db.migrations import init_db
from firm.obs import agent_span, stamp_decision
from firm.orchestrator.state import WorkingState
from tests.red_team.conftest import BrokerCall, CallLoggingBroker


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _MalformedVoterClient:
    """AnthropicMessagesClient stub that injects one malformed JSON response.

    Mirrors the pattern from
    ``tests/unit/test_pm_agent.py::_RecordingClient``.

    One of the three voters (any lens) will pop the malformed
    ``"not json at all"`` response and raise ``PmVoteSchemaError``;
    ``asyncio.gather`` then surfaces the exception.  Order is not
    deterministic because ``asyncio.to_thread`` runs on a thread pool —
    the ``malformed_emitted`` flag confirms only that the malformed entry
    WAS consumed, not which lens consumed it.

    A ``threading.Lock`` serialises ``pop(0)`` calls to make the intent
    explicit and silence any "looks racy" review noise (``list.pop`` is
    GIL-atomic for a single operation, but the explicit lock signals that
    the serialisation is deliberate).
    """

    def __init__(self) -> None:
        self._responses = [
            json.dumps(
                {
                    "vote": "BUY",
                    "confidence": 0.8,
                    "rationale": "q-ok",
                    "cited_claim_ids": ["c1"],
                }
            ),
            "not json at all",
            json.dumps(
                {
                    "vote": "BUY",
                    "confidence": 0.6,
                    "rationale": "c-ok",
                    "cited_claim_ids": [],
                }
            ),
        ]
        self._lock = Lock()
        self.calls: list[dict[str, Any]] = []
        self.malformed_emitted: bool = False

    def messages_create(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        with self._lock:
            text = self._responses.pop(0) if self._responses else self._responses[-1]
        if text == "not json at all":
            self.malformed_emitted = True
        return {"content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# Risk-node builder (mirrors firm/cli.py:329-358 verbatim)
# ---------------------------------------------------------------------------


def _build_risk_node(
    broker: CallLoggingBroker,
    policy: Any,
    universe: Any,
) -> Any:
    """Close over broker/policy/universe and return a risk node callable."""

    def risk_node(state: WorkingState) -> dict[str, Any]:
        with agent_span("risk") as span:
            proposal = state["pm_decision"]
            if not isinstance(proposal.payload, (BuyPayload, SellPayload)):
                stamp_decision(span, proposal.id, proposal.failure_mode)
                return {"risk_decision": proposal}
            ticker = proposal.payload.ticker
            quote = broker.get_quote(ticker)
            positions = {p.ticker: p.shares for p in broker.list_positions()}
            decision = evaluate_risk(
                RiskInput(
                    proposal=proposal,
                    quote_price=quote.price,
                    quote_age_seconds=0,
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


def test_graph_propagates_refuse_schema_validation_failed_and_writes_no_broker_call(
    tmp_path: Path,
) -> None:
    """PM voter malformed response → REFUSE / SCHEMA_VALIDATION_FAILED propagates
    through Risk and Execution, zero broker place_order calls, audit row written.
    """
    # --- infrastructure setup -----------------------------------------------
    db_path = tmp_path / "firm.db"
    init_db(db_path)

    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    inner_broker = FakeBroker(initial_cash=Decimal("100000"))
    broker = CallLoggingBroker(inner_broker)

    policy = load_policy(Path("config/policy.yaml"))
    universe = load_universe(Path("config/universe.yaml"))

    # --- PM node setup -------------------------------------------------------
    malformed_client = _MalformedVoterClient()
    voter = PmVoter(client=malformed_client, model="claude-sonnet-4-6")  # type: ignore[arg-type]  # structurally compatible stub
    pm_node = make_pm(voter=voter)

    # Research-stage BUY Decision supplied to PM
    research_decision = Decision(
        id="t20-research-1",
        decision_id_chain=[],
        action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="AAPL strong FCF; margin expansion expected.",
        confidence=0.7,
        citations=[],
        falsification_condition="AAPL FCF falls below $50B in FY2025.",
        escalation_reason=None,
        failure_mode=None,
        metadata={"agent": "research", "ticker": "AAPL"},
        nonce="t20-nonce",
    )

    valid_claim = Claim(
        text="AAPL generated $90B in free cash flow in FY2023.",
        source_chunk_id="chunk-t20-0",
    )
    claims_dump = [valid_claim.model_dump()]

    state: WorkingState = {
        "research_decision": research_decision,
        "claims": claims_dump,
    }

    # --- Step 1: PM node -----------------------------------------------------
    pm_out = pm_node(state)

    pm_decision: Decision = pm_out["pm_decision"]
    assert pm_decision.action == ActionEnum.REFUSE, (
        f"expected REFUSE, got {pm_decision.action}"
    )
    assert pm_decision.failure_mode == FailureMode.SCHEMA_VALIDATION_FAILED, (
        f"expected SCHEMA_VALIDATION_FAILED, got {pm_decision.failure_mode}"
    )
    assert isinstance(pm_decision.payload, RefusePayload), (
        f"expected RefusePayload, got {type(pm_decision.payload)}"
    )

    # --- Step 2: Risk node ---------------------------------------------------
    risk_node = _build_risk_node(broker, policy, universe)
    risk_out = risk_node(state | pm_out)

    risk_decision: Decision = risk_out["risk_decision"]
    # The risk node short-circuits on non-BUY/SELL payloads (RefusePayload),
    # passing the PM Decision through unchanged.
    assert risk_decision.failure_mode == FailureMode.SCHEMA_VALIDATION_FAILED, (
        f"risk node must propagate SCHEMA_VALIDATION_FAILED, got {risk_decision.failure_mode}"
    )
    assert risk_decision.id == pm_decision.id, (
        "risk node must pass REFUSE through unchanged (same Decision object)"
    )
    assert risk_decision.action == ActionEnum.REFUSE

    # --- Step 3: Execution node ----------------------------------------------
    execution_node = make_execution(db_path=db_path, broker=broker, clock=clock)
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

    # --- Step 4: Reporter node -----------------------------------------------
    reporter_node = make_reporter(
        reports_root=tmp_path / "reports",
        clock=clock,
        db_path=db_path,
    )
    reporter_node(
        {"pm_decision": pm_out["pm_decision"], "risk_decision": risk_out["risk_decision"]}
    )

    # --- Final assertions ----------------------------------------------------

    # (b) Zero place_order broker calls
    place_order_calls = [c for c in broker.calls if c.method == "place_order"]
    assert len(place_order_calls) == 0, (
        f"expected no place_order calls, got {place_order_calls}"
    )

    # (c) Audit row written to decisions table
    with closing(sqlite3.connect(str(db_path))) as conn:
        rows = conn.execute(
            "SELECT failure_mode FROM decisions WHERE failure_mode = 'schema_validation_failed'"
        ).fetchall()
    assert len(rows) >= 1, (
        "decisions table must contain at least one row with "
        "failure_mode='schema_validation_failed'"
    )

    # Optional: confirm the malformed response was actually emitted
    assert malformed_client.malformed_emitted, (
        "the stub must have emitted 'not json at all' to confirm the schema-error "
        "branch was exercised (not some other REFUSE path)"
    )
