"""Execution agent — wraps the outbox. See spec §3, §5.2."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from firm.agents.reporter import _persist_decisions_from_state
from firm.broker.protocol import Broker
from firm.core.clock import Clock
from firm.core.models import ActionEnum, Decision
from firm.orchestrator.state import WorkingState
from firm.outbox.outbox import place_order_via_outbox


def make_execution(
    *, db_path: Path, broker: Broker, clock: Clock
) -> Callable[[WorkingState], dict[str, Any]]:
    def execution(state: WorkingState) -> dict[str, Any]:
        risk: Decision | None = state.get("risk_decision")
        if risk is None:
            return {"execution_result": {"skipped": True, "reason": "no_risk_decision"}}
        if risk.action not in (ActionEnum.BUY, ActionEnum.SELL):
            return {"execution_result": {"skipped": True, "reason": f"action={risk.action.value}"}}
        if state.get("hitl_required") and not state.get("hitl_approved"):
            return {"execution_result": {"skipped": True, "reason": "hitl_not_approved"}}

        # Persist the risk decision before writing to outbox (outbox has FK → decisions).
        _persist_decisions_from_state({"risk_decision": risk}, db_path, clock)

        result = place_order_via_outbox(risk, db_path, broker, clock)
        return {"execution_result": result.model_dump(mode="json")}
    return execution
