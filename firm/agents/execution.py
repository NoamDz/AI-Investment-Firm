"""Execution agent — wraps the outbox. See spec §3, §5.2."""
from __future__ import annotations

from pathlib import Path

from firm.broker.protocol import Broker
from firm.core.clock import Clock
from firm.core.models import ActionEnum, Decision
from firm.outbox.outbox import place_order_via_outbox


def make_execution(*, db_path: Path, broker: Broker, clock: Clock):
    def execution(state: dict) -> dict:
        risk: Decision = state["risk_decision"]
        if risk.action not in (ActionEnum.BUY, ActionEnum.SELL):
            return {"execution_result": {"skipped": True, "reason": f"action={risk.action.value}"}}
        if state.get("hitl_required") and not state.get("hitl_approved"):
            return {"execution_result": {"skipped": True, "reason": "hitl_not_approved"}}

        result = place_order_via_outbox(risk, db_path, broker, clock)
        return {"execution_result": result.model_dump(mode="json")}
    return execution
