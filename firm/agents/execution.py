"""Execution agent — wraps the outbox. See spec §3, §5.2."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from firm.agents.reporter import _persist_decisions_from_state
from firm.broker.protocol import Broker
from firm.core.clock import Clock
from firm.core.ids import sign_nonce, ulid_new
from firm.core.models import (
    ActionEnum,
    Decision,
    EscalatePayload,
    FailureMode,
    RefusePayload,
)
from firm.orchestrator.state import WorkingState
from firm.outbox.outbox import BrokerUnavailableError, place_order_via_outbox


def _build_broker_unavailable_refuse(
    *,
    executable: Decision,
    exc: BrokerUnavailableError,
    clock: Clock,
    nonce_secret: bytes,
) -> Decision:
    """Construct a REFUSE Decision stamped with BROKER_UNAVAILABLE.

    ``decision_id_chain`` points back to the ``executable`` risk decision
    whose outbox submit exhausted retries — the audit trail must let an
    operator follow the chain back to the unfilled order so the EOD
    report (spec §10.5) can surface the same idempotency key.
    """
    now = clock.now()
    refuse_id = ulid_new()
    nonce = sign_nonce(
        nonce_secret,
        decision_id=refuse_id,
        timestamp=int(now.timestamp()),
    )
    return Decision(
        id=refuse_id,
        decision_id_chain=[executable.id],
        action=ActionEnum.REFUSE,
        payload=RefusePayload(reason="broker:unavailable"),
        rationale=(
            f"broker.submit failed after {exc.attempts} attempts; outbox row "
            f"{exc.idempotency_key} remains pending for next-heartbeat retry"
        ),
        confidence=0.0,
        citations=[],
        falsification_condition="broker availability is restored",
        escalation_reason=None,
        failure_mode=FailureMode.BROKER_UNAVAILABLE,
        metadata={
            "agent": "execution",
            "idempotency_key": exc.idempotency_key,
            "attempts": exc.attempts,
        },
        nonce=nonce,
    )


def make_execution(
    *, db_path: Path, broker: Broker, clock: Clock, nonce_secret: bytes
) -> Callable[[WorkingState], dict[str, Any]]:
    """Build the execution-node callable.

    ``nonce_secret`` is required so the agent can sign the REFUSE Decision
    it emits when :class:`BrokerUnavailableError` is raised by the outbox
    (spec §3.4 — every Decision row carries an HMAC nonce).  Callers in
    test scaffolding can pass any non-empty 32-byte string; production
    sources it from ``FIRM_HMAC_SECRET`` via :mod:`firm.cli`.
    """

    def execution(state: WorkingState) -> dict[str, Any]:
        risk: Decision | None = state.get("risk_decision")
        if risk is None:
            return {"execution_result": {"skipped": True, "reason": "no_risk_decision"}}

        # ESCALATE that has been HITL-approved executes the proposed inner trade.
        if risk.action == ActionEnum.ESCALATE and state.get("hitl_approved"):
            assert isinstance(risk.payload, EscalatePayload)
            # Unwrap: replace the escalate decision's payload with the proposed trade for outbox.
            executable = Decision(
                id=risk.id,
                decision_id_chain=risk.decision_id_chain,
                action=ActionEnum(risk.payload.proposed.kind.upper()),  # "buy"→BUY, "sell"→SELL
                payload=risk.payload.proposed,
                rationale=risk.rationale,
                confidence=risk.confidence,
                citations=risk.citations,
                falsification_condition=risk.falsification_condition,
                escalation_reason=risk.escalation_reason,
                failure_mode=risk.failure_mode,
                metadata=risk.metadata,
                nonce=risk.nonce,
            )
            _persist_decisions_from_state({"risk_decision": risk}, db_path, clock)
            try:
                result = place_order_via_outbox(executable, db_path, broker, clock)
            except BrokerUnavailableError as exc:
                refuse = _build_broker_unavailable_refuse(
                    executable=executable,
                    exc=exc,
                    clock=clock,
                    nonce_secret=nonce_secret,
                )
                _persist_decisions_from_state(
                    {"risk_decision": refuse}, db_path, clock
                )
                return {
                    "execution_result": {
                        "skipped": True,
                        "reason": "broker_unavailable",
                        "attempts": exc.attempts,
                    }
                }
            return {"execution_result": result.model_dump(mode="json")}

        if risk.action not in (ActionEnum.BUY, ActionEnum.SELL):
            return {"execution_result": {"skipped": True, "reason": f"action={risk.action.value}"}}
        if state.get("hitl_required") and not state.get("hitl_approved"):
            return {"execution_result": {"skipped": True, "reason": "hitl_not_approved"}}

        # Persist the risk decision before writing to outbox (outbox has FK → decisions).
        _persist_decisions_from_state({"risk_decision": risk}, db_path, clock)

        try:
            result = place_order_via_outbox(risk, db_path, broker, clock)
        except BrokerUnavailableError as exc:
            refuse = _build_broker_unavailable_refuse(
                executable=risk,
                exc=exc,
                clock=clock,
                nonce_secret=nonce_secret,
            )
            _persist_decisions_from_state(
                {"risk_decision": refuse}, db_path, clock
            )
            return {
                "execution_result": {
                    "skipped": True,
                    "reason": "broker_unavailable",
                    "attempts": exc.attempts,
                }
            }
        return {"execution_result": result.model_dump(mode="json")}
    return execution
