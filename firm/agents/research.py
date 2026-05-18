"""Research agent — deterministic stub for Plan 1.

Plan 2 swaps this for an LLM-backed agent with hybrid retrieval + Citations API.
The function signature stays stable across plans.
"""
from __future__ import annotations

from decimal import Decimal

from firm.broker.protocol import Broker
from firm.core.clock import Clock
from firm.core.config import UniverseConfig
from firm.core.ids import ulid_new
from firm.core.models import ActionEnum, BuyPayload, Decision


def make_research(*, clock: Clock, broker: Broker, universe: UniverseConfig):
    def research(state: dict) -> dict:
        # Deterministic ticker selection: cheapest in universe by FakeBroker's price function
        prices = {t: broker.get_quote(t).price for t in universe.tickers}
        chosen = min(prices, key=lambda t: prices[t])
        decision = Decision(
            id=ulid_new(), decision_id_chain=[], action=ActionEnum.BUY,
            payload=BuyPayload(ticker=chosen, shares=Decimal("10")),
            rationale=f"deterministic stub: cheapest of universe at heartbeat {state.get('heartbeat_at')}",
            confidence=0.5, citations=[],
            falsification_condition=f"if {chosen} drops more than 5% by EOD",
            escalation_reason=None, failure_mode=None,
            metadata={"agent": "research", "stub": True}, nonce="research-stub",
        )
        return {"research_decision": decision}
    return research
