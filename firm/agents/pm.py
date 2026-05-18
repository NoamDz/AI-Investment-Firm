"""PM agent — deterministic pass-through stub for Plan 1.

Plan 2 swaps this for vote-of-3 self-consistency over LLM rationales.
"""
from __future__ import annotations

from firm.core.ids import ulid_new
from firm.core.models import Decision


def make_pm():
    def pm(state: dict) -> dict:
        research: Decision = state["research_decision"]
        decision = Decision(
            id=ulid_new(), decision_id_chain=[research.id],
            action=research.action, payload=research.payload,
            rationale=f"pm pass-through: {research.rationale}",
            confidence=research.confidence, citations=research.citations,
            falsification_condition=research.falsification_condition,
            escalation_reason=None, failure_mode=None,
            metadata={"agent": "pm", "stub": True}, nonce="pm-stub",
        )
        return {"pm_decision": decision}
    return pm
