from decimal import Decimal
from firm.agents.pm import make_pm
from firm.core.models import ActionEnum, BuyPayload, Decision


def _research_decision() -> Decision:
    return Decision(
        id="res-1", decision_id_chain=[], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="stub", confidence=0.5, citations=[],
        falsification_condition="x", escalation_reason=None,
        failure_mode=None, metadata={}, nonce="n",
    )


def test_pm_passes_through_with_pm_provenance():
    pm = make_pm()
    state = {"research_decision": _research_decision()}
    out = pm(state)
    d = out["pm_decision"]
    assert d.action == ActionEnum.BUY
    assert "res-1" in d.decision_id_chain
    assert d.metadata["agent"] == "pm"
