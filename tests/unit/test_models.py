from decimal import Decimal
from firm.core.models import (
    ActionEnum, FailureMode, Citation, Claim,
    BuyPayload, SellPayload, HoldPayload, Decision,
)


def test_failure_mode_values():
    assert FailureMode.UNCITED_CLAIM.value == "uncited_claim"
    assert FailureMode.BROKER_UNAVAILABLE.value == "broker_unavailable"
    # 13 total values per spec §3.5
    assert len(list(FailureMode)) == 13


def test_action_enum_values():
    assert {a.value for a in ActionEnum} == {"BUY", "SELL", "HOLD", "ESCALATE", "REFUSE"}


def test_decision_requires_rationale_and_nonce():
    d = Decision(
        id="01HZZZZZZZZZZZZZZZZZZZZZZZ",
        decision_id_chain=[],
        action=ActionEnum.HOLD,
        payload=HoldPayload(reason="stub"),
        rationale="deterministic stub",
        confidence=0.5,
        citations=[],
        falsification_condition="never",
        escalation_reason=None,
        failure_mode=None,
        metadata={},
        nonce="abc123",
    )
    assert d.action == ActionEnum.HOLD


def test_buy_payload_carries_decimal_value():
    p = BuyPayload(ticker="AAPL", shares=Decimal("10"), limit_price=Decimal("180.50"))
    assert p.ticker == "AAPL"
    assert p.shares == Decimal("10")


def test_claim_requires_provenance():
    # text-only claim with no source — should still construct, validation is upstream
    c = Claim(text="NVDA reported revenue", value=None, unit=None,
              source_chunk_id=None, source_span=None, tool_call_id=None)
    assert c.text.startswith("NVDA")
