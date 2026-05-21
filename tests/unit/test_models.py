from decimal import Decimal
import pytest
from pydantic import ValidationError
from firm.core.models import (
    ActionEnum, Citation, FailureMode, Claim,
    BuyPayload, HoldPayload, Decision,
)


def test_failure_mode_values():
    assert FailureMode.UNCITED_CLAIM.value == "uncited_claim"
    assert FailureMode.BROKER_UNAVAILABLE.value == "broker_unavailable"
    # 15 total values: 13 original + RECONCILIATION_DRIFT + SIGNED_APPROVAL_INVALID (T25)
    assert len(list(FailureMode)) == 15


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


# --- Citation new fields ---

def test_citation_accepts_anthropic_fields():
    c = Citation(
        source_id="aapl-10k",
        chunk_id="aapl-10k::0001",
        span=(0, 50),
        cited_text="Revenue grew 8% year-over-year.",
        document_index=0,
        document_title="AAPL 10-K",
    )
    assert c.cited_text == "Revenue grew 8% year-over-year."
    assert c.document_index == 0
    assert c.document_title == "AAPL 10-K"


def test_citation_back_compat_positional():
    c = Citation("doc1", "doc1::0001", (0, 10))
    assert c.source_id == "doc1"
    assert c.chunk_id == "doc1::0001"
    assert c.span == (0, 10)
    assert c.cited_text is None
    assert c.document_index is None
    assert c.document_title is None


# --- Claim validator ---

def test_claim_rejects_numeric_without_provenance():
    with pytest.raises(ValidationError):
        Claim(text="rev grew 12%", value=Decimal("0.12"))


def test_claim_accepts_chunk_provenance():
    c = Claim(text="rev grew 12%", value=Decimal("0.12"), source_chunk_id="aapl-10k::0003")
    assert c.source_chunk_id == "aapl-10k::0003"


def test_claim_accepts_tool_provenance():
    c = Claim(text="P/E was 28", value=Decimal("28.0"), tool_call_id="toolu_01abc")
    assert c.tool_call_id == "toolu_01abc"


def test_claim_textonly_without_value_allowed_without_provenance():
    c = Claim(text="management is confident about growth")
    assert c.value is None
    assert c.source_chunk_id is None
    assert c.tool_call_id is None
