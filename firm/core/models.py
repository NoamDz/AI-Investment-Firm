"""Core typed contracts emitted by every agent. See design spec §3.4, §3.5, §7.2."""
from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


class ActionEnum(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    ESCALATE = "ESCALATE"
    REFUSE = "REFUSE"


class FailureMode(StrEnum):
    UNCITED_CLAIM = "uncited_claim"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    HITL_TIMEOUT = "hitl_timeout"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    LLM_UNAVAILABLE = "llm_unavailable"
    STALE_DATA = "stale_data"
    UNGROUNDED_CLAIM = "ungrounded_claim"
    TOOL_PERMISSION_DENIED = "tool_permission_denied"
    UNAPPROVED_HIGH_RISK = "unapproved_high_risk"
    BROKER_UNAVAILABLE = "broker_unavailable"
    UNKNOWN = "unknown"


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    span: tuple[int, int]
    cited_text: str | None = None
    document_index: int | None = None
    document_title: str | None = None

    def __init__(
        self,
        source_id: str,
        chunk_id: str,
        span: tuple[int, int],
        *,
        cited_text: str | None = None,
        document_index: int | None = None,
        document_title: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            source_id=source_id,
            chunk_id=chunk_id,
            span=span,
            cited_text=cited_text,
            document_index=document_index,
            document_title=document_title,
            **kwargs,
        )


class Claim(BaseModel):
    text: str
    value: Decimal | None = None
    unit: str | None = None
    source_chunk_id: str | None = None
    source_span: tuple[int, int] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _numeric_claim_requires_provenance(self) -> "Claim":
        if self.value is not None and self.source_chunk_id is None and self.tool_call_id is None:
            raise ValueError("numeric Claim requires source_chunk_id or tool_call_id")
        return self


class BuyPayload(BaseModel):
    kind: Literal["buy"] = "buy"
    ticker: str
    shares: Decimal
    limit_price: Decimal | None = None


class SellPayload(BaseModel):
    kind: Literal["sell"] = "sell"
    ticker: str
    shares: Decimal
    limit_price: Decimal | None = None


class HoldPayload(BaseModel):
    kind: Literal["hold"] = "hold"
    reason: str


class EscalatePayload(BaseModel):
    kind: Literal["escalate"] = "escalate"
    proposed: Annotated[
        Union[BuyPayload, SellPayload],
        Field(discriminator="kind"),
    ]
    reason: str


class RefusePayload(BaseModel):
    kind: Literal["refuse"] = "refuse"
    reason: str


TypedPayload = Annotated[
    Union[BuyPayload, SellPayload, HoldPayload, EscalatePayload, RefusePayload],
    Field(discriminator="kind"),
]


class Decision(BaseModel):
    id: str
    decision_id_chain: list[str] = Field(default_factory=list)
    action: ActionEnum
    payload: TypedPayload
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    falsification_condition: str = Field(min_length=1)
    escalation_reason: str | None = None
    failure_mode: FailureMode | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    nonce: str = Field(min_length=1)
