"""Core typed contracts emitted by every agent. See design spec §3.4, §3.5, §7.2."""
from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Literal, Union

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
    RECONCILIATION_DRIFT = "reconciliation_drift"
    SIGNED_APPROVAL_INVALID = "signed_approval_invalid"
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
    # Verbatim ``cited_text`` from the Anthropic Citations API (the actual
    # source-document substring), as opposed to ``text`` which is the LLM's
    # block-level assertion built around it.  Populated by
    # AnthropicCitationsExtractor; ``None`` for tool-derived Claims or older
    # records that predate this field.
    source_quote: str | None = None
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


# ---------------------------------------------------------------------------
# Router features (Plan 3 T05)
# ---------------------------------------------------------------------------
#
# Per-request features the CostRouter uses to pick a model profile.  Values are
# normalized scalars in ``[0.0, 1.0]``; the upstream caller (Research / PM) is
# responsible for projecting raw inputs (e.g. portfolio %-of-NAV, novelty
# score, prompt-token estimate, SLA deadline distance) into this space.
#
# ``score()`` takes the per-feature *weights* (one float per feature name)
# supplied by ``config/router.yaml`` (T06), computes the weighted-sum of
# features normalized by the weight total, and maps the result to one of the
# three model profiles ``haiku`` / ``sonnet`` / ``opus``.  The thresholds are
# fixed here rather than in config because they encode the contract with the
# profile names themselves (see Plan 3 §4 "Cost-aware routing").

# Normalized-score cutoffs for profile selection.  ``score < _HAIKU_MAX`` →
# haiku; ``_HAIKU_MAX <= score < _OPUS_MIN`` → sonnet; ``score >= _OPUS_MIN``
# → opus.
_HAIKU_MAX = 0.33   # normalized score strictly below this → haiku
_OPUS_MIN = 0.66    # normalized score at or above this → opus

# Profile-name alias.  T06's router-config loader imports this to validate
# the ``profiles:`` keys in ``config/router.yaml``.  Kept as a ``Literal``
# (not a ``StrEnum``) — lighter, no runtime cost, and matches how other
# config-driven name unions are typed in this codebase.
ProfileName = Literal["haiku", "sonnet", "opus"]


class RouterFeatures(BaseModel):
    """Per-request features used by the CostRouter to pick a model profile."""

    risk_weight: float = Field(ge=0.0, le=1.0)
    novelty: float = Field(ge=0.0, le=1.0)
    complexity: float = Field(ge=0.0, le=1.0)
    time_pressure: float = Field(ge=0.0, le=1.0)

    _FEATURE_NAMES: ClassVar[tuple[str, ...]] = (
        "risk_weight",
        "novelty",
        "complexity",
        "time_pressure",
    )

    def score(self, weights: dict[str, float]) -> ProfileName:
        """Return the model profile name (``"haiku"`` / ``"sonnet"`` / ``"opus"``).

        Computes a weighted sum of the four features using ``weights`` and
        normalizes by the sum of weights so the result lies in ``[0, 1]``
        independent of weight magnitudes.  Buckets:

        * ``s < _HAIKU_MAX`` → ``"haiku"`` (cheap, low-stakes)
        * ``_HAIKU_MAX ≤ s < _OPUS_MIN`` → ``"sonnet"`` (standard)
        * ``s ≥ _OPUS_MIN`` → ``"opus"`` (high-stakes)

        Raises ``ValueError`` if ``weights`` contains an unknown key, if any
        feature key is missing from ``weights``, if any weight is negative,
        or if all weights sum to zero — all four indicate a router-config
        bug and should fail loudly rather than be masked by a silent default.
        """
        # Reject unknown keys up front: a typo in router.yaml should not be
        # silently dropped.
        extra = set(weights) - set(self._FEATURE_NAMES)
        if extra:
            raise ValueError(f"unknown router weight key(s): {sorted(extra)}")

        # Validate weight shape & sign so callers see config errors, not
        # arithmetic ones.
        weighted_sum = 0.0
        weight_total = 0.0
        for name in self._FEATURE_NAMES:
            if name not in weights:
                raise ValueError(f"missing router weight for: {name}")
            w = float(weights[name])
            if w < 0.0:
                raise ValueError(f"negative router weight for {name}: {w}")
            weighted_sum += w * float(getattr(self, name))
            weight_total += w

        if weight_total == 0.0:
            raise ValueError("router weights sum to zero")

        s = weighted_sum / weight_total
        if s < _HAIKU_MAX:
            return "haiku"
        if s < _OPUS_MIN:
            return "sonnet"
        return "opus"
