"""Grounding schema: ClaimSupport enum and SufficiencyResult. See design spec §7."""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ClaimSupport(StrEnum):
    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


class ClaimAssessment(BaseModel):
    claim_id: str
    support: ClaimSupport
    reasoning: str = Field(min_length=1)


class SufficiencyResult(BaseModel):
    claim_assessments: list[ClaimAssessment]
    overall_reasoning: str = ""

    def aggregate_status(self) -> str:
        supports = [a.support for a in self.claim_assessments]
        if any(s == ClaimSupport.UNSUPPORTED for s in supports):
            return "insufficient"
        if any(s == ClaimSupport.PARTIAL for s in supports):
            return "partial"
        return "ok"
