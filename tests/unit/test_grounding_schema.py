from __future__ import annotations

from firm.grounding.schema import ClaimAssessment, ClaimSupport, SufficiencyResult


def test_claim_support_enum_values():
    assert ClaimSupport.SUPPORTED.value == "SUPPORTED"
    assert ClaimSupport.PARTIAL.value == "PARTIAL"
    assert ClaimSupport.UNSUPPORTED.value == "UNSUPPORTED"


def test_sufficiency_result_aggregation_status():
    def make(support: ClaimSupport) -> ClaimAssessment:
        return ClaimAssessment(claim_id="c1", support=support, reasoning="reason")

    all_supported = SufficiencyResult(
        claim_assessments=[make(ClaimSupport.SUPPORTED), make(ClaimSupport.SUPPORTED)],
    )
    assert all_supported.aggregate_status() == "ok"

    mixed_partial = SufficiencyResult(
        claim_assessments=[make(ClaimSupport.SUPPORTED), make(ClaimSupport.PARTIAL)],
    )
    assert mixed_partial.aggregate_status() == "partial"

    with_unsupported = SufficiencyResult(
        claim_assessments=[make(ClaimSupport.SUPPORTED), make(ClaimSupport.UNSUPPORTED)],
    )
    assert with_unsupported.aggregate_status() == "insufficient"

    empty = SufficiencyResult(claim_assessments=[])
    assert empty.aggregate_status() == "ok"
