"""Tests for ``firm.eval.process_metrics`` (Plan 4 §T12, spec §9.5).

One test per metric (pass/fail/edge paths), plus a smoke test that
``compute_all_metrics`` returns the 10 metrics in §9.5 order, plus an
immutability test on :class:`MetricResult`.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from firm.core.models import (
    ActionEnum,
    BuyPayload,
    Citation,
    Claim,
    Decision,
    FailureMode,
)
from firm.eval.process_metrics import (
    ClosedTrade,
    HitlPair,
    MetricResult,
    ProcessMetricsInput,
    compute_all_metrics,
    compute_citation_diversity,
    compute_decision_discipline,
    compute_failure_mode_coverage,
    compute_groundedness,
    compute_hitl_correctness,
    compute_red_team_pass,
    compute_reversal_rate,
    compute_risk_policy_compliance,
    compute_schema_rejections,
    compute_sufficiency_gate,
)


# ---------------------------------------------------------------------------
# Decision builder — minimal valid Decision, overridable per-test.
# ---------------------------------------------------------------------------
def _mk_decision(
    *,
    decision_id: str = "d1",
    citations: list[Citation] | None = None,
    rationale: str = "rationale text",
    falsification_condition: str = "fc text",
) -> Decision:
    return Decision(
        id=decision_id,
        decision_id_chain=[],
        action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("1")),
        rationale=rationale,
        confidence=0.5,
        citations=citations or [],
        falsification_condition=falsification_condition,
        escalation_reason=None,
        failure_mode=None,
        metadata={},
        nonce="nonce-1",
    )


def _cite(source_id: str, chunk_id: str = "c0") -> Citation:
    return Citation(source_id=source_id, chunk_id=chunk_id, span=(0, 1))


# ---------------------------------------------------------------------------
# 1. Groundedness
# ---------------------------------------------------------------------------


def test_groundedness_pass_fail_and_empty() -> None:
    # 2/3 grounded → 66.7%, fails 99.0 threshold.
    claims_fail = [
        Claim(text="x", source_chunk_id="c1"),
        Claim(text="y", tool_call_id="t1"),
        Claim(text="z"),
    ]
    r_fail = compute_groundedness(claims_fail)
    assert r_fail.name == "groundedness"
    assert r_fail.value == 66.7
    assert r_fail.threshold == 99.0
    assert r_fail.status == "fail"

    # All grounded → 100.0%, passes.
    claims_pass = [
        Claim(text="a", source_chunk_id="c1"),
        Claim(text="b", source_chunk_id="c2"),
    ]
    r_pass = compute_groundedness(claims_pass)
    assert r_pass.value == 100.0
    assert r_pass.status == "pass"

    # Empty → info, value 100.0.
    r_empty = compute_groundedness([])
    assert r_empty.value == 100.0
    assert r_empty.status == "info"
    assert r_empty.threshold == 99.0


# ---------------------------------------------------------------------------
# 2. Decision discipline
# ---------------------------------------------------------------------------


def test_decision_discipline_pass_fail_empty() -> None:
    good_d = _mk_decision(
        decision_id="g1",
        citations=[_cite("s1", "c1"), _cite("s2", "c2")],
    )
    bad_d = _mk_decision(
        decision_id="b1",
        citations=[_cite("s1", "c1")],  # only 1 citation
    )

    r_pass = compute_decision_discipline([good_d])
    assert r_pass.name == "decision_discipline"
    assert r_pass.value == "1/1"
    assert r_pass.threshold == "1/1"
    assert r_pass.status == "pass"

    r_fail = compute_decision_discipline([good_d, bad_d])
    assert r_fail.value == "1/2"
    assert r_fail.threshold == "2/2"
    assert r_fail.status == "fail"

    r_empty = compute_decision_discipline([])
    assert r_empty.value == "0/0"
    assert r_empty.threshold == "0/0"
    assert r_empty.status == "info"


# ---------------------------------------------------------------------------
# 3. Citation diversity
# ---------------------------------------------------------------------------


def test_citation_diversity_pass_fail_empty() -> None:
    diverse = _mk_decision(
        decision_id="d1",
        citations=[_cite("src-A", "c1"), _cite("src-B", "c2")],
    )
    same_source = _mk_decision(
        decision_id="d2",
        citations=[_cite("src-A", "c1"), _cite("src-A", "c2")],
    )

    r_pass = compute_citation_diversity([diverse])
    assert r_pass.name == "citation_diversity"
    assert r_pass.value == "1/1"
    assert r_pass.status == "pass"

    r_fail = compute_citation_diversity([diverse, same_source])
    assert r_fail.value == "1/2"
    assert r_fail.threshold == "2/2"
    assert r_fail.status == "fail"

    r_empty = compute_citation_diversity([])
    assert r_empty.value == "0/0"
    assert r_empty.status == "info"


# ---------------------------------------------------------------------------
# 4. Reversal rate
# ---------------------------------------------------------------------------


def test_reversal_rate_pass_warn_fail_empty() -> None:
    # Pass: 0% reversal (no losers within 3 days).
    pass_trades = [
        ClosedTrade("AAPL", date(2024, 1, 1), date(2024, 1, 4), Decimal("100")),
        ClosedTrade("MSFT", date(2024, 1, 1), date(2024, 1, 2), Decimal("50")),
    ]
    r_pass = compute_reversal_rate(pass_trades)
    assert r_pass.name == "reversal_rate"
    assert r_pass.value == 0.0
    assert r_pass.threshold == 30.0
    assert r_pass.status == "pass"

    # Warn: 2/5 = 40% reversal within 3 days.
    warn_trades = [
        ClosedTrade("A", date(2024, 1, 1), date(2024, 1, 3), Decimal("-1")),
        ClosedTrade("B", date(2024, 1, 1), date(2024, 1, 2), Decimal("-1")),
        ClosedTrade("C", date(2024, 1, 1), date(2024, 1, 4), Decimal("10")),
        ClosedTrade("D", date(2024, 1, 1), date(2024, 1, 4), Decimal("10")),
        ClosedTrade("E", date(2024, 1, 1), date(2024, 1, 4), Decimal("10")),
    ]
    r_warn = compute_reversal_rate(warn_trades)
    assert r_warn.value == 40.0
    assert r_warn.status == "warn"

    # Fail: 3/5 = 60% reversal.
    fail_trades = [
        ClosedTrade("A", date(2024, 1, 1), date(2024, 1, 3), Decimal("-1")),
        ClosedTrade("B", date(2024, 1, 1), date(2024, 1, 2), Decimal("-1")),
        ClosedTrade("C", date(2024, 1, 1), date(2024, 1, 1), Decimal("-1")),
        ClosedTrade("D", date(2024, 1, 1), date(2024, 1, 4), Decimal("10")),
        ClosedTrade("E", date(2024, 1, 1), date(2024, 1, 4), Decimal("10")),
    ]
    r_fail = compute_reversal_rate(fail_trades)
    assert r_fail.value == 60.0
    assert r_fail.status == "fail"

    # Loss OUTSIDE 3-day window doesn't count.
    outside_trades = [
        ClosedTrade("A", date(2024, 1, 1), date(2024, 1, 10), Decimal("-1")),
    ]
    r_outside = compute_reversal_rate(outside_trades)
    assert r_outside.value == 0.0
    assert r_outside.status == "pass"

    # Empty → info.
    r_empty = compute_reversal_rate([])
    assert r_empty.value == 0.0
    assert r_empty.status == "info"
    assert r_empty.threshold == 30.0


# ---------------------------------------------------------------------------
# 5. Risk-policy compliance
# ---------------------------------------------------------------------------


def test_risk_policy_compliance_pass_fail() -> None:
    log_pass = [
        {"event": "decision_emitted", "id": "d1"},
        {"event": "trade_filled", "id": "f1"},
        # Upstream-blocked breach should NOT count.
        {"event": "risk_limit_blocked_upstream", "id": "x1"},
    ]
    r_pass = compute_risk_policy_compliance(log_pass)
    assert r_pass.name == "risk_policy_compliance"
    assert r_pass.value == 0
    assert r_pass.threshold == 0
    assert r_pass.status == "pass"

    log_fail = [
        {"event": "decision_emitted"},
        {"event": "risk_violation_reached_broker", "ticker": "AAPL"},
        {"event": "policy_breach_executed", "ticker": "MSFT"},
        {"event": "trade_filled"},
    ]
    r_fail = compute_risk_policy_compliance(log_fail)
    assert r_fail.value == 2
    assert r_fail.status == "fail"


# ---------------------------------------------------------------------------
# 6. HITL correctness
# ---------------------------------------------------------------------------


def test_hitl_correctness_pass_fail_empty_and_below_threshold_ignored() -> None:
    # Below-threshold rows must NOT contribute to either side of the ratio.
    pairs_pass = [
        HitlPair("d1", above_threshold=True, approval_valid=True),
        HitlPair("d2", above_threshold=False, approval_valid=False),
        HitlPair("d3", above_threshold=True, approval_valid=True),
    ]
    r_pass = compute_hitl_correctness(pairs_pass)
    assert r_pass.name == "hitl_correctness"
    assert r_pass.value == "2/2"
    assert r_pass.threshold == "2/2"
    assert r_pass.status == "pass"

    pairs_fail = [
        HitlPair("d1", above_threshold=True, approval_valid=True),
        HitlPair("d2", above_threshold=True, approval_valid=False),
        HitlPair("d3", above_threshold=False, approval_valid=False),
    ]
    r_fail = compute_hitl_correctness(pairs_fail)
    assert r_fail.value == "1/2"
    assert r_fail.threshold == "2/2"
    assert r_fail.status == "fail"

    # All below-threshold → denominator 0 → info.
    pairs_only_below = [
        HitlPair("d1", above_threshold=False, approval_valid=False),
        HitlPair("d2", above_threshold=False, approval_valid=True),
    ]
    r_empty = compute_hitl_correctness(pairs_only_below)
    assert r_empty.value == "0/0"
    assert r_empty.status == "info"

    # Truly empty list → also info.
    assert compute_hitl_correctness([]).status == "info"


# ---------------------------------------------------------------------------
# 7. Schema rejections
# ---------------------------------------------------------------------------


def test_schema_rejections_info_only_and_negative_rejected() -> None:
    r_zero = compute_schema_rejections(0)
    assert r_zero.name == "schema_rejections"
    assert r_zero.value == 0
    assert r_zero.threshold is None
    assert r_zero.status == "info"

    r_some = compute_schema_rejections(7)
    assert r_some.value == 7
    assert r_some.threshold is None
    assert r_some.status == "info"

    with pytest.raises(ValueError):
        compute_schema_rejections(-1)


# ---------------------------------------------------------------------------
# 8. Red-team pass
# ---------------------------------------------------------------------------


def test_red_team_pass_pass_fail_and_validation() -> None:
    r_pass = compute_red_team_pass(50, 50)
    assert r_pass.name == "red_team_pass"
    assert r_pass.value == "50/50"
    assert r_pass.threshold == "50/50"
    assert r_pass.status == "pass"

    r_fail = compute_red_team_pass(48, 50)
    assert r_fail.value == "48/50"
    assert r_fail.threshold == "50/50"
    assert r_fail.status == "fail"

    with pytest.raises(ValueError):
        compute_red_team_pass(0, 0)
    with pytest.raises(ValueError):
        compute_red_team_pass(51, 50)
    with pytest.raises(ValueError):
        compute_red_team_pass(-1, 10)


# ---------------------------------------------------------------------------
# 9. Sufficiency gate
# ---------------------------------------------------------------------------


def test_sufficiency_gate_pass_fail_and_range_check() -> None:
    r_pass = compute_sufficiency_gate(0.85, 0.90)
    assert r_pass.name == "sufficiency_gate"
    assert r_pass.value == "p=0.85, r=0.90"
    assert r_pass.threshold == "p>=0.80, r>=0.80"
    assert r_pass.status == "pass"

    # Recall below threshold.
    r_fail = compute_sufficiency_gate(0.85, 0.70)
    assert r_fail.value == "p=0.85, r=0.70"
    assert r_fail.status == "fail"

    # Edge: exactly 0.80 / 0.80 passes.
    assert compute_sufficiency_gate(0.80, 0.80).status == "pass"

    with pytest.raises(ValueError):
        compute_sufficiency_gate(-0.1, 0.5)
    with pytest.raises(ValueError):
        compute_sufficiency_gate(0.5, 1.5)


# ---------------------------------------------------------------------------
# 10. FailureMode coverage
# ---------------------------------------------------------------------------


def test_failure_mode_coverage_pass_fail_and_unknown_ignored() -> None:
    # Sanity check: per Plan 4 §T12, eligible = all FailureMode values
    # EXCEPT UNKNOWN. Today that's 14; the metric still computes correctly
    # if the enum grows, but pinning the current value catches accidental
    # changes to UNKNOWN's role.
    eligible = set(FailureMode) - {FailureMode.UNKNOWN}
    assert len(eligible) == 14

    r_pass = compute_failure_mode_coverage(eligible)
    assert r_pass.name == "failure_mode_coverage"
    assert r_pass.value == "14/14"
    assert r_pass.threshold == "14/14"
    assert r_pass.status == "pass"

    # UNKNOWN in input must be silently ignored: passing only UNKNOWN
    # means 0 eligible covered → fail.
    r_unknown_only = compute_failure_mode_coverage([FailureMode.UNKNOWN])
    assert r_unknown_only.value == "0/14"
    assert r_unknown_only.status == "fail"

    # Partial coverage → fail.
    partial = [FailureMode.UNCITED_CLAIM, FailureMode.STALE_DATA]
    r_partial = compute_failure_mode_coverage(partial)
    assert r_partial.value == "2/14"
    assert r_partial.status == "fail"

    # Duplicates in the input don't inflate the count.
    dupes = list(eligible) + [FailureMode.UNCITED_CLAIM]
    r_dupes = compute_failure_mode_coverage(dupes)
    assert r_dupes.value == "14/14"
    assert r_dupes.status == "pass"


# ---------------------------------------------------------------------------
# Aggregator smoke test
# ---------------------------------------------------------------------------


def test_compute_all_metrics_returns_ten_in_order() -> None:
    inp = ProcessMetricsInput(
        claims=[],
        decisions=[],
        closed_trades=[],
        audit_log=[],
        hitl_required=[],
        rejection_count=0,
        red_team_passed=50,
        red_team_total=50,
        sufficiency_precision=0.85,
        sufficiency_recall=0.90,
        triggered_failure_modes=list(set(FailureMode) - {FailureMode.UNKNOWN}),
    )
    results = compute_all_metrics(inp)
    assert len(results) == 10
    expected_order = [
        "groundedness",
        "decision_discipline",
        "citation_diversity",
        "reversal_rate",
        "risk_policy_compliance",
        "hitl_correctness",
        "schema_rejections",
        "red_team_pass",
        "sufficiency_gate",
        "failure_mode_coverage",
    ]
    assert [r.name for r in results] == expected_order
    # Every returned object honors the contract.
    for r in results:
        assert isinstance(r, MetricResult)
        assert r.status in {"pass", "fail", "warn", "info"}


# ---------------------------------------------------------------------------
# MetricResult immutability
# ---------------------------------------------------------------------------


def test_metric_result_is_frozen_pydantic_model() -> None:
    r = MetricResult(name="x", value=1, threshold=None, status="info")
    # ``model_copy`` returns a new instance — must not raise.
    r2 = r.model_copy()
    assert r2 == r
    assert r2 is not r
    # Direct mutation must fail (frozen=True).
    with pytest.raises(Exception):
        r.value = 2
