"""Tests for aggregate_votes (Plan 2 §T26).

aggregate_votes is a deterministic pure-Python helper that combines three
single-lens PmVote objects (one per PmLens) into a committee decision.
T27 will wire this into make_pm().

The 7 tests below cover the 6 locked spec rules plus the rationale-carry
guarantee. Each test is a single-case test as required by the spec.
"""
from __future__ import annotations

from firm.agents.pm import PmLens, PmVote, aggregate_votes
from firm.core.models import ActionEnum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _vote(
    lens: PmLens,
    action: ActionEnum,
    *,
    confidence: float = 0.8,
    rationale: str = "default rationale text",
) -> PmVote:
    """Construct a PmVote with the given lens/action and sensible defaults."""
    return PmVote(
        lens=lens,
        vote=action,
        confidence=confidence,
        rationale=rationale,
        cited_claim_ids=["c1"],
    )


# ---------------------------------------------------------------------------
# Required tests — one per locked rule, plus the rationale-carry test.
# ---------------------------------------------------------------------------


def test_three_buy_yields_buy_high_confidence() -> None:
    """Rule 1 (unanimous): 3 BUY → BUY at full mean confidence."""
    votes = [
        _vote(PmLens.QUALITY, ActionEnum.BUY, confidence=0.9),
        _vote(PmLens.VALUATION, ActionEnum.BUY, confidence=0.8),
        _vote(PmLens.CATALYST, ActionEnum.BUY, confidence=0.7),
    ]
    action, confidence, rationale, failure = aggregate_votes(votes)
    assert action is ActionEnum.BUY
    # Unanimous: full mean of the three confidences (no discount).
    assert confidence >= min(v.confidence for v in votes)
    assert abs(confidence - (0.9 + 0.8 + 0.7) / 3) < 1e-9
    assert failure is None
    assert rationale  # non-empty


def test_two_buy_one_hold_yields_buy_with_mild_reservation() -> None:
    """Rule 2: 2 BUY + 1 HOLD → BUY, but confidence is discounted vs unanimous."""
    votes = [
        _vote(PmLens.QUALITY, ActionEnum.BUY, confidence=0.9),
        _vote(PmLens.VALUATION, ActionEnum.BUY, confidence=0.9),
        _vote(PmLens.CATALYST, ActionEnum.HOLD, confidence=0.9),
    ]
    action, confidence, _rationale, failure = aggregate_votes(votes)
    assert action is ActionEnum.BUY
    # Mean of the two BUY confidences (0.9) discounted by 0.8 = 0.72.
    assert abs(confidence - 0.9 * 0.8) < 1e-9
    # Sanity: lower than the same three voters would have produced at unanimity.
    unanimous_buy = [
        _vote(PmLens.QUALITY, ActionEnum.BUY, confidence=0.9),
        _vote(PmLens.VALUATION, ActionEnum.BUY, confidence=0.9),
        _vote(PmLens.CATALYST, ActionEnum.BUY, confidence=0.9),
    ]
    _, unanimous_conf, _, _ = aggregate_votes(unanimous_buy)
    assert confidence < unanimous_conf
    assert failure is None


def test_two_buy_one_sell_yields_escalate() -> None:
    """Rule 3 (informative split): 2 BUY + 1 SELL → ESCALATE.

    Directional disagreement with no HOLD bridge means HITL must adjudicate.
    """
    votes = [
        _vote(PmLens.QUALITY, ActionEnum.BUY, confidence=0.8),
        _vote(PmLens.VALUATION, ActionEnum.BUY, confidence=0.7),
        _vote(PmLens.CATALYST, ActionEnum.SELL, confidence=0.9),
    ]
    action, _confidence, _rationale, failure = aggregate_votes(votes)
    assert action is ActionEnum.ESCALATE
    # FailureMode slot is None — ESCALATE is HITL routing, not a failure.
    assert failure is None


def test_one_buy_two_sell_yields_sell() -> None:
    """Rule 4: 1 BUY + 2 SELL → SELL (majority direction, with reservation)."""
    votes = [
        _vote(PmLens.QUALITY, ActionEnum.SELL, confidence=0.8),
        _vote(PmLens.VALUATION, ActionEnum.BUY, confidence=0.6),
        _vote(PmLens.CATALYST, ActionEnum.SELL, confidence=0.8),
    ]
    action, confidence, _rationale, failure = aggregate_votes(votes)
    assert action is ActionEnum.SELL
    # Mean of the two SELL confidences (0.8) discounted by 0.8 = 0.64.
    assert abs(confidence - 0.8 * 0.8) < 1e-9
    assert failure is None


def test_three_hold_yields_hold() -> None:
    """Rule 5 (unanimous HOLD): 3 HOLD → HOLD at full mean confidence."""
    votes = [
        _vote(PmLens.QUALITY, ActionEnum.HOLD, confidence=0.5),
        _vote(PmLens.VALUATION, ActionEnum.HOLD, confidence=0.6),
        _vote(PmLens.CATALYST, ActionEnum.HOLD, confidence=0.7),
    ]
    action, confidence, _rationale, failure = aggregate_votes(votes)
    assert action is ActionEnum.HOLD
    assert abs(confidence - (0.5 + 0.6 + 0.7) / 3) < 1e-9
    assert failure is None


def test_buy_hold_sell_mix_yields_escalate() -> None:
    """Rule 6 (full disagreement): 1 BUY + 1 HOLD + 1 SELL → ESCALATE."""
    votes = [
        _vote(PmLens.QUALITY, ActionEnum.BUY, confidence=0.7),
        _vote(PmLens.VALUATION, ActionEnum.HOLD, confidence=0.7),
        _vote(PmLens.CATALYST, ActionEnum.SELL, confidence=0.7),
    ]
    action, _confidence, _rationale, failure = aggregate_votes(votes)
    assert action is ActionEnum.ESCALATE
    assert failure is None


def test_aggregation_carries_per_lens_rationales_into_metadata() -> None:
    """Per-lens rationales must appear in the combined rationale string,
    ordered QUALITY → VALUATION → CATALYST, so T27 can lift them into
    Decision.metadata.
    """
    votes = [
        # Pass votes in arbitrary order to confirm the aggregator orders by lens.
        _vote(PmLens.CATALYST, ActionEnum.BUY, rationale="c-rationale"),
        _vote(PmLens.QUALITY, ActionEnum.BUY, rationale="q-rationale"),
        _vote(PmLens.VALUATION, ActionEnum.BUY, rationale="v-rationale"),
    ]
    _action, _confidence, rationale, _failure = aggregate_votes(votes)
    assert "q-rationale" in rationale
    assert "v-rationale" in rationale
    assert "c-rationale" in rationale
    # Strict ordering: quality before valuation before catalyst.
    q_idx = rationale.index("q-rationale")
    v_idx = rationale.index("v-rationale")
    c_idx = rationale.index("c-rationale")
    assert q_idx < v_idx < c_idx
    # Lens labels should appear so downstream consumers can disambiguate.
    assert "quality" in rationale
    assert "valuation" in rationale
    assert "catalyst" in rationale
