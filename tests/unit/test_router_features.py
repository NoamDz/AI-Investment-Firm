"""Tests for RouterFeatures model and score(weights) -> profile_name (Plan 3 T05)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from firm.core.models import RouterFeatures


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

def test_construction_valid_floats():
    rf = RouterFeatures(
        risk_weight=0.5,
        novelty=0.25,
        complexity=0.75,
        time_pressure=0.1,
    )
    assert rf.risk_weight == 0.5
    assert rf.novelty == 0.25
    assert rf.complexity == 0.75
    assert rf.time_pressure == 0.1


def test_construction_boundaries_zero_and_one():
    # Both 0.0 and 1.0 are inclusive boundaries.
    RouterFeatures(risk_weight=0.0, novelty=0.0, complexity=0.0, time_pressure=0.0)
    RouterFeatures(risk_weight=1.0, novelty=1.0, complexity=1.0, time_pressure=1.0)


@pytest.mark.parametrize("field", ["risk_weight", "novelty", "complexity", "time_pressure"])
def test_negative_value_rejected(field: str):
    kwargs = {"risk_weight": 0.5, "novelty": 0.5, "complexity": 0.5, "time_pressure": 0.5}
    kwargs[field] = -0.01
    with pytest.raises(ValidationError):
        RouterFeatures(**kwargs)


@pytest.mark.parametrize("field", ["risk_weight", "novelty", "complexity", "time_pressure"])
def test_over_one_value_rejected(field: str):
    kwargs = {"risk_weight": 0.5, "novelty": 0.5, "complexity": 0.5, "time_pressure": 0.5}
    kwargs[field] = 1.01
    with pytest.raises(ValidationError):
        RouterFeatures(**kwargs)


# ---------------------------------------------------------------------------
# Scoring — buckets
# ---------------------------------------------------------------------------

_EQUAL_WEIGHTS = {
    "risk_weight": 1.0,
    "novelty": 1.0,
    "complexity": 1.0,
    "time_pressure": 1.0,
}


def test_low_stakes_all_zero_returns_haiku():
    rf = RouterFeatures(risk_weight=0.0, novelty=0.0, complexity=0.0, time_pressure=0.0)
    # Any positive weights — score is 0.
    assert rf.score(_EQUAL_WEIGHTS) == "haiku"
    assert rf.score({"risk_weight": 3.0, "novelty": 0.5, "complexity": 7.0, "time_pressure": 0.1}) == "haiku"


def test_high_stakes_all_one_returns_opus():
    rf = RouterFeatures(risk_weight=1.0, novelty=1.0, complexity=1.0, time_pressure=1.0)
    assert rf.score(_EQUAL_WEIGHTS) == "opus"
    assert rf.score({"risk_weight": 2.5, "novelty": 0.3, "complexity": 9.9, "time_pressure": 0.7}) == "opus"


def test_standard_around_half_returns_sonnet():
    rf = RouterFeatures(risk_weight=0.5, novelty=0.5, complexity=0.5, time_pressure=0.5)
    # Normalized score is exactly 0.5 — falls in [0.33, 0.66) → sonnet.
    assert rf.score(_EQUAL_WEIGHTS) == "sonnet"


# ---------------------------------------------------------------------------
# Boundary cases — `<` and `>=` semantics
# ---------------------------------------------------------------------------
#
# Each case drives ``risk_weight`` to a chosen normalized score by putting
# weight 1.0 on risk_weight and 0.0 on the others, so normalized score equals
# the ``risk_weight`` value directly.
@pytest.mark.parametrize(
    "risk_value,expected",
    [
        (0.32, "haiku"),   # just below low cutoff
        (0.33, "sonnet"),  # exact low cutoff — `<` is strict, so this is sonnet
        (0.65, "sonnet"),  # just below high cutoff
        (0.66, "opus"),    # exact high cutoff — `>=` includes this
    ],
    ids=["just_below_low_haiku", "exact_low_sonnet", "just_below_high_sonnet", "exact_high_opus"],
)
def test_score_thresholds_boundary(risk_value: float, expected: str):
    rf = RouterFeatures(
        risk_weight=risk_value, novelty=0.0, complexity=0.0, time_pressure=0.0
    )
    weights = {"risk_weight": 1.0, "novelty": 0.0, "complexity": 0.0, "time_pressure": 0.0}
    assert rf.score(weights) == expected, (
        f"normalized score s={risk_value} should map to {expected!r}"
    )


# ---------------------------------------------------------------------------
# Weight semantics
# ---------------------------------------------------------------------------

def test_weights_steer_score_to_opus_via_single_feature():
    rf = RouterFeatures(risk_weight=1.0, novelty=0.0, complexity=0.0, time_pressure=0.0)
    weights = {"risk_weight": 1.0, "novelty": 0.0, "complexity": 0.0, "time_pressure": 0.0}
    # normalized = 1.0/1.0 = 1.0 → opus.
    assert rf.score(weights) == "opus"


def test_weights_steer_score_to_haiku_when_active_feature_has_zero_weight():
    rf = RouterFeatures(risk_weight=1.0, novelty=0.0, complexity=0.0, time_pressure=0.0)
    # Zero out the only nonzero feature's weight; other weights are nonzero but their features are 0.
    weights = {"risk_weight": 0.0, "novelty": 1.0, "complexity": 1.0, "time_pressure": 1.0}
    # weighted sum = 0; sum(weights) = 3.0 → normalized = 0 → haiku.
    assert rf.score(weights) == "haiku"


def test_weight_magnitudes_dont_change_normalized_score():
    rf = RouterFeatures(risk_weight=0.4, novelty=0.4, complexity=0.4, time_pressure=0.4)
    small = {"risk_weight": 0.1, "novelty": 0.1, "complexity": 0.1, "time_pressure": 0.1}
    big = {"risk_weight": 10.0, "novelty": 10.0, "complexity": 10.0, "time_pressure": 10.0}
    assert rf.score(small) == rf.score(big)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_all_zero_weights_raises_value_error():
    rf = RouterFeatures(risk_weight=0.5, novelty=0.5, complexity=0.5, time_pressure=0.5)
    weights = {"risk_weight": 0.0, "novelty": 0.0, "complexity": 0.0, "time_pressure": 0.0}
    with pytest.raises(ValueError, match="router weights sum to zero"):
        rf.score(weights)


def test_missing_weight_key_raises_value_error():
    rf = RouterFeatures(risk_weight=0.5, novelty=0.5, complexity=0.5, time_pressure=0.5)
    # 'complexity' is missing — should not silently default to 0 (hides config bugs).
    weights = {"risk_weight": 1.0, "novelty": 1.0, "time_pressure": 1.0}
    with pytest.raises(ValueError, match="missing router weight"):
        rf.score(weights)


def test_negative_weight_value_raises():
    # Negative weights are a config bug — disallow.
    rf = RouterFeatures(risk_weight=0.5, novelty=0.5, complexity=0.5, time_pressure=0.5)
    weights = {"risk_weight": 1.0, "novelty": -1.0, "complexity": 1.0, "time_pressure": 1.0}
    with pytest.raises(ValueError, match="negative router weight"):
        rf.score(weights)


def test_extra_weight_key_raises_value_error():
    # Unknown keys in the weights dict (e.g. a typo in router.yaml) must
    # fail loudly rather than be silently dropped.
    rf = RouterFeatures(risk_weight=0.5, novelty=0.5, complexity=0.5, time_pressure=0.5)
    weights = {
        "risk_weight": 1.0,
        "novelty": 1.0,
        "complexity": 1.0,
        "time_pressure": 1.0,
        "bogus_key": 0.5,
    }
    with pytest.raises(ValueError, match="unknown router weight"):
        rf.score(weights)
