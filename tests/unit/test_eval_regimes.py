"""Unit tests for firm.eval.regimes (Plan 4 T09)."""
from __future__ import annotations

import copy
from datetime import date
from types import MappingProxyType

import pytest
from pydantic import ValidationError

from firm.eval.regimes import (
    ALL_REGIMES,
    R1_EARNINGS,
    R2_DRAWDOWN,
    R3_QUIET,
    RegimeConfig,
    get_regime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regime_to_dict(r: RegimeConfig) -> dict:
    """Convert a RegimeConfig to a plain dict for serialization round-trip."""
    return {
        "regime_id": r.regime_id,
        "description": r.description,
        "start_date": r.start_date.isoformat(),
        "end_date": r.end_date.isoformat(),
        "universe": list(r.universe),
        "seed_overrides": dict(r.seed_overrides),
    }


def _regime_from_dict(d: dict) -> RegimeConfig:
    """Reconstruct a RegimeConfig from the dict produced by _regime_to_dict."""
    return RegimeConfig(
        regime_id=d["regime_id"],
        description=d["description"],
        start_date=date.fromisoformat(d["start_date"]),
        end_date=date.fromisoformat(d["end_date"]),
        universe=tuple(d["universe"]),
        seed_overrides=MappingProxyType(d["seed_overrides"]),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_three_regimes_have_correct_dates_and_descriptions() -> None:
    """Each named regime matches the spec §9.3 values verbatim."""
    # R1 — earnings-heavy
    assert R1_EARNINGS.regime_id == "r1_earnings"
    assert R1_EARNINGS.description == "earnings-heavy"
    assert R1_EARNINGS.start_date == date(2024, 3, 11)
    assert R1_EARNINGS.end_date == date(2024, 3, 15)

    # R2 — post-Aug-5 sell-off
    assert R2_DRAWDOWN.regime_id == "r2_drawdown"
    assert R2_DRAWDOWN.description == "post-Aug-5 sell-off"
    assert R2_DRAWDOWN.start_date == date(2024, 8, 5)
    assert R2_DRAWDOWN.end_date == date(2024, 8, 9)

    # R3 — low-volatility quiet
    assert R3_QUIET.regime_id == "r3_quiet"
    assert R3_QUIET.description == "low-volatility quiet"
    assert R3_QUIET.start_date == date(2023, 11, 6)
    assert R3_QUIET.end_date == date(2023, 11, 10)


def test_universe_is_frozen_30_tickers() -> None:
    """All three regimes share the same frozen 30-ticker universe."""
    # All three universes are identical
    assert R1_EARNINGS.universe == R2_DRAWDOWN.universe == R3_QUIET.universe

    # Exactly 30 tickers (verified from config/universe.yaml)
    assert len(R1_EARNINGS.universe) == 30

    # It is a tuple (frozen)
    assert isinstance(R1_EARNINGS.universe, tuple)

    # Spot-check a few known tickers
    assert "AAPL" in R1_EARNINGS.universe
    assert "MSFT" in R1_EARNINGS.universe
    assert "NVDA" in R1_EARNINGS.universe


def test_deepcopy_equality() -> None:
    """deepcopy of each regime equals the original."""
    for regime in (R1_EARNINGS, R2_DRAWDOWN, R3_QUIET):
        cloned = copy.deepcopy(regime)
        assert cloned == regime


def test_serialization_round_trip() -> None:
    """Each regime survives a dict serialization round-trip."""
    for original in (R1_EARNINGS, R2_DRAWDOWN, R3_QUIET):
        d = _regime_to_dict(original)
        reconstructed = _regime_from_dict(d)

        assert reconstructed.regime_id == original.regime_id
        assert reconstructed.description == original.description
        assert reconstructed.start_date == original.start_date
        assert reconstructed.end_date == original.end_date
        assert reconstructed.universe == original.universe
        assert dict(reconstructed.seed_overrides) == dict(original.seed_overrides)


def test_get_regime_returns_named_regime() -> None:
    """get_regime() returns the correct singleton; unknown id raises KeyError."""
    assert get_regime("r1_earnings") is R1_EARNINGS
    assert get_regime("r2_drawdown") is R2_DRAWDOWN
    assert get_regime("r3_quiet") is R3_QUIET

    with pytest.raises(KeyError):
        get_regime("nonexistent")


def test_all_regimes_constant() -> None:
    """ALL_REGIMES has length 3 and contains the three named regimes."""
    assert len(ALL_REGIMES) == 3
    assert R1_EARNINGS in ALL_REGIMES
    assert R2_DRAWDOWN in ALL_REGIMES
    assert R3_QUIET in ALL_REGIMES
    assert isinstance(ALL_REGIMES, tuple)


def test_seed_overrides_is_truly_immutable() -> None:
    """seed_overrides must be a MappingProxyType, not a coerced dict."""
    for regime in (R1_EARNINGS, R2_DRAWDOWN, R3_QUIET):
        assert isinstance(regime.seed_overrides, MappingProxyType)
        with pytest.raises(TypeError):
            regime.seed_overrides["new_key"] = 99  # type: ignore[index]


def test_regime_config_rejects_inverted_dates() -> None:
    """end_date < start_date must raise ValidationError at construction time."""
    with pytest.raises(ValidationError, match="must be >= start_date"):
        RegimeConfig(
            regime_id="bogus",
            description="x",
            start_date=date(2024, 8, 9),
            end_date=date(2024, 8, 5),
            universe=R1_EARNINGS.universe,
            seed_overrides={},
        )
