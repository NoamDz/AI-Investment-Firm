"""Named market-regime configs for eval harness (Plan 4 §9.3).

Three regimes are defined as module-level constants:
  R1_EARNINGS  — 2024-03-11→15, earnings-heavy window
  R2_DRAWDOWN  — 2024-08-05→09, post-Aug-5 sell-off
  R3_QUIET     — 2023-11-06→10, low-volatility quiet window

Each carries the full 30-ticker universe loaded from config/universe.yaml.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from firm.core.config import load_universe

# ---------------------------------------------------------------------------
# Load the frozen universe once at module import.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent
_universe_cfg = load_universe(_REPO_ROOT / "config" / "universe.yaml")
_UNIVERSE: tuple[str, ...] = tuple(_universe_cfg.tickers)


# ---------------------------------------------------------------------------
# RegimeConfig — immutable Pydantic model (matches project convention).
# ---------------------------------------------------------------------------
class RegimeConfig(BaseModel):
    """Immutable specification of a named back-test market regime.

    Fields
    ------
    regime_id    : machine-readable identifier (e.g. ``"r1_earnings"``).
    description  : human-readable label (e.g. ``"earnings-heavy"``).
    start_date   : first calendar date of the regime window (inclusive).
    end_date     : last calendar date of the regime window (inclusive).
    universe     : frozen tuple of ticker symbols loaded from universe.yaml.
    seed_overrides: optional per-component RNG seed overrides; defaults
                   to an empty immutable mapping.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    regime_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    start_date: date
    end_date: date
    universe: tuple[str, ...] = Field(default_factory=tuple)
    seed_overrides: Mapping[str, int] = Field(default_factory=lambda: MappingProxyType({}))


# ---------------------------------------------------------------------------
# Three named regime instances (spec §9.3).
# ---------------------------------------------------------------------------
R1_EARNINGS = RegimeConfig(
    regime_id="r1_earnings",
    description="earnings-heavy",
    start_date=date(2024, 3, 11),
    end_date=date(2024, 3, 15),
    universe=_UNIVERSE,
    seed_overrides=MappingProxyType({}),
)

R2_DRAWDOWN = RegimeConfig(
    regime_id="r2_drawdown",
    description="post-Aug-5 sell-off",
    start_date=date(2024, 8, 5),
    end_date=date(2024, 8, 9),
    universe=_UNIVERSE,
    seed_overrides=MappingProxyType({}),
)

R3_QUIET = RegimeConfig(
    regime_id="r3_quiet",
    description="low-volatility quiet",
    start_date=date(2023, 11, 6),
    end_date=date(2023, 11, 10),
    universe=_UNIVERSE,
    seed_overrides=MappingProxyType({}),
)

ALL_REGIMES: tuple[RegimeConfig, ...] = (R1_EARNINGS, R2_DRAWDOWN, R3_QUIET)

_REGIME_INDEX: dict[str, RegimeConfig] = {r.regime_id: r for r in ALL_REGIMES}


def get_regime(regime_id: str) -> RegimeConfig:
    """Return the :class:`RegimeConfig` for *regime_id*.

    Raises
    ------
    KeyError
        If *regime_id* does not match any known regime.
    """
    try:
        return _REGIME_INDEX[regime_id]
    except KeyError:
        known = list(_REGIME_INDEX)
        raise KeyError(
            f"Unknown regime {regime_id!r}. Known regimes: {known}"
        ) from None
