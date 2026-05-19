"""Pydantic-validated YAML config loaders. See spec §3.7."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class PolicyLimits(BaseModel):
    max_position_pct: float = Field(gt=0, le=1.0)
    max_sector_pct: float = Field(gt=0, le=1.0)
    max_gross_exposure: float = Field(gt=0, le=2.0)
    max_trade_pct: float = Field(gt=0, le=1.0)
    max_trades_per_day: int = Field(gt=0)
    min_cash_pct: float = Field(ge=0, lt=1.0)
    max_daily_loss_pct: float = Field(gt=0, le=1.0)
    stale_quote_seconds: int = Field(gt=0)
    stale_filing_days: int = Field(gt=0)


class HitlConfig(BaseModel):
    trade_threshold_pct: float = Field(ge=0, le=1.0)
    escalate_new_ticker: bool


class PolicyConfig(BaseModel):
    limits: PolicyLimits
    hitl: HitlConfig


class UniverseConfig(BaseModel):
    as_of: date
    tickers: list[str] = Field(min_length=1)
    sector_map: dict[str, str]

    @model_validator(mode="after")
    def _check_tickers_consistency(self) -> "UniverseConfig":
        unmapped = [t for t in self.tickers if t not in self.sector_map]
        if unmapped:
            raise ValueError(f"unmapped tickers: {unmapped}")
        if len(self.tickers) != len(set(self.tickers)):
            counts: dict[str, int] = {}
            for t in self.tickers:
                counts[t] = counts.get(t, 0) + 1
            dupes = sorted(t for t, c in counts.items() if c > 1)
            raise ValueError(f"duplicate tickers: {dupes}")
        orphans = sorted(k for k in self.sector_map if k not in set(self.tickers))
        if orphans:
            raise ValueError(f"sector_map keys not in tickers: {orphans}")
        return self


def load_policy(path: Path) -> PolicyConfig:
    return PolicyConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_universe(path: Path) -> UniverseConfig:
    return UniverseConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
