"""Pydantic-validated YAML config loaders. See spec §3.7."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class PolicyLimits(BaseModel):
    max_position_pct: float
    max_sector_pct: float
    max_gross_exposure: float
    max_trade_pct: float
    max_trades_per_day: int
    min_cash_pct: float
    max_daily_loss_pct: float
    stale_quote_seconds: int
    stale_filing_days: int


class HitlConfig(BaseModel):
    trade_threshold_pct: float
    escalate_new_ticker: bool


class PolicyConfig(BaseModel):
    limits: PolicyLimits
    hitl: HitlConfig


class UniverseConfig(BaseModel):
    as_of: date
    tickers: list[str]
    sector_map: dict[str, str]

    @model_validator(mode="after")
    def _check_all_tickers_mapped(self) -> "UniverseConfig":
        unmapped = [t for t in self.tickers if t not in self.sector_map]
        if unmapped:
            raise ValueError(f"unmapped tickers: {unmapped}")
        return self


def load_policy(path: Path) -> PolicyConfig:
    return PolicyConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_universe(path: Path) -> UniverseConfig:
    return UniverseConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
