"""Broker abstraction. See spec §5.2, §5.7."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Protocol

from pydantic import BaseModel


class Position(BaseModel):
    ticker: str
    shares: Decimal
    avg_cost: Decimal


class Quote(BaseModel):
    ticker: str
    price: Decimal
    timestamp: str  # ISO 8601


class OrderResult(BaseModel):
    order_id: str
    ticker: str
    filled_shares: Decimal
    avg_fill_price: Decimal
    commission: Decimal
    slippage: Decimal
    submitted_at: str
    filled_at: str


class Broker(Protocol):
    def list_positions(self) -> list[Position]: ...
    def get_cash(self) -> Decimal: ...
    def get_quote(self, ticker: str) -> Quote: ...
    def submit(self, decision_payload: dict[str, Any], idempotency_key: str) -> OrderResult: ...
