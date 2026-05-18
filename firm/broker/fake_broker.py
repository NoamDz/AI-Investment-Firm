"""Deterministic in-memory broker for demo and tests. See spec §5.2, §5.7."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from decimal import Decimal

from firm.broker.protocol import OrderResult, Position, Quote


def _deterministic_price(ticker: str) -> Decimal:
    """Stable, ticker-dependent price for replayability."""
    h = int(hashlib.sha256(ticker.encode()).hexdigest(), 16) % 1000
    return Decimal(50 + h) + Decimal("0.50")


class FakeBroker:
    COMMISSION = Decimal("0.005")  # 0.5% per trade

    def __init__(self, initial_cash: Decimal = Decimal("100000")) -> None:
        self._cash: Decimal = initial_cash
        self._positions: dict[str, Position] = {}
        self._order_cache: dict[str, OrderResult] = {}  # idempotency_key → result

    def list_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.shares != Decimal("0")]

    def get_cash(self) -> Decimal:
        return self._cash

    def get_quote(self, ticker: str) -> Quote:
        return Quote(
            ticker=ticker,
            price=_deterministic_price(ticker),
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

    def submit(self, decision_payload: dict, idempotency_key: str) -> OrderResult:
        if idempotency_key in self._order_cache:
            return self._order_cache[idempotency_key]

        ticker = decision_payload["ticker"]
        shares = Decimal(str(decision_payload["shares"]))
        price = _deterministic_price(ticker)
        kind = decision_payload["kind"]
        slippage = price * Decimal("0.0005")  # 5 bps
        fill_price = price + slippage if kind == "buy" else price - slippage
        gross = fill_price * shares
        commission = gross * self.COMMISSION

        if kind == "buy":
            self._cash -= gross + commission
            prev = self._positions.get(ticker, Position(ticker=ticker, shares=Decimal("0"), avg_cost=Decimal("0")))
            new_shares = prev.shares + shares
            if new_shares > 0:
                new_avg = ((prev.avg_cost * prev.shares) + (fill_price * shares)) / new_shares
            else:
                new_avg = Decimal("0")
            self._positions[ticker] = Position(ticker=ticker, shares=new_shares, avg_cost=new_avg)
        elif kind == "sell":
            self._cash += gross - commission
            prev = self._positions.get(ticker, Position(ticker=ticker, shares=Decimal("0"), avg_cost=Decimal("0")))
            self._positions[ticker] = Position(
                ticker=ticker, shares=prev.shares - shares, avg_cost=prev.avg_cost
            )
        else:
            raise ValueError(f"unsupported order kind: {kind}")

        now = datetime.now(tz=timezone.utc).isoformat()
        result = OrderResult(
            order_id=hashlib.sha256(idempotency_key.encode()).hexdigest()[:16],
            ticker=ticker,
            filled_shares=shares,
            avg_fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            submitted_at=now,
            filled_at=now,
        )
        self._order_cache[idempotency_key] = result
        return result
