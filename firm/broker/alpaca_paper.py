"""Alpaca paper trading adapter. Activated only when FIRM_BROKER=ALPACA."""
from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

from firm.broker.protocol import OrderResult, Position, Quote


class AlpacaBroker:
    """Thin wrapper around alpaca-py for paper trading.

    Lazy-imports alpaca-py so the FakeBroker default path has no SDK dependency.
    """

    def __init__(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as e:
            raise RuntimeError(
                "alpaca-py not installed; install with `pip install alpaca-py`"
            ) from e

        api_key = os.environ["ALPACA_API_KEY"]
        secret = os.environ["ALPACA_SECRET_KEY"]
        self._trading = TradingClient(api_key, secret, paper=True)
        self._data = StockHistoricalDataClient(api_key, secret)

    def list_positions(self) -> list[Position]:
        return [
            Position(
                ticker=p.symbol,
                shares=Decimal(str(p.qty)),
                avg_cost=Decimal(str(p.avg_entry_price)),
            )
            for p in self._trading.get_all_positions()
        ]

    def get_cash(self) -> Decimal:
        acct = self._trading.get_account()
        return Decimal(str(acct.cash))

    def get_quote(self, ticker: str) -> Quote:
        from alpaca.data.requests import StockLatestQuoteRequest
        q = self._data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))[ticker]
        # Use the midpoint of bid/ask as price
        price = (Decimal(str(q.bid_price)) + Decimal(str(q.ask_price))) / Decimal("2")
        return Quote(ticker=ticker, price=price, timestamp=q.timestamp.isoformat())

    def submit(self, decision_payload: dict[str, Any], idempotency_key: str) -> OrderResult:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        side = OrderSide.BUY if decision_payload["kind"] == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=decision_payload["ticker"],
            qty=float(decision_payload["shares"]),
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=idempotency_key,  # Alpaca's idempotency mechanism
        )
        order = self._trading.submit_order(req)
        return OrderResult(
            order_id=str(order.id),
            ticker=order.symbol,
            filled_shares=Decimal(str(order.filled_qty or 0)),
            avg_fill_price=Decimal(str(order.filled_avg_price or 0)),
            commission=Decimal("0"),  # Alpaca paper has no commission
            slippage=Decimal("0"),
            submitted_at=order.submitted_at.isoformat(),
            filled_at=order.filled_at.isoformat() if order.filled_at else order.submitted_at.isoformat(),
        )


def make_broker():
    """Factory: select broker by FIRM_BROKER env var. Default FakeBroker."""
    kind = os.environ.get("FIRM_BROKER", "FAKE").upper()
    if kind == "ALPACA":
        return AlpacaBroker()
    from firm.broker.fake_broker import FakeBroker
    return FakeBroker()
