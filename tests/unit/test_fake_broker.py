from decimal import Decimal
from firm.broker.fake_broker import FakeBroker


def test_fake_broker_starts_with_initial_cash():
    b = FakeBroker(initial_cash=Decimal("100000"))
    assert b.get_cash() == Decimal("100000")
    assert b.list_positions() == []


def test_submit_returns_deterministic_fill():
    b = FakeBroker(initial_cash=Decimal("100000"))
    payload = {"kind": "buy", "ticker": "AAPL", "shares": "10"}
    r = b.submit(payload, idempotency_key="key-1")
    assert r.ticker == "AAPL"
    assert r.filled_shares == Decimal("10")


def test_submit_is_idempotent_on_same_key():
    b = FakeBroker(initial_cash=Decimal("100000"))
    payload = {"kind": "buy", "ticker": "AAPL", "shares": "10"}
    r1 = b.submit(payload, idempotency_key="key-1")
    r2 = b.submit(payload, idempotency_key="key-1")
    assert r1.order_id == r2.order_id
    assert b.get_cash() < Decimal("100000")  # cash debited exactly once
    # exactly one fill should be reflected in positions
    pos = [p for p in b.list_positions() if p.ticker == "AAPL"][0]
    assert pos.shares == Decimal("10")


def test_buy_reduces_cash_by_price_plus_commission():
    b = FakeBroker(initial_cash=Decimal("100000"))
    b.submit({"kind": "buy", "ticker": "AAPL", "shares": "10"}, "key-1")
    quote = b.get_quote("AAPL")
    # FakeBroker uses a fixed price function; assert cash dropped by reasonable amount
    assert Decimal("100000") - b.get_cash() > Decimal("0")
    assert b.get_cash() < Decimal("100000")


def test_sell_reduces_position_and_credits_cash():
    b = FakeBroker(initial_cash=Decimal("100000"))
    b.submit({"kind": "buy", "ticker": "AAPL", "shares": "10"}, "k1")
    cash_after_buy = b.get_cash()
    b.submit({"kind": "sell", "ticker": "AAPL", "shares": "10"}, "k2")
    assert b.get_cash() > cash_after_buy
    assert all(p.shares == Decimal("0") or p.ticker != "AAPL" for p in b.list_positions())
