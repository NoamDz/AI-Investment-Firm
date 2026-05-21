"""Tests for ``firm.eval.perf_metrics`` (Plan 4 §T11).

All Decimal literals are constructed from strings to keep the inputs free of
float-to-Decimal noise; expected values are hand-computed and pinned to 1 dp.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from firm.eval.perf_metrics import Fill, compute_perf_metrics


def test_total_return_simple() -> None:
    # +5.0% with no trades; spy=+2.0%, basket=+1.0%.
    result = compute_perf_metrics(
        initial_cash=Decimal("10000"),
        fills=[],
        final_cash=Decimal("10500"),
        final_positions={},
        final_marks={},
        spy_return=0.02,
        basket_return=0.01,
    )
    assert result["total_return_pct"] == 5.0
    assert result["spy_return_pct"] == 2.0
    assert result["basket_return_pct"] == 1.0
    assert result["vs_spy_pp"] == 3.0
    assert result["vs_basket_pp"] == 4.0
    assert result["per_trade_returns_str"] == "(no closed trades)"
    assert result["hit_rate_str"] == "0/0 (n/a) — no closed trades"


def test_per_trade_returns_two_round_trips() -> None:
    # AAPL round-trip: buy 10 @ 100 → sell 10 @ 110 → +10.0%
    # MSFT round-trip: buy 10 @ 100 → sell 10 @ 95  → −5.0%
    # Zero commission for clean hand-computed values.
    fills = [
        Fill("buy",  "AAPL", Decimal("10"), Decimal("100"), Decimal("0")),
        Fill("sell", "AAPL", Decimal("10"), Decimal("110"), Decimal("0")),
        Fill("buy",  "MSFT", Decimal("10"), Decimal("100"), Decimal("0")),
        Fill("sell", "MSFT", Decimal("10"), Decimal("95"),  Decimal("0")),
    ]
    # Initial 10000; spent 1000 on AAPL, got back 1100; spent 1000 on MSFT,
    # got back 950 → final_cash = 10000 - 1000 + 1100 - 1000 + 950 = 10050.
    result = compute_perf_metrics(
        initial_cash=Decimal("10000"),
        fills=fills,
        final_cash=Decimal("10050"),
        final_positions={},
        final_marks={},
        spy_return=0.0,
        basket_return=0.0,
    )
    assert result["per_trade_returns_str"] == "+10.0%, -5.0%"
    assert (
        result["hit_rate_str"]
        == "1/2 (50%) — n=2, not statistically significant"
    )
    # Sanity: 50 / 10000 = 0.5%
    assert result["total_return_pct"] == 0.5


def test_open_position_excluded_from_per_trade_but_in_total() -> None:
    # BUY 100 AAPL @ 100, no sell. Mark AAPL @ 108 → position +8%.
    # Initial 20000 → spent 10000 → cash 10000; positions_value = 100×108 = 10800.
    # Ending equity = 20800; total return = 800/20000 = +4.0%.
    fills = [
        Fill("buy", "AAPL", Decimal("100"), Decimal("100"), Decimal("0")),
    ]
    result = compute_perf_metrics(
        initial_cash=Decimal("20000"),
        fills=fills,
        final_cash=Decimal("10000"),
        final_positions={"AAPL": Decimal("100")},
        final_marks={"AAPL": Decimal("108")},
        spy_return=0.0,
        basket_return=0.0,
    )
    assert result["per_trade_returns_str"] == "(no closed trades)"
    assert result["hit_rate_str"] == "0/0 (n/a) — no closed trades"
    assert result["total_return_pct"] == 4.0


def test_fifo_partial_consumption() -> None:
    # Two buy lots, one larger sell that straddles the boundary.
    # Lot 1: 10 AAPL @ 100. Lot 2: 10 AAPL @ 110. SELL 15 AAPL @ 120, zero comm.
    # FIFO match:
    #   - 10 shares vs lot1 @100 → (120-100)/100 × 100 = +20.0%
    #   - 5  shares vs lot2 @110 → (120-110)/110 × 100 = 9.0909…% → +9.1%
    fills = [
        Fill("buy",  "AAPL", Decimal("10"), Decimal("100"), Decimal("0")),
        Fill("buy",  "AAPL", Decimal("10"), Decimal("110"), Decimal("0")),
        Fill("sell", "AAPL", Decimal("15"), Decimal("120"), Decimal("0")),
    ]
    # Cash: 10000 - 1000 - 1100 + 1800 = 9700. 5 shares of lot2 left open @110.
    # final_positions[AAPL] = 5; final_mark @ 110 (cost) so position contributes
    # 550, ending equity = 9700 + 550 = 10250, total return = +2.5%.
    result = compute_perf_metrics(
        initial_cash=Decimal("10000"),
        fills=fills,
        final_cash=Decimal("9700"),
        final_positions={"AAPL": Decimal("5")},
        final_marks={"AAPL": Decimal("110")},
        spy_return=0.0,
        basket_return=0.0,
    )
    assert result["per_trade_returns_str"] == "+20.0%, +9.1%"
    assert (
        result["hit_rate_str"]
        == "2/2 (100%) — n=2, not statistically significant"
    )
    assert result["total_return_pct"] == 2.5


def test_hit_rate_zero_trades() -> None:
    result = compute_perf_metrics(
        initial_cash=Decimal("10000"),
        fills=[],
        final_cash=Decimal("10000"),
        final_positions={},
        final_marks={},
        spy_return=0.0,
        basket_return=0.0,
    )
    assert result["hit_rate_str"] == "0/0 (n/a) — no closed trades"
    assert result["per_trade_returns_str"] == "(no closed trades)"


def test_commission_included_in_per_trade_return() -> None:
    # BUY 100 AAPL @ 100, commission 50  → eff buy/share = (100×100 + 50)/100 = 100.5
    # SELL 100 AAPL @ 110, commission 55 → eff sell/share = (110×100 − 55)/100 = 109.45
    # Return = (109.45 − 100.5) / 100.5 × 100 = 8.9054…% → +8.9%
    fills = [
        Fill("buy",  "AAPL", Decimal("100"), Decimal("100"), Decimal("50")),
        Fill("sell", "AAPL", Decimal("100"), Decimal("110"), Decimal("55")),
    ]
    # Cash math irrelevant to per-trade assertion; pin something consistent.
    # Spent 100×100 + 50 = 10050. Received 100×110 − 55 = 10945.
    # final_cash = 10000 − 10050 + 10945 = 10895.
    result = compute_perf_metrics(
        initial_cash=Decimal("10000"),
        fills=fills,
        final_cash=Decimal("10895"),
        final_positions={},
        final_marks={},
        spy_return=0.0,
        basket_return=0.0,
    )
    assert result["per_trade_returns_str"] == "+8.9%"


def test_validation_negative_initial_cash_raises() -> None:
    with pytest.raises(ValueError, match="initial_cash must be positive"):
        compute_perf_metrics(
            initial_cash=Decimal("-1"),
            fills=[],
            final_cash=Decimal("0"),
            final_positions={},
            final_marks={},
            spy_return=0.0,
            basket_return=0.0,
        )
    with pytest.raises(ValueError, match="initial_cash must be positive"):
        compute_perf_metrics(
            initial_cash=Decimal("0"),
            fills=[],
            final_cash=Decimal("0"),
            final_positions={},
            final_marks={},
            spy_return=0.0,
            basket_return=0.0,
        )


def test_validation_missing_final_mark_raises() -> None:
    with pytest.raises(ValueError, match="missing final mark for ticker 'AAPL'"):
        compute_perf_metrics(
            initial_cash=Decimal("10000"),
            fills=[],
            final_cash=Decimal("5000"),
            final_positions={"AAPL": Decimal("50")},
            final_marks={},
            spy_return=0.0,
            basket_return=0.0,
        )


def test_validation_sell_without_buy_raises() -> None:
    fills = [
        Fill("sell", "AAPL", Decimal("10"), Decimal("100"), Decimal("0")),
    ]
    with pytest.raises(ValueError, match="sell of AAPL with no preceding buy"):
        compute_perf_metrics(
            initial_cash=Decimal("10000"),
            fills=fills,
            final_cash=Decimal("11000"),
            final_positions={},
            final_marks={},
            spy_return=0.0,
            basket_return=0.0,
        )


def test_dict_value_types_are_strictly_float_or_str() -> None:
    fills = [
        Fill("buy",  "AAPL", Decimal("10"), Decimal("100"), Decimal("0")),
        Fill("sell", "AAPL", Decimal("10"), Decimal("110"), Decimal("0")),
    ]
    result = compute_perf_metrics(
        initial_cash=Decimal("10000"),
        fills=fills,
        final_cash=Decimal("10100"),
        final_positions={},
        final_marks={},
        spy_return=0.02,
        basket_return=-0.01,
    )
    # Enforce the dict[str, float | str] contract: no ints, no Decimals, no lists.
    for key, value in result.items():
        assert isinstance(key, str), key
        assert isinstance(value, (float, str)), (key, type(value), value)
        # bool is a subclass of int, but also not float; guard explicitly.
        assert not isinstance(value, bool), (key, value)


def test_vs_spy_and_vs_basket_are_pp_deltas() -> None:
    # Pin spec §9.7's sample numbers: total=−1.2%, spy=+0.8%, basket=−0.4%.
    # vs_spy_pp = −1.2 − 0.8 = −2.0; vs_basket_pp = −1.2 − (−0.4) = −0.8.
    # Achieve total=−1.2% via final_cash=9880 against initial=10000.
    result = compute_perf_metrics(
        initial_cash=Decimal("10000"),
        fills=[],
        final_cash=Decimal("9880"),
        final_positions={},
        final_marks={},
        spy_return=0.008,
        basket_return=-0.004,
    )
    assert result["total_return_pct"] == -1.2
    assert result["spy_return_pct"] == 0.8
    assert result["basket_return_pct"] == -0.4
    assert result["vs_spy_pp"] == -2.0
    assert result["vs_basket_pp"] == -0.8
