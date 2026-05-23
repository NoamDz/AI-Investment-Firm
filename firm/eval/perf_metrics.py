"""Per-regime performance metrics for the eval report (Plan 4 §T11).

Turns broker fills + cash deltas into the structured dict consumed by the eval
report Jinja template. Output shape mirrors spec §9.7's regime summary:

* total / SPY / basket returns (and pp deltas) as 1-dp floats
* per-trade returns as a pre-formatted, signed-percent string
* hit rate as a pre-formatted ``winners/total (pct%) — n=…`` string

Per-trade matching is FIFO against same-ticker buys (not LIFO or avg-cost) so
the regime walk-forward narrative matches the chronological order in which
decisions were made; partial sells split a single sell into one per-trade
entry per consumed buy lot, prorating that sell's commission by share count.
Commission is folded into the effective buy cost-per-share and effective sell
proceeds-per-share so a positive return reflects net P&L, not gross price
movement.

The runner must inject a synthetic ``"buy"`` Fill at regime start for any
position carried over from a prior regime; this module never reads positions
outside the ``fills`` stream.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Mapping, Sequence

_ONE_DP = Decimal("0.1")
_HUNDRED = Decimal("100")


@dataclass(frozen=True)
class Fill:
    """One executed broker fill, derived from a ``broker.OrderResult`` plus the
    originating decision side.

    Attributes
    ----------
    side        : ``"buy"`` or ``"sell"`` — direction of the originating order.
    ticker      : symbol, case-sensitive.
    shares      : strictly positive share count (direction lives in ``side``).
    fill_price  : per-share fill price, already adjusted for slippage.
    commission  : absolute commission for the fill (e.g. 0.5%×gross), in the
                  same currency as ``fill_price`` × ``shares``.
    """

    side: Literal["buy", "sell"]
    ticker: str
    shares: Decimal
    fill_price: Decimal
    commission: Decimal


@dataclass
class _Lot:
    """Internal FIFO lot tracker — remaining shares + the original buy fill's
    per-share gross price and per-share commission allocation."""

    shares_remaining: Decimal
    buy_price: Decimal
    # Per-share commission of the original buy, captured at buy time so partial
    # consumption splits commission proportionally without needing to remember
    # the full original fill.
    buy_commission_per_share: Decimal


def _round_1dp(value: Decimal) -> float:
    """Quantize *value* to one decimal place and return as ``float``.

    The Decimal→float conversion happens only at the dict boundary so internal
    arithmetic stays exact.
    """
    return float(value.quantize(_ONE_DP))


def _format_per_trade(returns_pct: Sequence[Decimal]) -> str:
    if not returns_pct:
        return "(no closed trades)"
    return ", ".join(f"{_round_1dp(r):+.1f}%" for r in returns_pct)


def _format_hit_rate(returns_pct: Sequence[Decimal]) -> str:
    total = len(returns_pct)
    if total == 0:
        return "0/0 (n/a) — no closed trades"
    # Tied-at-exactly-zero counts as a loss (spec): strict ``>`` comparison.
    winners = sum(1 for r in returns_pct if r > 0)
    pct = round(winners / total * 100)
    return f"{winners}/{total} ({pct}%) — n={total}, not statistically significant"


def compute_perf_metrics(
    *,
    initial_cash: Decimal,
    fills: Sequence[Fill],
    final_cash: Decimal,
    final_positions: Mapping[str, Decimal],
    final_marks: Mapping[str, Decimal],
    spy_return: float,
    basket_return: float,
) -> dict[str, float | str]:
    """Compute the per-regime metrics dict for the eval report.

    Returns a ``dict[str, float | str]`` matching spec §9.7's sample shape:

    ``total_return_pct``, ``spy_return_pct``, ``basket_return_pct``,
    ``vs_spy_pp``, ``vs_basket_pp`` as 1-dp floats;
    ``per_trade_returns_str`` and ``hit_rate_str`` as pre-formatted strings.

    Parameters are keyword-only to keep call sites self-documenting.

    Raises
    ------
    ValueError
        * ``initial_cash <= 0``
        * any fill has negative ``shares``
        * a non-zero final position lacks an entry in ``final_marks``
        * a sell has no preceding same-ticker buy (the runner must inject a
          synthetic regime-start buy for carried positions; see module docs)
    """
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")

    # ----- Validate fills up front so per-trade loop assumes well-formed input.
    for f in fills:
        if f.shares < 0:
            raise ValueError(
                f"negative shares in fill for {f.ticker!r}: {f.shares}"
            )

    # ----- Total return: ending equity vs initial cash.
    positions_value = Decimal("0")
    for ticker, shares in final_positions.items():
        if shares == 0:
            # Zero-share entries are harmless and need no mark.
            continue
        if ticker not in final_marks:
            raise ValueError(f"missing final mark for ticker {ticker!r}")
        positions_value += shares * final_marks[ticker]

    ending_equity = final_cash + positions_value
    total_return_pct_dec = (ending_equity - initial_cash) / initial_cash * _HUNDRED

    # ----- Per-trade FIFO match: each sell consumes earliest same-ticker lots.
    lots_by_ticker: dict[str, deque[_Lot]] = {}
    per_trade_returns: list[Decimal] = []

    for f in fills:
        if f.side == "buy":
            # Capture per-share commission so partial consumption later can
            # allocate the buy's commission pro-rata by shares consumed.
            buy_comm_per_share = (
                f.commission / f.shares if f.shares > 0 else Decimal("0")
            )
            lots_by_ticker.setdefault(f.ticker, deque()).append(
                _Lot(
                    shares_remaining=f.shares,
                    buy_price=f.fill_price,
                    buy_commission_per_share=buy_comm_per_share,
                )
            )
            continue

        # f.side == "sell"
        lots = lots_by_ticker.get(f.ticker)
        if not lots:
            raise ValueError(f"sell of {f.ticker} with no preceding buy")

        remaining_to_match = f.shares
        # Prorate the sell's own commission per share matched in this iteration.
        sell_comm_per_share = (
            f.commission / f.shares if f.shares > 0 else Decimal("0")
        )

        while remaining_to_match > 0:
            if not lots:
                raise ValueError(f"sell of {f.ticker} with no preceding buy")
            lot = lots[0]
            matched = min(lot.shares_remaining, remaining_to_match)

            # Effective per-share cost / proceeds INCLUDE the matched lot's
            # share of commission. Doing the split per-lot (rather than once at
            # fill level) is what makes partial-consumption per-trade returns
            # commission-correct without double-counting.
            effective_buy_per_share = lot.buy_price + lot.buy_commission_per_share
            effective_sell_per_share = f.fill_price - sell_comm_per_share

            trade_return_pct = (
                (effective_sell_per_share - effective_buy_per_share)
                / effective_buy_per_share
                * _HUNDRED
            )
            per_trade_returns.append(trade_return_pct)

            lot.shares_remaining -= matched
            remaining_to_match -= matched
            if lot.shares_remaining == 0:
                lots.popleft()

    # ----- Benchmark deltas: round inputs first so dict arithmetic is exact.
    spy_pct_dec = Decimal(str(spy_return)) * _HUNDRED
    basket_pct_dec = Decimal(str(basket_return)) * _HUNDRED

    total_return_pct = _round_1dp(total_return_pct_dec)
    spy_return_pct = _round_1dp(spy_pct_dec)
    basket_return_pct = _round_1dp(basket_pct_dec)
    # vs_*_pp uses already-rounded values so total - spy = vs_spy_pp holds in
    # the emitted dict; spec §9.7's sample numbers rely on this identity.
    vs_spy_pp = _round_1dp(
        Decimal(str(total_return_pct)) - Decimal(str(spy_return_pct))
    )
    vs_basket_pp = _round_1dp(
        Decimal(str(total_return_pct)) - Decimal(str(basket_return_pct))
    )

    return {
        "total_return_pct": total_return_pct,
        "spy_return_pct": spy_return_pct,
        "basket_return_pct": basket_return_pct,
        "vs_spy_pp": vs_spy_pp,
        "vs_basket_pp": vs_basket_pp,
        "per_trade_returns_str": _format_per_trade(per_trade_returns),
        "hit_rate_str": _format_hit_rate(per_trade_returns),
    }


__all__ = ["Fill", "compute_perf_metrics"]
