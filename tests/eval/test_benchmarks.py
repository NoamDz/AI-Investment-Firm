"""Tests for ``firm.eval.benchmarks`` (Plan 4 §T10).

All tests run in REPLAY mode against hand-written parquets in ``tmp_path``;
no test calls yfinance.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from firm.eval.benchmarks import (
    PriceCassetteMissError,
    PricesMode,
    compute_basket_return,
    compute_spy_return,
)


def _write_parquet(
    prices_dir: Path,
    ticker: str,
    rows: Sequence[tuple[str, float]],
) -> Path:
    prices_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "date": pa.array([d for d, _ in rows], type=pa.string()),
            "adj_close": pa.array([c for _, c in rows], type=pa.float64()),
        }
    )
    path = prices_dir / f"{ticker}.parquet"
    pq.write_table(table, str(path))  # type: ignore[no-untyped-call]
    return path


def test_compute_basket_return_single_ticker_replay(tmp_path: Path) -> None:
    prices_dir = tmp_path / "prices"
    _write_parquet(
        prices_dir,
        "TEST",
        [
            ("2024-03-11", 100.0),
            ("2024-03-12", 105.0),
            ("2024-03-13", 110.0),
        ],
    )
    result = compute_basket_return(
        ["TEST"],
        start=date(2024, 3, 11),
        end=date(2024, 3, 13),
        prices_dir=prices_dir,
        mode=PricesMode.REPLAY,
    )
    assert result == pytest.approx(0.10)


def test_compute_basket_return_equal_weight(tmp_path: Path) -> None:
    prices_dir = tmp_path / "prices"
    # Ticker A: 100 -> 110 = +10%
    _write_parquet(
        prices_dir,
        "A",
        [("2024-03-11", 100.0), ("2024-03-13", 110.0)],
    )
    # Ticker B: 50 -> 65 = +30%
    _write_parquet(
        prices_dir,
        "B",
        [("2024-03-11", 50.0), ("2024-03-13", 65.0)],
    )
    result = compute_basket_return(
        ["A", "B"],
        start=date(2024, 3, 11),
        end=date(2024, 3, 13),
        prices_dir=prices_dir,
        mode=PricesMode.REPLAY,
    )
    # Arithmetic mean of +10% and +30% = +20%
    assert result == pytest.approx(0.20)


def test_compute_basket_return_empty_tickers_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_basket_return(
            [],
            start=date(2024, 3, 11),
            end=date(2024, 3, 13),
            mode=PricesMode.REPLAY,
        )


def test_compute_basket_return_replay_miss_raises(tmp_path: Path) -> None:
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    with pytest.raises(PriceCassetteMissError) as excinfo:
        compute_basket_return(
            ["AAPL"],
            start=date(2024, 3, 11),
            end=date(2024, 3, 13),
            prices_dir=prices_dir,
            mode=PricesMode.REPLAY,
        )
    message = str(excinfo.value)
    assert "AAPL" in message
    assert str(prices_dir / "AAPL.parquet") in message


def test_compute_spy_return_delegates_to_basket(tmp_path: Path) -> None:
    prices_dir = tmp_path / "prices"
    _write_parquet(
        prices_dir,
        "SPY",
        [
            ("2024-03-11", 500.0),
            ("2024-03-12", 510.0),
            ("2024-03-13", 525.0),
        ],
    )
    result = compute_spy_return(
        start=date(2024, 3, 11),
        end=date(2024, 3, 13),
        prices_dir=prices_dir,
        mode=PricesMode.REPLAY,
    )
    # 525 / 500 - 1 = 0.05
    assert result == pytest.approx(0.05)


def test_compute_basket_return_uses_window_boundary_trading_days(
    tmp_path: Path,
) -> None:
    prices_dir = tmp_path / "prices"
    # Rows outside the window (before and after) plus rows inside.
    _write_parquet(
        prices_dir,
        "WIN",
        [
            ("2024-03-08", 80.0),   # outside (before)
            ("2024-03-11", 100.0),  # in-window first
            ("2024-03-12", 105.0),
            ("2024-03-13", 120.0),  # in-window last
            ("2024-03-15", 200.0),  # outside (after)
        ],
    )
    result = compute_basket_return(
        ["WIN"],
        start=date(2024, 3, 11),
        end=date(2024, 3, 13),
        prices_dir=prices_dir,
        mode=PricesMode.REPLAY,
    )
    # Must ignore the 80.0 and 200.0 rows; 120/100 - 1 = 0.20
    assert result == pytest.approx(0.20)


def test_compute_basket_return_no_trading_days_in_window_raises(
    tmp_path: Path,
) -> None:
    prices_dir = tmp_path / "prices"
    # Parquet has data, but none in the requested window.
    _write_parquet(
        prices_dir,
        "GAP",
        [
            ("2024-01-01", 100.0),
            ("2024-12-31", 200.0),
        ],
    )
    with pytest.raises(ValueError, match="no trading days in window"):
        compute_basket_return(
            ["GAP"],
            start=date(2024, 6, 1),
            end=date(2024, 6, 5),
            prices_dir=prices_dir,
            mode=PricesMode.REPLAY,
        )
