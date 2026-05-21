"""SPY and equal-weight basket return benchmarks (Plan 4 §T10).

Provides ``compute_spy_return`` and ``compute_basket_return`` for the eval
harness. Both return the total return over a date window as a float
(``last_adj_close / first_adj_close - 1.0``).

Price-cache layer
-----------------
Adjusted closes are cached as parquet files under
``data/eval/prices/<TICKER>.parquet`` with two columns:

* ``date``      — string ``YYYY-MM-DD`` (sorted ascending)
* ``adj_close`` — float64

Mode dispatch is driven by the ``FIRM_PRICES_MODE`` env var (kept orthogonal
to ``FIRM_VCR_MODE``, which governs LLM cassettes):

* ``replay`` (default): require ``<TICKER>.parquet`` to exist; otherwise
  raise :class:`PriceCassetteMissError`. Never touches the network.
* ``record``: if the parquet is missing, fetch from ``yfinance`` once and
  persist it. If the parquet already exists it is read as-is — re-recording
  requires an explicit ``unlink`` first so everyday code paths never silently
  re-hit the network. The eval-capture script (T16) owns the re-record flow.
* ``live``: bypass the cache entirely; always call yfinance. Used only by
  ``scripts/eval_capture.py`` in T16.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class PricesMode(StrEnum):
    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"


class PriceCassetteMissError(Exception):
    """Raised in REPLAY mode when a ticker's parquet is absent."""


def _default_prices_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "eval" / "prices"


def _resolve_mode(mode: PricesMode | None) -> PricesMode:
    if mode is not None:
        return mode
    raw = os.environ.get("FIRM_PRICES_MODE", PricesMode.REPLAY.value)
    return PricesMode(raw)


def _read_parquet_series(path: Path) -> list[tuple[date, float]]:
    table = pq.read_table(str(path))  # type: ignore[no-untyped-call]
    dates = table.column("date").to_pylist()
    closes = table.column("adj_close").to_pylist()
    rows: list[tuple[date, float]] = []
    for d_raw, c_raw in zip(dates, closes):
        d = d_raw if isinstance(d_raw, date) else date.fromisoformat(str(d_raw))
        rows.append((d, float(c_raw)))
    rows.sort(key=lambda pair: pair[0])
    return rows


def _write_parquet_series(path: Path, rows: Sequence[tuple[date, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "date": pa.array([d.isoformat() for d, _ in rows], type=pa.string()),
            "adj_close": pa.array([c for _, c in rows], type=pa.float64()),
        }
    )
    pq.write_table(table, str(path))  # type: ignore[no-untyped-call]


def _fetch_yfinance(ticker: str, start: date, end: date) -> list[tuple[date, float]]:
    # Imported lazily so replay-mode tests don't pay the import cost and so
    # the dep is only required when actually fetching.
    import yfinance  # type: ignore[import-untyped]

    # yfinance treats ``end`` as exclusive; bump by one day so the requested
    # window is inclusive on both sides.
    df = yfinance.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
    )
    if df is None or len(df) == 0:
        raise RuntimeError(
            f"yfinance returned no rows for {ticker} {start}..{end}"
        )
    adj_col = df["Adj Close"]
    # MultiIndex columns appear when yfinance is asked for one ticker but
    # still nests under ticker; collapse to the single series.
    if hasattr(adj_col, "columns"):
        adj_col = adj_col.iloc[:, 0]
    rows: list[tuple[date, float]] = []
    for idx, value in adj_col.items():
        d = idx.date() if hasattr(idx, "date") else date.fromisoformat(str(idx))
        rows.append((d, float(value)))
    rows.sort(key=lambda pair: pair[0])
    return rows


def _load_ticker_series(
    ticker: str,
    start: date,
    end: date,
    prices_dir: Path,
    mode: PricesMode,
) -> list[tuple[date, float]]:
    if mode == PricesMode.LIVE:
        return _fetch_yfinance(ticker, start, end)

    parquet_path = prices_dir / f"{ticker}.parquet"
    if parquet_path.exists():
        return _read_parquet_series(parquet_path)

    if mode == PricesMode.REPLAY:
        raise PriceCassetteMissError(
            f"missing price cassette for ticker {ticker!r} at {parquet_path}"
        )

    # RECORD with no cached parquet: fetch + persist + read back.
    rows = _fetch_yfinance(ticker, start, end)
    _write_parquet_series(parquet_path, rows)
    return _read_parquet_series(parquet_path)


def _ticker_total_return(
    ticker: str,
    start: date,
    end: date,
    prices_dir: Path,
    mode: PricesMode,
) -> float:
    series = _load_ticker_series(ticker, start, end, prices_dir, mode)
    in_window = [(d, c) for d, c in series if start <= d <= end]
    if not in_window:
        raise ValueError(
            f"no trading days in window [{start}, {end}] for ticker {ticker!r}"
        )
    first_close = in_window[0][1]
    last_close = in_window[-1][1]
    if first_close == 0.0:
        raise ValueError(
            f"first adj_close for {ticker!r} on {in_window[0][0]} is zero; "
            "cannot compute return"
        )
    return last_close / first_close - 1.0


def compute_basket_return(
    tickers: Sequence[str],
    start: date,
    end: date,
    *,
    prices_dir: Path | None = None,
    mode: PricesMode | None = None,
) -> float:
    """Return the equal-weight arithmetic-mean total return for *tickers*.

    Each ticker's return is computed over ``[start, end]`` inclusive using the
    first and last trading-day adjusted closes within the window. The basket
    return is the arithmetic mean of those per-ticker returns.

    Raises
    ------
    ValueError
        If *tickers* is empty, or if any ticker has no trading days inside
        ``[start, end]``.
    PriceCassetteMissError
        In REPLAY mode, if any ticker's parquet cassette is missing.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty sequence")
    resolved_dir = prices_dir if prices_dir is not None else _default_prices_dir()
    resolved_mode = _resolve_mode(mode)
    returns = [
        _ticker_total_return(t, start, end, resolved_dir, resolved_mode)
        for t in tickers
    ]
    return sum(returns) / len(returns)


def compute_spy_return(
    start: date,
    end: date,
    *,
    prices_dir: Path | None = None,
    mode: PricesMode | None = None,
) -> float:
    """Return the SPY total return over ``[start, end]`` (thin wrapper)."""
    return compute_basket_return(
        ["SPY"], start, end, prices_dir=prices_dir, mode=mode
    )


__all__ = [
    "PriceCassetteMissError",
    "PricesMode",
    "compute_basket_return",
    "compute_spy_return",
]
