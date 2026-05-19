"""risk.get_metric MCP-style tool with PIT lookup.

Plan 2 §T23. The LLM is forbidden from computing risk metrics itself
(spec §7.3); this tool wraps a pre-computed parquet of
(ticker, metric_id, as_of, value) rows. The window is baked into the
metric_id (e.g., ``volatility_30d`` is a 30-day rolling stdev).

The PIT contract mirrors :class:`firm.tools.fundamentals.FundamentalsTool`:
:meth:`get_metric` selects the latest row with ``as_of_date <= as_of``.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from firm.tools.fundamentals import ToolDef

_SUPPORTED_METRICS: tuple[str, ...] = (
    "volatility_30d",
    "beta_180d",
    "max_drawdown_90d",
)

# Type alias for the in-memory index: (ticker, metric_id) -> sorted list of
# (as_of_date, Decimal_value) ordered by date ascending.
_Index = dict[tuple[str, str], list[tuple[date, Decimal]]]


class RiskMetricsTool:
    """Point-in-time risk metric lookup backed by a pre-computed parquet.

    The parquet is read once at ``__init__`` and held in memory as a dict of
    sorted ``(date, Decimal)`` lists keyed by ``(ticker, metric_id)``.

    Usage
    -----
    >>> tool = RiskMetricsTool(Path("data/precomputed/risk_metrics.parquet"))
    >>> tool.get_metric(ticker="AAPL", metric="volatility_30d", as_of=date(2024, 11, 1))
    Decimal('0.21')
    """

    tool_def: ClassVar[ToolDef] = ToolDef(
        name="risk.get_metric",
        description=(
            "Return a pre-computed risk metric for a publicly-traded ticker "
            "as of a given date, using point-in-time (PIT) semantics: the value from "
            "the latest precomputed row on or before the requested date is returned. "
            "Supported metrics: volatility_30d, beta_180d, max_drawdown_90d. "
            "The window is baked into the metric name (e.g., volatility_30d is a "
            "30-day rolling standard deviation). Raises if no data is available."
        ),
        input_schema=MappingProxyType(
            {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Exchange ticker symbol, e.g. AAPL or NVDA.",
                    },
                    "metric": {
                        "type": "string",
                        "enum": list(_SUPPORTED_METRICS),
                        "description": (
                            "Name of the risk metric to retrieve. "
                            "One of: volatility_30d, beta_180d, max_drawdown_90d."
                        ),
                    },
                    "as_of": {
                        "type": "string",
                        "format": "date",
                        "description": (
                            "ISO 8601 date YYYY-MM-DD; PIT lookup uses the latest "
                            "precomputed row on or before this date."
                        ),
                    },
                },
                "required": ["ticker", "metric", "as_of"],
            }
        ),
    )

    def __init__(self, parquet_path: Path) -> None:
        """Load the parquet and build the in-memory PIT index.

        Parameters
        ----------
        parquet_path:
            Path to the pre-computed parquet written by
            ``scripts/precompute_risk_metrics.py``.
        """
        table = pq.read_table(str(parquet_path))
        index: _Index = {}

        tickers = table.column("ticker").to_pylist()
        metric_ids = table.column("metric_id").to_pylist()
        as_of_dates = table.column("as_of").to_pylist()
        values = table.column("value").to_pylist()

        for ticker_val, metric_val, as_of_val, value_val in zip(
            tickers, metric_ids, as_of_dates, values
        ):
            key = (str(ticker_val), str(metric_val))
            # as_of column is stored as date32 -> python date
            if isinstance(as_of_val, date):
                as_of_date = as_of_val
            else:
                # fallback: parse string
                as_of_date = date.fromisoformat(str(as_of_val))
            # Fail-fast if the parquet column type drifts away from string;
            # value is stored as str(Decimal) to avoid float-precision loss on read.
            assert isinstance(value_val, str), (
                f"value column must be string, got {type(value_val)}"
            )
            decimal_value = Decimal(value_val)
            if key not in index:
                index[key] = []
            index[key].append((as_of_date, decimal_value))

        # Sort each list by date ascending to enable PIT lookup.
        for entries in index.values():
            entries.sort(key=lambda pair: pair[0])

        self._index = index

    def get_metric(
        self, *, ticker: str, metric: str, as_of: date | str
    ) -> Decimal:
        """Return the risk metric value for the given PIT parameters.

        Performs a PIT lookup: returns the value from the latest precomputed
        row whose ``as_of`` date is on or before the requested ``as_of`` date.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol (case-sensitive, e.g. ``"AAPL"``).
        metric:
            One of the supported metric IDs (see ``tool_def.input_schema``).
        as_of:
            The point-in-time date for the lookup. Accepts either a
            :class:`datetime.date` or an ISO 8601 date string (``YYYY-MM-DD``).
            The string form supports direct invocation from an Anthropic
            ``tool_use`` block, where ``tool_input["as_of"]`` is a JSON string.

        Returns
        -------
        Decimal
            The metric value. Never a float — Decimal preserves the precision
            stored in the pre-computed parquet.

        Raises
        ------
        KeyError
            If ``ticker`` is unknown, ``metric`` is unknown for the ticker,
            or no row exists on or before ``as_of``.
        """
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)
        # Check ticker presence across all metric names first.
        ticker_keys = [k for k in self._index if k[0] == ticker]
        if not ticker_keys:
            raise KeyError(
                f"Ticker {ticker!r} not found in risk metrics index. "
                "Run scripts/precompute_risk_metrics.py to refresh the parquet."
            )

        key = (ticker, metric)
        if key not in self._index:
            raise KeyError(
                f"metric {metric!r} not available for ticker {ticker!r}. "
                f"Available metrics for {ticker!r}: "
                f"{sorted({k[1] for k in ticker_keys})}"
            )

        entries = self._index[key]
        # Walk backwards through date-sorted entries to find the latest on or before as_of.
        best: Decimal | None = None
        for filing_date, value in reversed(entries):
            if filing_date <= as_of:
                best = value
                break

        if best is None:
            raise KeyError(
                f"No row for ({ticker!r}, {metric!r}) on or before {as_of.isoformat()}. "
                f"Earliest available date: {entries[0][0].isoformat()}"
            )

        return best


__all__ = ["RiskMetricsTool", "ToolDef"]
