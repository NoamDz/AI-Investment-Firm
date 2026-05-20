"""MCP-style tool for point-in-time fundamental ratio lookups.

This module wraps a pre-computed parquet file of ``(ticker, ratio_name,
as_of, value)`` rows and exposes a single ``get_ratio`` method that performs
a point-in-time (PIT) lookup: it returns the value from the latest filing
whose ``as_of`` date is on or before the requested ``as_of`` date.

Design rationale (spec §7.3): the LLM must NOT perform arithmetic. All ratio
values are pre-computed by ``scripts/precompute_fundamentals.py`` and stored
as ``Decimal`` strings in the parquet. The LLM calls this tool by name
(``fundamentals.get_ratio``) via the Anthropic ``tools=`` parameter; it
receives a ``Decimal`` value back without ever computing it.

PIT contract: "latest filing on or before as_of". If no filing exists on or
before the requested date, a ``KeyError`` is raised so the agent can surface
an ``INSUFFICIENT_EVIDENCE`` failure mode rather than silently returning a
stale or wrong value.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

import pyarrow.parquet as pq  # type: ignore[import-untyped]

_SUPPORTED_RATIOS = [
    "pe_ratio",
    "gross_margin",
    "revenue_yoy_growth",
    "debt_to_equity",
    "current_ratio",
]

# Type alias for the in-memory index: (ticker, ratio_name) -> sorted list of
# (as_of_date, Decimal_value) ordered by date ascending.
_Index = dict[tuple[str, str], list[tuple[date, Decimal]]]


@dataclass(frozen=True)
class ToolDef:
    """Serialisable descriptor for an MCP / Anthropic tool= payload entry."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


class FundamentalsTool:
    """Point-in-time fundamental ratio lookup backed by a pre-computed parquet.

    The parquet is read once at ``__init__`` and held in memory as a dict of
    sorted ``(date, Decimal)`` lists keyed by ``(ticker, ratio_name)``.

    Usage
    -----
    >>> tool = FundamentalsTool(Path("data/precomputed/fundamentals.parquet"))
    >>> tool.get_ratio(ticker="AAPL", ratio_name="pe_ratio", as_of=date(2024, 11, 1))
    Decimal('28.5')
    """

    tool_def: ClassVar[ToolDef] = ToolDef(
        name="fundamentals.get_ratio",
        description=(
            "Return a pre-computed fundamental ratio for a publicly-traded ticker "
            "as of a given date, using point-in-time (PIT) semantics: the value from "
            "the latest SEC filing on or before the requested date is returned. "
            "Supported ratios: pe_ratio, gross_margin, revenue_yoy_growth, "
            "debt_to_equity, current_ratio. Raises if no data is available."
        ),
        input_schema=MappingProxyType(
            {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Exchange ticker symbol, e.g. AAPL or NVDA.",
                    },
                    "ratio_name": {
                        "type": "string",
                        "enum": list(_SUPPORTED_RATIOS),
                        "description": (
                            "Name of the fundamental ratio to retrieve. "
                            "One of: pe_ratio, gross_margin, revenue_yoy_growth, "
                            "debt_to_equity, current_ratio."
                        ),
                    },
                    "as_of": {
                        "type": "string",
                        "format": "date",
                        "description": (
                            "ISO 8601 date YYYY-MM-DD; PIT lookup uses the latest "
                            "filing on or before this date."
                        ),
                    },
                },
                "required": ["ticker", "ratio_name", "as_of"],
            }
        ),
    )

    def __init__(self, parquet_path: Path) -> None:
        """Load the parquet and build the in-memory PIT index.

        Parameters
        ----------
        parquet_path:
            Path to the pre-computed parquet written by
            ``scripts/precompute_fundamentals.py``.
        """
        table = pq.read_table(str(parquet_path))
        index: _Index = {}

        tickers = table.column("ticker").to_pylist()
        ratio_names = table.column("ratio_name").to_pylist()
        as_of_dates = table.column("as_of").to_pylist()
        values = table.column("value").to_pylist()

        for ticker_val, ratio_val, as_of_val, value_val in zip(
            tickers, ratio_names, as_of_dates, values
        ):
            key = (str(ticker_val), str(ratio_val))
            # as_of column is stored as date32 -> python date
            if isinstance(as_of_val, date):
                as_of_date = as_of_val
            else:
                # fallback: parse string
                as_of_date = date.fromisoformat(str(as_of_val))
            # Fail-fast if the parquet column type drifts away from string;
            # see scripts/precompute_fundamentals.py: value is stored as str(Decimal)
            # specifically so we can avoid float-precision loss on read.
            assert isinstance(value_val, str), (
                f"value column must be string, got {type(value_val)}"
            )
            # Preserve precision by going through str representation
            decimal_value = Decimal(value_val)
            if key not in index:
                index[key] = []
            index[key].append((as_of_date, decimal_value))

        # Sort each list by date ascending to enable binary-search PIT lookup.
        for entries in index.values():
            entries.sort(key=lambda pair: pair[0])

        self._index = index

    def get_ratio(
        self, *, ticker: str, ratio_name: str, as_of: date | str
    ) -> Decimal:
        """Return the fundamental ratio value for the given PIT parameters.

        Performs a PIT lookup: returns the value from the latest filing whose
        ``as_of`` date is on or before the requested ``as_of`` date.

        Parameters
        ----------
        ticker:
            Exchange ticker symbol (case-sensitive, e.g. ``"AAPL"``).
        ratio_name:
            One of the supported ratio names (see ``tool_def.input_schema``).
        as_of:
            The point-in-time date for the lookup. Accepts either a
            :class:`datetime.date` or an ISO 8601 date string (``YYYY-MM-DD``).
            The string form supports direct invocation from an Anthropic
            ``tool_use`` block, where ``tool_input["as_of"]`` is a JSON string.

        Returns
        -------
        Decimal
            The ratio value. Never a float — Decimal preserves the precision
            stored in the pre-computed parquet.

        Raises
        ------
        KeyError
            If ``ticker`` is unknown, ``ratio_name`` is unknown for the ticker,
            or no filing exists on or before ``as_of``.
        """
        if isinstance(as_of, str):
            as_of = date.fromisoformat(as_of)
        # Check ticker presence across all ratio names first.
        ticker_keys = [k for k in self._index if k[0] == ticker]
        if not ticker_keys:
            raise KeyError(
                f"Ticker {ticker!r} not found in fundamentals index. "
                "Run scripts/precompute_fundamentals.py to refresh the parquet."
            )

        key = (ticker, ratio_name)
        if key not in self._index:
            raise KeyError(
                f"ratio_name {ratio_name!r} not available for ticker {ticker!r}. "
                f"Available ratios for {ticker!r}: "
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
                f"No filing for ({ticker!r}, {ratio_name!r}) on or before {as_of.isoformat()}. "
                f"Earliest available date: {entries[0][0].isoformat()}"
            )

        return best

    def run(self, **kwargs: Any) -> Decimal:
        """Generic dispatch shim — delegates to :meth:`get_ratio`.

        Allows the extractor to call tools uniformly via ``tool.run(**block["input"])``
        without knowing whether the tool is a ``FundamentalsTool`` or ``RiskMetricsTool``.
        """
        return self.get_ratio(**kwargs)


__all__ = ["FundamentalsTool", "ToolDef"]
