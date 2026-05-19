"""Pre-compute fundamental ratios and write a parquet of (ticker, ratio_name, as_of, value).

Fixture mode (FIRM_FUNDAMENTALS_FIXTURE=1):
    Reads tests/fixtures/financebench_two_docs.json and writes a deterministic
    parquet to tests/fixtures/fundamentals_fixture.parquet.  The fixture values
    are hard-coded so unit tests get a stable, reproducible parquet without
    depending on real ratio extraction from filing tables.

Real corpus mode (default):
    # TODO(plan-3): extract real ratios from filing tables
    Raises NotImplementedError — Plan 2 only requires the fixture path.

Usage
-----
    # fixture mode (used by tests):
    FIRM_FUNDAMENTALS_FIXTURE=1 python scripts/precompute_fundamentals.py

    # real corpus (not yet implemented):
    python scripts/precompute_fundamentals.py
"""
from __future__ import annotations

import os
from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_FIXTURE_JSON = _REPO_ROOT / "tests" / "fixtures" / "financebench_two_docs.json"
_FIXTURE_PARQUET = _REPO_ROOT / "tests" / "fixtures" / "fundamentals_fixture.parquet"
_REAL_PARQUET = _REPO_ROOT / "data" / "precomputed" / "fundamentals.parquet"

# ---------------------------------------------------------------------------
# Fixture data (deterministic hard-coded rows)
# ---------------------------------------------------------------------------
# Main rows from the two FinanceBench fixture docs:
#   AAPL 10-K filed 2024-10-30, NVDA 10-Q filed 2024-11-20.
# Plus two earlier AAPL pe_ratio rows to exercise PIT lookup in tests.

_FIXTURE_ROWS: list[tuple[str, str, date, Decimal]] = [
    # AAPL — main filing (2024-10-30)
    ("AAPL", "pe_ratio", date(2024, 10, 30), Decimal("28.5")),
    ("AAPL", "gross_margin", date(2024, 10, 30), Decimal("0.45")),
    ("AAPL", "revenue_yoy_growth", date(2024, 10, 30), Decimal("0.033")),
    ("AAPL", "debt_to_equity", date(2024, 10, 30), Decimal("1.62")),
    ("AAPL", "current_ratio", date(2024, 10, 30), Decimal("1.05")),
    # NVDA — main filing (2024-11-20)
    ("NVDA", "pe_ratio", date(2024, 11, 20), Decimal("64.2")),
    ("NVDA", "gross_margin", date(2024, 11, 20), Decimal("0.661")),
    ("NVDA", "revenue_yoy_growth", date(2024, 11, 20), Decimal("0.94")),
    ("NVDA", "debt_to_equity", date(2024, 11, 20), Decimal("0.41")),
    ("NVDA", "current_ratio", date(2024, 11, 20), Decimal("4.10")),
    # AAPL — earlier filings to exercise PIT "latest on or before" selection.
    # Test: as_of=2024-03-13 should return the 2024-02-01 row (Decimal("27.0")),
    #        not the 2024-05-01 row (Decimal("27.8")), not the 2024-10-30 row.
    ("AAPL", "pe_ratio", date(2024, 2, 1), Decimal("27.0")),
    ("AAPL", "pe_ratio", date(2024, 5, 1), Decimal("27.8")),
]


def _write_fixture_parquet() -> None:
    """Write the deterministic fixture parquet to tests/fixtures/."""
    tickers: list[str] = []
    ratio_names: list[str] = []
    as_of_dates: list[date] = []
    values: list[str] = []  # store as string to preserve Decimal precision

    for ticker, ratio_name, as_of, value in _FIXTURE_ROWS:
        tickers.append(ticker)
        ratio_names.append(ratio_name)
        as_of_dates.append(as_of)
        values.append(str(value))

    schema = pa.schema(
        [
            pa.field("ticker", pa.string()),
            pa.field("ratio_name", pa.string()),
            pa.field("as_of", pa.date32()),
            pa.field("value", pa.string()),
        ]
    )

    table = pa.table(
        {
            "ticker": pa.array(tickers, type=pa.string()),
            "ratio_name": pa.array(ratio_names, type=pa.string()),
            "as_of": pa.array(as_of_dates, type=pa.date32()),
            "value": pa.array(values, type=pa.string()),
        },
        schema=schema,
    )

    _FIXTURE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(_FIXTURE_PARQUET))
    print(f"Written {len(_FIXTURE_ROWS)} rows to {_FIXTURE_PARQUET}")


def _write_real_parquet() -> None:
    """Write the real pre-computed parquet from the full FinanceBench corpus.

    # TODO(plan-3): extract real ratios from filing tables
    """
    raise NotImplementedError(
        "Real ratio extraction is not implemented in Plan 2. "
        "See the TODO(plan-3) comment in scripts/precompute_fundamentals.py "
        "for the planned implementation."
    )


if __name__ == "__main__":
    if os.environ.get("FIRM_FUNDAMENTALS_FIXTURE") == "1":
        _write_fixture_parquet()
    else:
        _write_real_parquet()
