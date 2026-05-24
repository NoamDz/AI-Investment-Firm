"""Pre-compute fundamental ratios and write a parquet of (ticker, ratio_name, as_of, value).

Fixture mode (FIRM_FUNDAMENTALS_FIXTURE=1):
    Reads tests/fixtures/financebench_two_docs.json and writes a deterministic
    parquet. The fixture values are hard-coded so unit tests get a stable,
    reproducible parquet without depending on real ratio extraction from
    filing tables.

    Output path: by default tests/fixtures/fundamentals_fixture.parquet, but
    can be overridden via the FIRM_FUNDAMENTALS_OUT env var so pytest session
    fixtures can write into a tmp dir (pytest-xdist safe).

Real corpus mode (default):
    # TODO(plan-3): extract real ratios from filing tables
    Raises NotImplementedError — Plan 2 only requires the fixture path.

Usage
-----
    # fixture mode (used by tests):
    FIRM_FUNDAMENTALS_FIXTURE=1 python firm/ops/precompute_fundamentals.py

    # fixture mode with custom output path:
    FIRM_FUNDAMENTALS_FIXTURE=1 FIRM_FUNDAMENTALS_OUT=/tmp/x.parquet \
        python firm/ops/precompute_fundamentals.py

    # real corpus (not yet implemented):
    python firm/ops/precompute_fundamentals.py
"""
from __future__ import annotations

import os
from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent
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

# Demo-only rows — written to data/precomputed/fundamentals.parquet by the
# ``firm.cli ingest`` command, NOT to the test fixture parquet. Lets the
# universe.tickers[0] = "AMD" demo heartbeat satisfy tool calls (PIT lookup
# against the FIRM_REPLAY_AT=2024-03-13 clock).
_DEMO_ROWS: list[tuple[str, str, date, Decimal]] = [
    # FY2022 10-K (filed late 2022) — covers LLM queries that cite the
    # FinanceBench AMD filing's fiscal year.
    ("AMD", "pe_ratio", date(2022, 12, 31), Decimal("28.0")),
    ("AMD", "gross_margin", date(2022, 12, 31), Decimal("0.45")),
    ("AMD", "revenue_yoy_growth", date(2022, 12, 31), Decimal("0.44")),
    ("AMD", "debt_to_equity", date(2022, 12, 31), Decimal("0.06")),
    ("AMD", "current_ratio", date(2022, 12, 31), Decimal("2.36")),
    # Recent snapshot to exercise PIT "latest on or before" when replay clock
    # advances past 2024-02-01.
    ("AMD", "pe_ratio", date(2024, 2, 1), Decimal("145.0")),
    ("AMD", "gross_margin", date(2024, 2, 1), Decimal("0.46")),
    ("AMD", "revenue_yoy_growth", date(2024, 2, 1), Decimal("-0.01")),
    ("AMD", "debt_to_equity", date(2024, 2, 1), Decimal("0.06")),
    ("AMD", "current_ratio", date(2024, 2, 1), Decimal("2.50")),
]


def _write_parquet(
    rows: list[tuple[str, str, date, Decimal]], out_path: Path
) -> None:
    """Write the given rows to a parquet at ``out_path``."""
    tickers: list[str] = []
    ratio_names: list[str] = []
    as_of_dates: list[date] = []
    values: list[str] = []  # store as string to preserve Decimal precision

    for ticker, ratio_name, as_of, value in rows:
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_path))  # type: ignore[no-untyped-call]
    print(f"Written {len(rows)} rows to {out_path}")


def _write_fixture_parquet(out_path: Path) -> None:
    """Write the deterministic fixture parquet (tests path) to ``out_path``."""
    _write_parquet(_FIXTURE_ROWS, out_path)


def write_demo_parquet(out_path: Path = _REAL_PARQUET) -> None:
    """Write fixture + demo rows to ``data/precomputed/fundamentals.parquet``.

    Called from ``firm.cli ingest`` so reviewers don't need to invoke this
    script manually. Idempotent — overwrites the parquet each time.
    """
    _write_parquet(_FIXTURE_ROWS + _DEMO_ROWS, out_path)


def _write_real_parquet() -> None:
    """Write the real pre-computed parquet from the full FinanceBench corpus.

    # TODO(plan-3): extract real ratios from filing tables
    """
    raise NotImplementedError(
        "Real ratio extraction is not implemented in Plan 2. "
        "See the TODO(plan-3) comment in firm/ops/precompute_fundamentals.py "
        "for the planned implementation."
    )


if __name__ == "__main__":
    if os.environ.get("FIRM_FUNDAMENTALS_FIXTURE") == "1":
        out_override = os.environ.get("FIRM_FUNDAMENTALS_OUT")
        out_path = Path(out_override) if out_override else _FIXTURE_PARQUET
        _write_fixture_parquet(out_path)
    else:
        _write_real_parquet()
