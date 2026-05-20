"""Pre-compute risk metrics and write a parquet of (ticker, metric_id, as_of, value).

Fixture mode (FIRM_RISK_FIXTURE=1):
    Reads tests/fixtures/financebench_two_docs.json for the AAPL + NVDA tickers
    and writes a deterministic fixture parquet. The fixture values are hard-coded
    so unit tests get a stable, reproducible parquet without depending on a real
    price series.

    Output path: by default tests/fixtures/risk_metrics_fixture.parquet, but
    can be overridden via the FIRM_RISK_OUT env var so pytest session fixtures
    can write into a tmp dir (pytest-xdist safe).

Real corpus mode (default):
    # TODO(plan-3): compute real risk metrics from a price series
    Raises NotImplementedError — Plan 2 only requires the fixture path.

Usage
-----
    # fixture mode (used by tests):
    FIRM_RISK_FIXTURE=1 python scripts/precompute_risk_metrics.py

    # fixture mode with custom output path:
    FIRM_RISK_FIXTURE=1 FIRM_RISK_OUT=/tmp/x.parquet \\
        python scripts/precompute_risk_metrics.py

    # real corpus (not yet implemented):
    python scripts/precompute_risk_metrics.py
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
_FIXTURE_PARQUET = _REPO_ROOT / "tests" / "fixtures" / "risk_metrics_fixture.parquet"
_REAL_PARQUET = _REPO_ROOT / "data" / "precomputed" / "risk_metrics.parquet"

# ---------------------------------------------------------------------------
# Fixture data (deterministic hard-coded rows)
# ---------------------------------------------------------------------------
# Main rows from the two FinanceBench fixture docs:
#   AAPL 10-K filed 2024-10-30, NVDA 10-Q filed 2024-11-20.
# Plus two earlier AAPL volatility_30d rows to exercise PIT lookup in tests.
#   Test: as_of=2024-03-13 should return the 2024-02-01 row (Decimal("0.18")),
#          not the 2024-05-01 row (Decimal("0.19")), not the 2024-10-30 row.

_FIXTURE_ROWS: list[tuple[str, str, date, Decimal]] = [
    # AAPL — main filing (2024-10-30)
    ("AAPL", "volatility_30d", date(2024, 10, 30), Decimal("0.21")),
    ("AAPL", "beta_180d", date(2024, 10, 30), Decimal("1.13")),
    ("AAPL", "max_drawdown_90d", date(2024, 10, 30), Decimal("0.078")),
    # NVDA — main filing (2024-11-20)
    ("NVDA", "volatility_30d", date(2024, 11, 20), Decimal("0.48")),
    ("NVDA", "beta_180d", date(2024, 11, 20), Decimal("1.72")),
    ("NVDA", "max_drawdown_90d", date(2024, 11, 20), Decimal("0.155")),
    # AAPL — earlier rows to exercise PIT "latest on or before" selection.
    ("AAPL", "volatility_30d", date(2024, 2, 1), Decimal("0.18")),
    ("AAPL", "volatility_30d", date(2024, 5, 1), Decimal("0.19")),
]


def _write_fixture_parquet(out_path: Path) -> None:
    """Write the deterministic fixture parquet to ``out_path``."""
    tickers: list[str] = []
    metric_ids: list[str] = []
    as_of_dates: list[date] = []
    values: list[str] = []  # store as string to preserve Decimal precision

    for ticker, metric_id, as_of, value in _FIXTURE_ROWS:
        tickers.append(ticker)
        metric_ids.append(metric_id)
        as_of_dates.append(as_of)
        values.append(str(value))

    schema = pa.schema(
        [
            pa.field("ticker", pa.string()),
            pa.field("metric_id", pa.string()),
            pa.field("as_of", pa.date32()),
            pa.field("value", pa.string()),
        ]
    )

    table = pa.table(
        {
            "ticker": pa.array(tickers, type=pa.string()),
            "metric_id": pa.array(metric_ids, type=pa.string()),
            "as_of": pa.array(as_of_dates, type=pa.date32()),
            "value": pa.array(values, type=pa.string()),
        },
        schema=schema,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_path))
    print(f"Written {len(_FIXTURE_ROWS)} rows to {out_path}")


def _write_real_parquet() -> None:
    """Write the real pre-computed parquet from a price series.

    # TODO(plan-3): compute real risk metrics from a price series
    """
    raise NotImplementedError(
        "Real risk metric computation is not implemented in Plan 2. "
        "See the TODO(plan-3) comment in scripts/precompute_risk_metrics.py "
        "for the planned implementation."
    )


if __name__ == "__main__":
    if os.environ.get("FIRM_RISK_FIXTURE") == "1":
        out_override = os.environ.get("FIRM_RISK_OUT")
        out_path = Path(out_override) if out_override else _FIXTURE_PARQUET
        _write_fixture_parquet(out_path)
    else:
        _write_real_parquet()
