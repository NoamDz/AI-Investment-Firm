"""Unit tests for CLI risk_node helpers (Plan 4 T21).

Covers the previously-stubbed inputs that ``firm/cli.py:risk_node`` now
wires into ``RiskInput``:

  * ``_compute_quote_age_seconds`` — parse ISO 8601 timestamp on a broker
    Quote, return whole-second age vs. ``clock.now()``; falls back to 0
    on malformed input rather than crashing the heartbeat.
  * ``_count_trades_today`` — SQL count of BUY/SELL decisions persisted
    today (UTC midnight onwards); skips yesterday's rows.
  * ``_compute_daily_pnl_pct`` — stub returning 0.0 until the SOD-NAV
    snapshotter ships (see helper docstring TODO).

These tests pin the helper contracts so the risk gates downstream
(``policy.max_trades_per_day``, ``max_quote_age_seconds``, etc.) are
fed deterministically-correct inputs.
"""
from __future__ import annotations

import json
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.broker.protocol import Quote
from firm.cli import (
    _compute_daily_pnl_pct,
    _compute_quote_age_seconds,
    _count_trades_today,
)
from firm.core.clock import ReplayClock
from firm.db.connection import get_conn
from firm.db.migrations import init_db


_T0 = datetime(2024, 3, 13, 12, 0, 0, tzinfo=timezone.utc)


def test_compute_quote_age_seconds_with_fresh_quote() -> None:
    """A quote timestamped 60s before clock.now() returns age = 60."""
    clock = ReplayClock(_T0)
    quote_ts = (_T0.replace(second=0)).isoformat()  # 12:00:00
    # Advance clock to 12:01:00 — quote is 60s old.
    clock.advance(60)
    quote = Quote(ticker="AAPL", price=Decimal("180.00"), timestamp=quote_ts)
    assert _compute_quote_age_seconds(quote, clock) == 60


def test_compute_quote_age_seconds_with_malformed_timestamp_falls_back_to_zero() -> None:
    """Malformed timestamp must NOT crash the heartbeat; helper returns 0.

    Documented fallback: surfacing STALE_DATA is the risk evaluator's job
    when parsing succeeds; an unparseable timestamp is a different
    failure surface (broker contract violation) that the risk gate
    should not crash on.
    """
    clock = ReplayClock(_T0)
    quote = Quote(ticker="AAPL", price=Decimal("180.00"), timestamp="not-an-iso-string")
    assert _compute_quote_age_seconds(quote, clock) == 0


def test_count_trades_today_skips_yesterdays_executions(tmp_path: Path) -> None:
    """Yesterday's BUY/SELL rows must NOT count toward today's tally.

    SQL gate: ``created_at >= today's UTC midnight``.  We seed two rows
    (one yesterday-ish, one today-ish) and assert the count is 1.
    """
    db_path = tmp_path / "firm.db"
    init_db(db_path)
    clock = ReplayClock(_T0)

    yesterday_ts = (_T0.replace(hour=23, minute=59).replace(day=12)).isoformat()
    today_ts = _T0.replace(hour=9, minute=15).isoformat()

    def _insert(row_id: str, action: str, created_at: str) -> None:
        with closing(get_conn(db_path)) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row_id,
                    json.dumps([]),
                    action,
                    json.dumps({"ticker": "AAPL", "shares": "1", "kind": action.lower()}),
                    "test",
                    0.5,
                    json.dumps([]),
                    "test",
                    None,
                    None,
                    json.dumps({}),
                    "n",
                    created_at,
                ),
            )

    _insert("dec-yesterday-1", "BUY", yesterday_ts)
    _insert("dec-today-1", "BUY", today_ts)
    # A non-trade today row that must not count (REFUSE/HOLD).
    _insert("dec-today-refuse", "REFUSE", today_ts)

    assert _count_trades_today(db_path, clock) == 1


def test_compute_daily_pnl_pct_falls_back_to_zero_when_no_sod_nav(tmp_path: Path) -> None:
    """Without an SOD-NAV snapshotter the helper returns 0.0 (gate inert)."""
    db_path = tmp_path / "firm.db"
    init_db(db_path)
    clock = ReplayClock(_T0)
    result = _compute_daily_pnl_pct(db_path, clock, current_nav=Decimal("100000"))
    assert result == 0.0
