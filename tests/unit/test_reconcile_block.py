"""Tests for firm.reports.reconcile_block (Plan 3 §T17)."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from firm.broker.protocol import Position, Quote
from firm.core.clock import ReplayClock
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.reports.reconcile_block import (
    _format_decimal_shares,
    _format_signed_cash,
    render_reconcile_block,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GOLDEN_DATE = datetime(2024, 3, 13, 16, 0, 0, tzinfo=timezone.utc)
_GOLDEN_DIR = (
    Path(__file__).parent.parent / "fixtures" / "reports" / "2024-03-13"
)


# ---------------------------------------------------------------------------
# Stub broker
# ---------------------------------------------------------------------------


class _StubBroker:
    """Minimal Broker Protocol implementation with configurable positions/cash."""

    def __init__(
        self,
        positions: list[tuple[str, str]] | None = None,
        cash: str = "94250.00",
    ) -> None:
        self._positions: list[tuple[str, str]] = positions or []
        self._cash = Decimal(cash)

    def list_positions(self) -> list[Position]:
        return [
            Position(ticker=t, shares=Decimal(s), avg_cost=Decimal("0"))
            for t, s in self._positions
        ]

    def get_cash(self) -> Decimal:
        return self._cash

    def get_quote(self, ticker: str) -> Quote:
        raise NotImplementedError

    def submit(self, decision_payload: dict[str, Any], idempotency_key: str) -> Any:  # noqa: ANN401
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Seed helpers (raw DB inserts matching schema column layout)
# ---------------------------------------------------------------------------


def _seed_position(db: Path, ticker: str, shares: str, ts: str) -> None:
    with closing(get_conn(db)) as conn:
        conn.execute(
            "INSERT INTO positions (ticker, shares, avg_cost, updated_at) VALUES (?, ?, ?, ?)",
            (ticker, shares, "0", ts),
        )


def _seed_cash(db: Path, amount: str, ts: str) -> None:
    with closing(get_conn(db)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cash (id, amount, updated_at) VALUES (1, ?, ?)",
            (amount, ts),
        )


# ---------------------------------------------------------------------------
# Golden-file helpers
# ---------------------------------------------------------------------------


def _read_golden(name: str) -> str:
    return (_GOLDEN_DIR / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_reconcile_clean_matches_golden(tmp_path: Path) -> None:
    """Positions and cash fully match → clean output, no footnote."""
    db = tmp_path / "firm.db"
    init_db(db)
    clock = ReplayClock(_GOLDEN_DATE)
    ts = clock.now().isoformat()

    # Broker: AAPL 100, MSFT 200, NVDA 50 @ $94,250.00
    broker = _StubBroker(
        positions=[("AAPL", "100"), ("MSFT", "200"), ("NVDA", "50")],
        cash="94250.00",
    )
    # Local matches broker exactly.
    _seed_position(db, "AAPL", "100", ts)
    _seed_position(db, "MSFT", "200", ts)
    _seed_position(db, "NVDA", "50", ts)
    _seed_cash(db, "94250.00", ts)

    result = render_reconcile_block(db_path=db, broker=broker, clock=clock)
    expected = _read_golden("reconcile_clean.md")
    assert result == expected, (
        f"reconcile_clean mismatch.\nGOT:\n{result!r}\nEXPECTED:\n{expected!r}"
    )


def test_reconcile_position_drift_matches_golden(tmp_path: Path) -> None:
    """Broker NVDA=50, local NVDA=51 → position diff, mismatch footnote with id=1."""
    db = tmp_path / "firm.db"
    init_db(db)
    clock = ReplayClock(_GOLDEN_DATE)
    ts = clock.now().isoformat()

    broker = _StubBroker(
        positions=[("AAPL", "100"), ("MSFT", "200"), ("NVDA", "50")],
        cash="94250.00",
    )
    # Local has one extra NVDA share.
    _seed_position(db, "AAPL", "100", ts)
    _seed_position(db, "MSFT", "200", ts)
    _seed_position(db, "NVDA", "51", ts)
    _seed_cash(db, "94250.00", ts)

    result = render_reconcile_block(db_path=db, broker=broker, clock=clock)
    expected = _read_golden("reconcile_position_drift.md")
    assert result == expected, (
        f"reconcile_position_drift mismatch.\nGOT:\n{result!r}\nEXPECTED:\n{expected!r}"
    )


def test_reconcile_cash_drift_matches_golden(tmp_path: Path) -> None:
    """Positions equal, broker cash $94,250.00 vs local $93,000.00 → cash diff $1,250.00."""
    db = tmp_path / "firm.db"
    init_db(db)
    clock = ReplayClock(_GOLDEN_DATE)
    ts = clock.now().isoformat()

    broker = _StubBroker(
        positions=[("AAPL", "100"), ("MSFT", "200"), ("NVDA", "50")],
        cash="94250.00",
    )
    # Local has less cash.
    _seed_position(db, "AAPL", "100", ts)
    _seed_position(db, "MSFT", "200", ts)
    _seed_position(db, "NVDA", "50", ts)
    _seed_cash(db, "93000.00", ts)

    result = render_reconcile_block(db_path=db, broker=broker, clock=clock)
    expected = _read_golden("reconcile_cash_drift.md")
    assert result == expected, (
        f"reconcile_cash_drift mismatch.\nGOT:\n{result!r}\nEXPECTED:\n{expected!r}"
    )


def test_audit_row_id_in_footnote(tmp_path: Path) -> None:
    """Mismatch block must reference the exact audit_log row id just written."""
    db = tmp_path / "firm.db"
    init_db(db)
    clock = ReplayClock(_GOLDEN_DATE)
    ts = clock.now().isoformat()

    # Position mismatch so we get a footnote.
    broker = _StubBroker(positions=[("AAPL", "100")], cash="10000.00")
    # Local has no positions → mismatch.
    _seed_cash(db, "10000.00", ts)

    result = render_reconcile_block(db_path=db, broker=broker, clock=clock)

    # Verify the audit_log row exists and its id matches what's in the block.
    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT id FROM audit_log WHERE event='reconcile.boot' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert row is not None, "audit_log must have a reconcile.boot row"
    audit_id = int(row["id"])

    assert f"[^audit-{audit_id}]" in result, (
        f"Expected [^audit-{audit_id}] in result but got:\n{result}"
    )


def test_renders_inside_daily_report(tmp_path: Path) -> None:
    """Integration: reconcile_block pipes through render_daily_report without error."""
    from datetime import date as _date

    from firm.reports.daily import render_daily_report

    db = tmp_path / "firm.db"
    init_db(db)
    clock = ReplayClock(_GOLDEN_DATE)
    ts = clock.now().isoformat()

    broker = _StubBroker(
        positions=[("AAPL", "100")],
        cash="50000.00",
    )
    _seed_position(db, "AAPL", "100", ts)
    _seed_cash(db, "50000.00", ts)

    reconcile_block = render_reconcile_block(db_path=db, broker=broker, clock=clock)

    out = render_daily_report(
        date=_date(2024, 3, 13),
        db_path=db,
        broker=broker,
        traces_path=tmp_path / "traces.jsonl",
        reports_root=tmp_path / "reports",
        reconcile_block=reconcile_block,
    )

    content = out.read_text(encoding="utf-8")
    assert "RECONCILIATION (EOD)" in content
    # Clean reconcile → check mark present.
    assert "✓" in content  # ✓


# ---------------------------------------------------------------------------
# Formatter regression tests (lock against future refactor regressions)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (Decimal("0"), "$0.00"),
        (Decimal("1250.00"), "$1,250.00"),
        (Decimal("-1250.00"), "-$1,250.00"),
        (Decimal("-0.01"), "-$0.01"),
    ],
)
def test_format_signed_cash(value: Decimal, expected: str) -> None:
    """Locks sign handling: bug that flips sign or drops abs() must fail here."""
    assert _format_signed_cash(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (Decimal("100"), "100"),
        (Decimal("100.00"), "100"),
        (Decimal("12.5"), "12.5"),
        (Decimal("0.001"), "0.001"),
        (Decimal("0"), "0"),
    ],
)
def test_format_decimal_shares(value: Decimal, expected: str) -> None:
    """Locks trailing-zero stripping: regression to Decimal.normalize() must fail here."""
    assert _format_decimal_shares(value) == expected
