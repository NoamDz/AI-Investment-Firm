"""T26 — Cost ledger smoke: heartbeat-end cost-so-far print.

Verifies that ``make_reporter`` prints ``Cost so far today: $X.XXX`` to stdout
at the end of every heartbeat, driven by a live ``cost_ledger`` query.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from firm.agents.reporter import make_reporter
from firm.core.clock import ReplayClock
from firm.db.cost_ledger import write_cost_ledger_row
from firm.db.migrations import init_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> Path:
    db = tmp_path / "firm.db"
    init_db(db)
    return db


def _fixed_clock(year: int = 2026, month: int = 5, day: int = 21) -> ReplayClock:
    """Return a clock fixed at noon UTC on the given date."""
    return ReplayClock(datetime(year, month, day, 12, 0, 0, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cost_print_with_one_row(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """A single cost_ledger row of $0.012 → 'Cost so far today: $0.012'."""
    db = _make_db(tmp_path)
    clock = _fixed_clock()

    # Insert a row timestamped a few seconds after midnight today (UTC).
    today = clock.now().astimezone(timezone.utc).strftime("%Y-%m-%d")
    created_at_clock = ReplayClock(
        datetime.fromisoformat(f"{today}T00:00:05+00:00")
    )
    write_cost_ledger_row(
        db_path=db,
        decision_id="d1",
        agent="research",
        model="claude-sonnet-4-6",
        input_tokens=100,
        output_tokens=50,
        cached_tokens=None,
        cost_usd=0.012,
        clock=created_at_clock,
    )

    reporter = make_reporter(
        reports_root=tmp_path / "reports",
        clock=clock,
        db_path=db,
    )
    reporter({})

    captured = capsys.readouterr()
    # Regex check: must match Cost so far today: $<digits>.<3 digits>
    assert re.search(r"Cost so far today: \$\d+\.\d{3}", captured.out), (
        f"Expected cost print line not found in stdout: {captured.out!r}"
    )
    # Exact-value check: one 0.012 row → $0.012
    assert "Cost so far today: $0.012" in captured.out, (
        f"Expected exact '$0.012' but got: {captured.out!r}"
    )


def test_cost_print_zero_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """No rows in cost_ledger → 'Cost so far today: $0.000'."""
    db = _make_db(tmp_path)
    clock = _fixed_clock()

    reporter = make_reporter(
        reports_root=tmp_path / "reports",
        clock=clock,
        db_path=db,
    )
    reporter({})

    captured = capsys.readouterr()
    assert re.search(r"Cost so far today: \$\d+\.\d{3}", captured.out), (
        f"Expected cost print line not found in stdout: {captured.out!r}"
    )
    assert "Cost so far today: $0.000" in captured.out, (
        f"Expected exact '$0.000' but got: {captured.out!r}"
    )
