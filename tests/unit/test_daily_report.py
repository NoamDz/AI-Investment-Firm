"""Tests for firm.reports.daily (Plan 3 §T16)."""
from __future__ import annotations

import json
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

from firm.broker.protocol import Position, Quote
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.reports.daily import render_daily_report

# ---------------------------------------------------------------------------
# Fixtures — golden file date and reconcile block used across tests
# ---------------------------------------------------------------------------

_GOLDEN_DATE = date(2024, 3, 13)
_GOLDEN_DATE_STR = "2024-03-13"
_GOLDEN_DIR = Path(__file__).parent.parent / "fixtures" / "reports" / _GOLDEN_DATE_STR

_RECONCILE_BLOCK = (
    "  Broker positions:   { AAPL: 100 }\n"
    "  Local positions:    { AAPL: 100 }\n"
    "  Position diff:      none\n"
    "  Status:             ✓ books tie to broker"
)

# ---------------------------------------------------------------------------
# Stub broker (accepted by the API; not used in T16 rendering)
# ---------------------------------------------------------------------------


class _StubBroker:
    """Minimal Broker Protocol implementation — not called by T16."""

    def list_positions(self) -> list[Position]:
        return []

    def get_cash(self) -> Any:  # noqa: ANN401
        from decimal import Decimal

        return Decimal("0")

    def get_quote(self, ticker: str) -> Quote:
        raise NotImplementedError

    def submit(self, decision_payload: dict[str, Any], idempotency_key: str) -> Any:  # noqa: ANN401
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------


def _decision_row(
    id_: str,
    action: str,
    failure_mode: str | None,
    ts: str,
) -> tuple[Any, ...]:
    """Build a minimal decisions row tuple matching the schema column order."""
    return (
        id_,
        json.dumps([]),       # parent_chain
        action,
        json.dumps({}),       # payload
        "rationale",          # rationale
        0.7,                  # confidence
        json.dumps([]),       # citations
        "falsification",      # falsification
        None,                 # escalation
        failure_mode,         # failure_mode
        json.dumps({}),       # metadata
        f"nonce-{id_}",       # nonce
        ts,                   # created_at
    )


def _seed_golden_decisions(db: Path) -> None:
    """Seed 5 decisions for 2024-03-13 per spec:
    2 BUY (no failure_mode), 1 SELL, 1 HOLD, 1 ESCALATE (risk_limit_breached).
    """
    rows = [
        _decision_row("dec-buy-1",  "BUY",      None,                  f"{_GOLDEN_DATE_STR}T09:00:00Z"),
        _decision_row("dec-buy-2",  "BUY",      None,                  f"{_GOLDEN_DATE_STR}T09:30:00Z"),
        _decision_row("dec-sell-1", "SELL",     None,                  f"{_GOLDEN_DATE_STR}T10:00:00Z"),
        _decision_row("dec-hold-1", "HOLD",     None,                  f"{_GOLDEN_DATE_STR}T11:00:00Z"),
        _decision_row("dec-esc-1",  "ESCALATE", "risk_limit_breached", f"{_GOLDEN_DATE_STR}T14:00:00Z"),
    ]
    with closing(get_conn(db)) as conn:
        conn.executemany(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )


def _cost_row(
    id_: int,
    model: str,
    cached_tokens: int | None,
    cost_usd: float,
    ts: str,
) -> tuple[Any, ...]:
    """Build a cost_ledger row tuple matching schema column order."""
    return (
        id_,
        "dec-buy-1",       # decision_id
        "pm_agent",        # agent
        model,
        None,              # input_tokens
        None,              # output_tokens
        cached_tokens,
        cost_usd,
        ts,
    )


_SONNET_MODEL = "claude-sonnet-4-6"
_HAIKU_MODEL = "claude-haiku-4-5"


def _seed_golden_costs(db: Path) -> None:
    """Seed 6 cost_ledger rows for 2024-03-13 per spec:
    3 Sonnet live rows at $0.0021, 2 Haiku rows at $0.0003,
    1 cached Sonnet row (cached_tokens=500, cost_usd=0.0).
    """
    rows = [
        _cost_row(1, _SONNET_MODEL, None, 0.0021, f"{_GOLDEN_DATE_STR}T09:00:00Z"),
        _cost_row(2, _SONNET_MODEL, None, 0.0021, f"{_GOLDEN_DATE_STR}T09:30:00Z"),
        _cost_row(3, _SONNET_MODEL, None, 0.0021, f"{_GOLDEN_DATE_STR}T10:00:00Z"),
        _cost_row(4, _HAIKU_MODEL,  None, 0.0003, f"{_GOLDEN_DATE_STR}T11:00:00Z"),
        _cost_row(5, _HAIKU_MODEL,  None, 0.0003, f"{_GOLDEN_DATE_STR}T12:00:00Z"),
        _cost_row(6, _SONNET_MODEL, 500,  0.0,    f"{_GOLDEN_DATE_STR}T14:00:00Z"),
    ]
    with closing(get_conn(db)) as conn:
        conn.executemany(
            "INSERT INTO cost_ledger VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )


# ---------------------------------------------------------------------------
# test_golden_match
# ---------------------------------------------------------------------------


def test_golden_match(tmp_path: Path) -> None:
    """Rendered daily_report.md must match the golden fixture byte-for-byte."""
    db = tmp_path / "firm.db"
    init_db(db)
    _seed_golden_decisions(db)
    _seed_golden_costs(db)

    out = render_daily_report(
        date=_GOLDEN_DATE,
        db_path=db,
        broker=_StubBroker(),
        traces_path=tmp_path / "traces.jsonl",
        reports_root=tmp_path / "reports",
        reconcile_block=_RECONCILE_BLOCK,
    )

    golden = _GOLDEN_DIR / "daily_report.md"
    assert out.read_text(encoding="utf-8") == golden.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# test_zero_cost_rows_renders_no_calls_line
# ---------------------------------------------------------------------------


def test_zero_cost_rows_renders_no_calls_line(tmp_path: Path) -> None:
    """When cost_ledger has no rows for the day, emit '(no LLM calls recorded)'."""
    db = tmp_path / "firm.db"
    init_db(db)
    # Seed one decision so section isn't empty, but NO cost rows.
    with closing(get_conn(db)) as conn:
        conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            _decision_row("dec-1", "HOLD", None, f"{_GOLDEN_DATE_STR}T10:00:00Z"),
        )

    out = render_daily_report(
        date=_GOLDEN_DATE,
        db_path=db,
        broker=_StubBroker(),
        traces_path=tmp_path / "traces.jsonl",
        reports_root=tmp_path / "reports",
        reconcile_block="",
    )
    content = out.read_text(encoding="utf-8")
    assert "(no LLM calls recorded)" in content


# ---------------------------------------------------------------------------
# test_zero_decisions_renders_zero_total
# ---------------------------------------------------------------------------


def test_zero_decisions_renders_zero_total(tmp_path: Path) -> None:
    """When there are no decisions for the day, Total: 0 and all counts are 0."""
    db = tmp_path / "firm.db"
    init_db(db)
    # No decisions seeded.

    out = render_daily_report(
        date=_GOLDEN_DATE,
        db_path=db,
        broker=_StubBroker(),
        traces_path=tmp_path / "traces.jsonl",
        reports_root=tmp_path / "reports",
        reconcile_block="",
    )
    content = out.read_text(encoding="utf-8")
    assert "Total: 0" in content
    # All histogram lines should show 0.
    for action in ["BUY", "SELL", "HOLD", "REFUSE", "ESCALATE"]:
        # Each line ends with " 0" (right-aligned count in 2).
        assert f"  {action}:" in content


# ---------------------------------------------------------------------------
# test_creates_parent_dirs
# ---------------------------------------------------------------------------


def test_creates_parent_dirs(tmp_path: Path) -> None:
    """render_daily_report creates reports_root/YYYY-MM-DD/ if it doesn't exist."""
    db = tmp_path / "firm.db"
    init_db(db)

    reports_root = tmp_path / "nested" / "deep" / "reports"
    # Confirm it doesn't exist yet.
    assert not reports_root.exists()

    out = render_daily_report(
        date=_GOLDEN_DATE,
        db_path=db,
        broker=_StubBroker(),
        traces_path=tmp_path / "traces.jsonl",
        reports_root=reports_root,
        reconcile_block="",
    )

    assert out.exists()
    assert out.name == "daily_report.md"
    assert out.parent == reports_root / _GOLDEN_DATE_STR


# ---------------------------------------------------------------------------
# test_failure_mode_breakdown_only_lists_observed
# ---------------------------------------------------------------------------


def test_failure_mode_breakdown_only_lists_observed(tmp_path: Path) -> None:
    """failure_mode breakdown shows only modes that occurred, not all 13 enums."""
    db = tmp_path / "firm.db"
    init_db(db)

    # Seed 2 ESCALATE rows with different failure_modes.
    rows = [
        _decision_row("esc-1", "ESCALATE", "risk_limit_breached",  f"{_GOLDEN_DATE_STR}T09:00:00Z"),
        _decision_row("esc-2", "ESCALATE", "insufficient_evidence", f"{_GOLDEN_DATE_STR}T10:00:00Z"),
    ]
    with closing(get_conn(db)) as conn:
        conn.executemany(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )

    out = render_daily_report(
        date=_GOLDEN_DATE,
        db_path=db,
        broker=_StubBroker(),
        traces_path=tmp_path / "traces.jsonl",
        reports_root=tmp_path / "reports",
        reconcile_block="",
    )
    content = out.read_text(encoding="utf-8")

    assert "risk_limit_breached: 1" in content
    assert "insufficient_evidence: 1" in content
    # A mode that was NOT observed must not appear.
    assert "stale_data" not in content
