"""Tests for firm.reports.html (Plan 4 §T1)."""
from __future__ import annotations

import json
from contextlib import closing
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from firm.broker.protocol import Position, Quote
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.reports.html import render_daily_html

# ---------------------------------------------------------------------------
# Test fixtures (mirror tests/unit/test_daily_report.py shape)
# ---------------------------------------------------------------------------

_DATE = date(2024, 3, 13)
_DATE_STR = "2024-03-13"

_RECONCILE_BLOCK = (
    "  Broker positions:   { AAPL: 100 }\n"
    "  Local positions:    { AAPL: 100 }\n"
    "  Position diff:      none\n"
    "  Status:             ✓ books tie to broker"
)


class _StubBroker:
    """Minimal Broker Protocol implementation — not called by render_daily_html."""

    def list_positions(self) -> list[Position]:
        return []

    def get_cash(self) -> Decimal:
        return Decimal("0")

    def get_quote(self, ticker: str) -> Quote:  # noqa: ARG002
        raise NotImplementedError

    def submit(self, decision_payload: dict[str, Any], idempotency_key: str) -> Any:  # noqa: ANN401, ARG002
        raise NotImplementedError


def _decision_row(
    id_: str,
    action: str,
    ts: str,
    *,
    payload: dict[str, Any] | None = None,
    rationale: str = "rationale",
    citations: list[dict[str, Any]] | None = None,
    failure_mode: str | None = None,
    confidence: float = 0.7,
) -> tuple[Any, ...]:
    """Build a decisions-table row tuple matching the schema column order."""
    return (
        id_,
        json.dumps([]),                       # parent_chain
        action,
        json.dumps(payload or {}),            # payload
        rationale,
        confidence,
        json.dumps(citations or []),          # citations
        "falsification",
        None,                                 # escalation
        failure_mode,
        json.dumps({}),                       # metadata
        f"nonce-{id_}",
        ts,
    )


def _insert_decisions(db: Path, rows: list[tuple[Any, ...]]) -> None:
    with closing(get_conn(db)) as conn:
        conn.executemany(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_renders_to_expected_path(tmp_path: Path) -> None:
    """Empty DB → file lands at reports_root/<date>/daily_report.html."""
    db = tmp_path / "firm.db"
    init_db(db)
    reports_root = tmp_path / "reports"

    out = render_daily_html(
        date=_DATE,
        db_path=db,
        broker=_StubBroker(),
        traces_path=tmp_path / "traces",
        reports_root=reports_root,
        reconcile_block=_RECONCILE_BLOCK,
    )

    assert out.exists()
    assert out == reports_root / _DATE_STR / "daily_report.html"
    # Empty-day summary text shows up.
    assert f"No decisions recorded for {_DATE_STR}." in out.read_text(encoding="utf-8")


def test_deterministic(tmp_path: Path) -> None:
    """Same inputs → byte-identical bytes across two writes into separate dirs."""
    db = tmp_path / "firm.db"
    init_db(db)
    _insert_decisions(
        db,
        [
            _decision_row(
                "dec-buy-1", "BUY",
                f"{_DATE_STR}T09:00:00Z",
                payload={"ticker": "AAPL", "shares": 10},
            ),
            _decision_row(
                "dec-sell-1", "SELL",
                f"{_DATE_STR}T10:00:00Z",
                payload={"ticker": "AAPL", "shares": 5},
            ),
        ],
    )

    out1 = render_daily_html(
        date=_DATE, db_path=db, broker=_StubBroker(),
        traces_path=tmp_path / "traces",
        reports_root=tmp_path / "r1",
        reconcile_block=_RECONCILE_BLOCK,
    )
    out2 = render_daily_html(
        date=_DATE, db_path=db, broker=_StubBroker(),
        traces_path=tmp_path / "traces",
        reports_root=tmp_path / "r2",
        reconcile_block=_RECONCILE_BLOCK,
    )

    assert out1.read_bytes() == out2.read_bytes()


def test_action_count_summary(tmp_path: Path) -> None:
    """Executive summary names the action counts (2 BUY, 1 SELL)."""
    db = tmp_path / "firm.db"
    init_db(db)
    _insert_decisions(
        db,
        [
            _decision_row("d1", "BUY",  f"{_DATE_STR}T09:00:00Z",
                          payload={"ticker": "AAPL", "shares": 5}),
            _decision_row("d2", "BUY",  f"{_DATE_STR}T09:30:00Z",
                          payload={"ticker": "MSFT", "shares": 3}),
            _decision_row("d3", "SELL", f"{_DATE_STR}T10:00:00Z",
                          payload={"ticker": "AAPL", "shares": 1}),
        ],
    )

    out = render_daily_html(
        date=_DATE, db_path=db, broker=_StubBroker(),
        traces_path=tmp_path / "traces",
        reports_root=tmp_path / "reports",
        reconcile_block="",
    )
    content = out.read_text(encoding="utf-8")
    assert "2 BUY" in content
    assert "1 SELL" in content


def test_html_escapes_rationale(tmp_path: Path) -> None:
    """Rationale must be HTML-escaped — no live <script> tag may leak through."""
    db = tmp_path / "firm.db"
    init_db(db)
    _insert_decisions(
        db,
        [
            _decision_row(
                "d-xss", "HOLD",
                f"{_DATE_STR}T09:00:00Z",
                rationale="<script>alert(1)</script>",
            ),
        ],
    )

    out = render_daily_html(
        date=_DATE, db_path=db, broker=_StubBroker(),
        traces_path=tmp_path / "traces",
        reports_root=tmp_path / "reports",
        reconcile_block="",
    )
    content = out.read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in content
    assert "&lt;script&gt;" in content
