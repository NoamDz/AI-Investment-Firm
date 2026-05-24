"""Integration test for `firm.cli report` (Plan 4 §T2 / PLAN_reports_overhaul.md).

Verifies that a single `firm.cli report --date YYYY-MM-DD` invocation produces
all three artifacts (daily_report.md, daily_report.html, positions.xlsx) and
enumerates each of them in stdout — the listing is line-stable so downstream
scripts (and CI greps) can rely on the filenames appearing on their own lines.
"""
from __future__ import annotations

import json
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from click.testing import CliRunner

from firm.broker.protocol import Position, Quote
from firm.cli import cli
from firm.db.connection import get_conn
from firm.db.migrations import init_db


# ---------------------------------------------------------------------------
# Stub broker — no network calls. Mirrors tests/unit/test_cli_report.py.
# ---------------------------------------------------------------------------


class _StubBroker:
    _positions = [
        Position(ticker="AAPL", shares=Decimal("100"), avg_cost=Decimal("150.00")),
    ]
    _cash = Decimal("94250.00")

    def list_positions(self) -> list[Position]:
        return list(self._positions)

    def get_cash(self) -> Decimal:
        return self._cash

    def get_quote(self, ticker: str) -> Quote:
        return Quote(
            ticker=ticker,
            price=Decimal("182.00"),
            timestamp="2024-03-13T16:00:00+00:00",
        )

    def submit(self, decision_payload: dict[str, Any], idempotency_key: str) -> Any:  # noqa: ANN401, ARG002
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Constants and DB seeding
# ---------------------------------------------------------------------------

_REPLAY_AT = "2024-03-13T16:00:00+00:00"
_REPORT_DATE = "2024-03-13"


def _seed_db(db: Path) -> None:
    """Seed one BUY decision and a cost_ledger row for the test date."""
    ts = "2024-03-13T14:30:00+00:00"
    with closing(get_conn(db)) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cash (id, amount, updated_at) VALUES (1, ?, ?)",
            ("94250.00", ts),
        )
        conn.execute(
            "INSERT INTO positions (ticker, shares, avg_cost, updated_at) VALUES (?, ?, ?, ?)",
            ("AAPL", "100", "150.00", ts),
        )
        conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "dec-buy-1",
                json.dumps(["pm-1"]),
                "BUY",
                json.dumps({"kind": "buy", "ticker": "AAPL", "shares": "100"}),
                "strong momentum",
                0.85,
                json.dumps([]),
                "if price drops 20%",
                None,
                None,
                json.dumps({}),
                "nonce-buy-1",
                ts,
            ),
        )
        # One cost_ledger row so the HTML cost-summary path is exercised.
        conn.execute(
            "INSERT INTO cost_ledger "
            "(decision_id, agent, model, input_tokens, output_tokens, "
            "cached_tokens, cost_usd, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("dec-buy-1", "research", "claude-sonnet-4-5", 1000, 500, 200, 0.012, ts),
        )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_report_writes_all_three_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`firm.cli report` writes md+html+xlsx and lists each filename on its own line."""
    db = tmp_path / "firm.db"
    reports_root = tmp_path / "reports"
    init_db(db)
    _seed_db(db)

    monkeypatch.setenv("FIRM_DB_PATH", str(db))
    monkeypatch.setenv("FIRM_REPORTS_ROOT", str(reports_root))
    monkeypatch.setenv("FIRM_REPLAY_AT", _REPLAY_AT)
    monkeypatch.setenv("FIRM_BROKER", "FAKE")

    runner = CliRunner()
    with patch("firm.cli.make_broker", return_value=_StubBroker()):
        result = runner.invoke(cli, ["report", "--date", _REPORT_DATE])

    assert result.exit_code == 0, result.output

    bundle_dir = reports_root / _REPORT_DATE
    assert (bundle_dir / "daily_report.md").exists(), "daily_report.md missing"
    assert (bundle_dir / "daily_report.html").exists(), "daily_report.html missing"
    assert (bundle_dir / "positions.xlsx").exists(), "positions.xlsx missing"

    # Each filename must appear in stdout — downstream parsing relies on this.
    assert "daily_report.md" in result.output
    assert "daily_report.html" in result.output
    assert "positions.xlsx" in result.output
    assert "Report bundle written" in result.output
