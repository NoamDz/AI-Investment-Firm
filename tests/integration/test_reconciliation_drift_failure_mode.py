"""Plan 3 T25 — RECONCILIATION_DRIFT end-to-end triggering fixture.

Verifies that when reconcile_on_boot detects a position mismatch between
the local DB and the broker, the resulting ReconcileResult carries
failure_mode=RECONCILIATION_DRIFT and an audit_log entry is written with
the failure_mode in its JSON payload.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.models import FailureMode
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.reconcile.boot import reconcile_on_boot


_CLOCK = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))


def _seed_local_position(db: Path, ticker: str, shares: str, avg_cost: str) -> None:
    with get_conn(db) as conn:
        conn.execute(
            "INSERT INTO positions (ticker, shares, avg_cost, updated_at) VALUES (?, ?, ?, ?)",
            (ticker, shares, avg_cost, _CLOCK.now().isoformat()),
        )


def _seed_local_cash(db: Path, amount: str) -> None:
    with get_conn(db) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cash (id, amount, updated_at) VALUES (1, ?, ?)",
            (amount, _CLOCK.now().isoformat()),
        )


def test_boot_reconcile_mismatch_emits_reconciliation_drift(tmp_path: Path) -> None:
    """Local DB has AAPL=10 shares; broker has AAPL=20 shares → mismatch.

    Asserts:
    - result.status == 'mismatch'
    - result.failure_mode == FailureMode.RECONCILIATION_DRIFT
    - audit_log contains a reconcile.boot entry whose JSON payload includes
      failure_mode == 'reconciliation_drift'.
    """
    db = tmp_path / "drift.db"
    init_db(db)

    # Broker: buy 20 shares of AAPL
    broker = FakeBroker(initial_cash=Decimal("100000"))
    broker.submit({"kind": "buy", "ticker": "AAPL", "shares": "20"}, "order-1")

    # Local DB: only 10 shares of AAPL + matching cash
    _seed_local_position(db, "AAPL", "10", "150")
    _seed_local_cash(db, str(broker.get_cash()))

    result = reconcile_on_boot(db, broker, _CLOCK)

    # FailureMode stamped on result
    assert result.status == "mismatch"
    assert result.failure_mode == FailureMode.RECONCILIATION_DRIFT

    # audit_log entry carries failure_mode in its JSON payload
    with get_conn(db) as conn:
        rows = list(
            conn.execute(
                "SELECT detail FROM audit_log WHERE event='reconcile.boot'"
            )
        )
    assert len(rows) == 1, "Expected exactly one reconcile.boot audit entry"
    detail = json.loads(rows[0]["detail"])
    assert detail.get("failure_mode") == FailureMode.RECONCILIATION_DRIFT.value, (
        f"Expected failure_mode='reconciliation_drift' in audit payload, got: {detail}"
    )
    assert detail.get("status") == "mismatch"
    assert "positions" in detail.get("diff", {})


def test_boot_reconcile_ok_does_not_emit_failure_mode(tmp_path: Path) -> None:
    """When local DB matches broker exactly, no failure_mode is set."""
    db = tmp_path / "ok.db"
    init_db(db)

    broker = FakeBroker(initial_cash=Decimal("100000"))
    _seed_local_cash(db, "100000")

    result = reconcile_on_boot(db, broker, _CLOCK)

    assert result.status == "ok"
    assert result.failure_mode is None

    with get_conn(db) as conn:
        rows = list(
            conn.execute(
                "SELECT detail FROM audit_log WHERE event='reconcile.boot'"
            )
        )
    assert len(rows) == 1
    detail = json.loads(rows[0]["detail"])
    assert "failure_mode" not in detail
