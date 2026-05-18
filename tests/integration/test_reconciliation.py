from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.reconcile.boot import reconcile_on_boot


def _seed_local_position(db: Path, ticker: str, shares: str, avg_cost: str, clock):
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO positions (ticker, shares, avg_cost, updated_at) VALUES (?, ?, ?, ?)",
        (ticker, shares, avg_cost, clock.now().isoformat()),
    )


def _seed_local_cash(db: Path, amount: str, clock):
    conn = get_conn(db)
    conn.execute(
        "INSERT OR REPLACE INTO cash (id, amount, updated_at) VALUES (1, ?, ?)",
        (amount, clock.now().isoformat()),
    )


def test_reconcile_clean_match_returns_ok(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    _seed_local_cash(db, "100000", clock)

    result = reconcile_on_boot(db, broker, clock)
    assert result.status == "ok"
    assert result.diff == {}


def test_reconcile_returns_mismatch_when_cash_differs(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    _seed_local_cash(db, "95000", clock)  # local thinks 95k; broker says 100k

    result = reconcile_on_boot(db, broker, clock)
    assert result.status == "mismatch"
    assert "cash" in result.diff


def test_reconcile_returns_mismatch_when_position_differs(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    broker.submit({"kind": "buy", "ticker": "AAPL", "shares": "10"}, "k1")
    _seed_local_cash(db, str(broker.get_cash()), clock)  # cash matches

    # local DB has no AAPL position; broker has 10 shares
    result = reconcile_on_boot(db, broker, clock)
    assert result.status == "mismatch"
    assert "positions" in result.diff


def test_reconcile_writes_to_reconciliations_table(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    _seed_local_cash(db, "100000", clock)

    reconcile_on_boot(db, broker, clock)
    rows = list(get_conn(db).execute("SELECT * FROM reconciliations WHERE kind='boot'"))
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
