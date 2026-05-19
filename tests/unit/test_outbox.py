import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.models import (
    ActionEnum, BuyPayload, Decision,
)
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.outbox.outbox import place_order_via_outbox, recover_pending


def _decision(decision_id: str = "dec-1") -> Decision:
    return Decision(
        id=decision_id, decision_id_chain=[], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="test", confidence=0.5, citations=[],
        falsification_condition="if AAPL drops 10%",
        escalation_reason=None, failure_mode=None, metadata={}, nonce="n-1",
    )


def _persist_decision(db: Path, d: Decision, clock):
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            d.id, json.dumps(d.decision_id_chain), d.action.value,
            d.payload.model_dump_json(), d.rationale, d.confidence,
            json.dumps([c.model_dump(mode="json") for c in d.citations]),
            d.falsification_condition, d.escalation_reason,
            d.failure_mode.value if d.failure_mode else None,
            json.dumps(d.metadata), d.nonce, clock.now().isoformat(),
        ),
    )


def test_outbox_places_order_and_marks_confirmed(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = _decision()
    _persist_decision(db, d, clock)

    result = place_order_via_outbox(d, db, broker, clock)
    assert result.ticker == "AAPL"

    row = get_conn(db).execute("SELECT status FROM outbox WHERE decision_id=?", (d.id,)).fetchone()
    assert row["status"] == "confirmed"


def test_outbox_is_idempotent_on_replay(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = _decision()
    _persist_decision(db, d, clock)

    r1 = place_order_via_outbox(d, db, broker, clock)
    r2 = place_order_via_outbox(d, db, broker, clock)  # second call must be no-op
    assert r1.order_id == r2.order_id
    rows = get_conn(db).execute("SELECT COUNT(*) c FROM outbox WHERE decision_id=?", (d.id,)).fetchone()
    assert rows["c"] == 1
    # cash debited exactly once
    pos = [p for p in broker.list_positions() if p.ticker == "AAPL"][0]
    assert pos.shares == Decimal("10")


def test_recover_pending_drives_pending_rows_to_confirmed(tmp_path: Path):
    """Manually insert a pending row using the canonical idempotency key,
    then verify recover_pending re-submits with that key (so FakeBroker's
    dedup cache is exercised on the actual recovery path)."""
    from firm.outbox.outbox import _idempotency_key
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = _decision()
    _persist_decision(db, d, clock)

    # Simulate crash after-insert-before-call: pending row with the real key.
    key = _idempotency_key(d)
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO outbox (key, decision_id, payload, status, created_at, updated_at) "
        "VALUES (?, ?, ?, 'pending', ?, ?)",
        (key, d.id, d.model_dump_json(), clock.now().isoformat(), clock.now().isoformat()),
    )

    recovered = recover_pending(db, broker, clock)
    assert len(recovered) == 1

    row = get_conn(db).execute("SELECT status FROM outbox WHERE key=?", (key,)).fetchone()
    assert row["status"] == "confirmed"
    # And the broker should have exactly one position from this recovery.
    pos = [p for p in broker.list_positions() if p.ticker == "AAPL"][0]
    assert pos.shares == Decimal("10")


def test_place_order_after_pending_row_recovers_synchronously(tmp_path: Path):
    """If a pending row already exists (from a prior crash) and place_order_via_outbox
    is called again synchronously, it must call broker.submit and flip to confirmed."""
    from firm.outbox.outbox import _idempotency_key
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = _decision()
    _persist_decision(db, d, clock)

    # Pre-insert a pending row with the canonical key (simulates crash after insert).
    key = _idempotency_key(d)
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO outbox (key, decision_id, payload, status, created_at, updated_at) "
        "VALUES (?, ?, ?, 'pending', ?, ?)",
        (key, d.id, d.model_dump_json(), clock.now().isoformat(), clock.now().isoformat()),
    )

    result = place_order_via_outbox(d, db, broker, clock)
    assert result.ticker == "AAPL"

    row = get_conn(db).execute("SELECT status FROM outbox WHERE key=?", (key,)).fetchone()
    assert row["status"] == "confirmed"
    # Exactly one position — the recovery resubmit hit FakeBroker's idempotency cache.
    pos = [p for p in broker.list_positions() if p.ticker == "AAPL"][0]
    assert pos.shares == Decimal("10")


def test_different_decisions_produce_different_outbox_rows(tmp_path: Path):
    """Two decisions with different ids/nonces must produce distinct keys and rows."""
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("200000"))

    d1 = _decision(decision_id="dec-1")
    d2 = Decision(
        id="dec-2", decision_id_chain=[], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="MSFT", shares=Decimal("5")),
        rationale="test2", confidence=0.5, citations=[],
        falsification_condition="if MSFT drops 10%",
        escalation_reason=None, failure_mode=None, metadata={}, nonce="n-2",
    )
    _persist_decision(db, d1, clock)
    _persist_decision(db, d2, clock)

    r1 = place_order_via_outbox(d1, db, broker, clock)
    r2 = place_order_via_outbox(d2, db, broker, clock)
    assert r1.order_id != r2.order_id

    rows = list(get_conn(db).execute("SELECT key FROM outbox ORDER BY decision_id"))
    assert len(rows) == 2
    assert rows[0]["key"] != rows[1]["key"]
