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
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = _decision()
    _persist_decision(db, d, clock)

    # simulate crash between outbox-insert and broker-call: insert pending row manually
    conn = get_conn(db)
    key = "fake-key-1"
    conn.execute(
        "INSERT INTO outbox (key, decision_id, payload, status, created_at, updated_at) "
        "VALUES (?, ?, ?, 'pending', ?, ?)",
        (key, d.id, d.model_dump_json(), clock.now().isoformat(), clock.now().isoformat()),
    )

    recovered = recover_pending(db, broker, clock)
    assert len(recovered) == 1

    row = get_conn(db).execute("SELECT status FROM outbox WHERE key=?", (key,)).fetchone()
    assert row["status"] == "confirmed"
