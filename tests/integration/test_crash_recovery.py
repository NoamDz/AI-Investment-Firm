import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.outbox.outbox import place_order_via_outbox, recover_pending


def _persist_decision(db: Path, d: Decision, clock):
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            d.id, json.dumps(d.decision_id_chain), d.action.value,
            d.payload.model_dump_json(), d.rationale, d.confidence,
            "[]", d.falsification_condition, d.escalation_reason,
            d.failure_mode.value if d.failure_mode else None,
            json.dumps(d.metadata), d.nonce, clock.now().isoformat(),
        ),
    )


def test_kill_mid_order_restart_is_exactly_once(tmp_path: Path):
    db = tmp_path / "firm.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = Decision(
        id="dec-1", decision_id_chain=[], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="x", confidence=0.5, citations=[],
        falsification_condition="y", escalation_reason=None,
        failure_mode=None, metadata={}, nonce="n",
    )
    _persist_decision(db, d, clock)

    # Step 1: normal place — confirmed
    r1 = place_order_via_outbox(d, db, broker, clock)
    assert r1.filled_shares == Decimal("10")

    # Step 2: simulate crash — flip the outbox status back to 'pending' manually
    # (mimicking the case where confirm-update never landed)
    conn = get_conn(db)
    conn.execute("UPDATE outbox SET status='pending' WHERE decision_id=?", (d.id,))

    # Step 3: restart recovery
    recovered = recover_pending(db, broker, clock)
    assert len(recovered) == 1
    assert recovered[0].order_id == r1.order_id  # same fill, no duplicate

    # Step 4: broker still has exactly 10 shares (not 20)
    pos = [p for p in broker.list_positions() if p.ticker == "AAPL"]
    assert len(pos) == 1
    assert pos[0].shares == Decimal("10")
