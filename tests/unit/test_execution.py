from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import json

from firm.agents.execution import make_execution
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision, RefusePayload
from firm.db.connection import get_conn
from firm.db.migrations import init_db


def _persist(db, d, clock):
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


def test_execution_fires_buy(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = Decision(
        id="risk-1", decision_id_chain=["pm-1"], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="ok", confidence=0.7, citations=[],
        falsification_condition="x", escalation_reason=None,
        failure_mode=None, metadata={}, nonce="n",
    )
    _persist(db, d, clock)
    exe = make_execution(db_path=db, broker=broker, clock=clock, nonce_secret=b"x" * 32)
    out = exe({"risk_decision": d, "hitl_required": False})
    assert "execution_result" in out
    assert out["execution_result"]["ticker"] == "AAPL"


def test_execution_skips_on_refuse(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = Decision(
        id="risk-1", decision_id_chain=["pm-1"], action=ActionEnum.REFUSE,
        payload=RefusePayload(reason="limit breach"),
        rationale="hard limit", confidence=1.0, citations=[],
        falsification_condition="x", escalation_reason=None,
        failure_mode=None, metadata={}, nonce="n",
    )
    _persist(db, d, clock)
    exe = make_execution(db_path=db, broker=broker, clock=clock, nonce_secret=b"x" * 32)
    out = exe({"risk_decision": d, "hitl_required": False})
    assert out.get("execution_result") is None or out["execution_result"].get("skipped")


def test_execution_skips_when_hitl_not_approved(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = Decision(
        id="risk-1", decision_id_chain=["pm-1"], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="ok", confidence=0.7, citations=[],
        falsification_condition="x", escalation_reason=None,
        failure_mode=None, metadata={}, nonce="n",
    )
    _persist(db, d, clock)
    exe = make_execution(db_path=db, broker=broker, clock=clock, nonce_secret=b"x" * 32)
    out = exe({"risk_decision": d, "hitl_required": True, "hitl_approved": False})
    assert out.get("execution_result") is None or out["execution_result"].get("skipped")
