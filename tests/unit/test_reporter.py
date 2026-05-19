import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.agents.reporter import make_reporter
from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision


def _make_decision() -> Decision:
    return Decision(
        id="dec-test-1",
        decision_id_chain=[],
        action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="Test rationale.",
        confidence=0.8,
        citations=[],
        falsification_condition="If revenue declines.",
        escalation_reason=None,
        failure_mode=None,
        metadata={},
        nonce="test-nonce-1",
    )


def test_reporter_writes_jsonl(tmp_path: Path):
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    reporter = make_reporter(reports_root=tmp_path, clock=clock)
    state = {
        "heartbeat_at": "2024-03-13T14:30:00+00:00",
        "execution_result": {"ticker": "AAPL", "filled_shares": "10"},
    }
    out = reporter(state)
    p = Path(out["report_path"])
    assert p.exists()
    lines = [json.loads(line) for line in p.read_text().splitlines()]
    assert any(line.get("execution_result") for line in lines)


def test_reporter_jsonl_decision_values_are_structured_dicts(tmp_path: Path):
    """Decision-valued state fields must be serialized as structured dicts, not Pydantic repr strings (I4 fix).

    Before the fix, json.dumps(..., default=str) turned Decision objects into
    their Pydantic __repr__ strings, making the JSONL non-round-trippable.
    """
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    reporter = make_reporter(reports_root=tmp_path, clock=clock)
    decision = _make_decision()
    state = {
        "heartbeat_at": "2024-03-13T14:30:00+00:00",
        "risk_decision": decision,
    }
    out = reporter(state)
    p = Path(out["report_path"])
    lines = [json.loads(line) for line in p.read_text().splitlines()]
    assert lines, "JSONL file must have at least one line"
    line = lines[0]

    # risk_decision must be a dict, not a string repr
    risk_val = line.get("risk_decision")
    assert isinstance(risk_val, dict), (
        f"risk_decision in JSONL should be a dict (structured), got {type(risk_val).__name__!r}: {risk_val!r}"
    )
    # Must contain the key fields for round-trippability
    for key in ("id", "action", "payload"):
        assert key in risk_val, f"risk_decision dict missing key {key!r}: {risk_val}"
