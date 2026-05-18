import json
from datetime import datetime, timezone
from pathlib import Path

from firm.agents.reporter import make_reporter
from firm.core.clock import ReplayClock


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
    lines = [json.loads(l) for l in p.read_text().splitlines()]
    assert any(l.get("execution_result") for l in lines)
