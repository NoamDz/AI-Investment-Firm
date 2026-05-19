import json
from datetime import datetime, timezone
from pathlib import Path

from firm.audit.log import AuditLog
from firm.core.clock import ReplayClock
from firm.db.migrations import init_db


def test_audit_append_and_read(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    log = AuditLog(db, clock)

    log.append("reconcile.boot", {"status": "ok", "diff": None})
    log.append("hitl.ack", {"decision_id": "dec-1", "approver": "alice"})

    rows = log.read_all()
    assert len(rows) == 2
    assert rows[0]["event"] == "reconcile.boot"
    assert json.loads(rows[0]["detail"]) == {"status": "ok", "diff": None}
    assert rows[0]["ts"] == "2024-03-13T14:30:00+00:00"
    assert rows[1]["event"] == "hitl.ack"
