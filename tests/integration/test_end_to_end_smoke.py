import os
import subprocess
import sys
from pathlib import Path


def test_walking_skeleton_end_to_end(tmp_path: Path):
    env = os.environ.copy()
    env["FIRM_BROKER"] = "FAKE"
    env["FIRM_DB_PATH"] = str(tmp_path / "firm.db")
    env["FIRM_HMAC_SECRET"] = "a" * 64
    env["FIRM_REPORTS_ROOT"] = str(tmp_path / "reports")
    env["FIRM_REPLAY_AT"] = "2024-03-13T14:30:00+00:00"

    result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "run", "--once"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    # Report file exists
    reports = list((tmp_path / "reports").rglob("*.jsonl"))
    assert reports

    # Outbox has one confirmed row
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "firm.db"))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM outbox WHERE status='confirmed'").fetchall()
    assert len(rows) == 1, f"expected 1 confirmed order, got {len(rows)}"

    # Decisions table has at least research, pm, risk decisions
    decisions = conn.execute("SELECT COUNT(*) c FROM decisions").fetchone()["c"]
    assert decisions >= 3, f"expected at least 3 decisions, got {decisions}"
