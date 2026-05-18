import os
import subprocess
import sys
from pathlib import Path


def test_cli_run_produces_decision(tmp_path: Path):
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
    # at least one report file written
    reports = list((tmp_path / "reports").rglob("*.jsonl"))
    assert reports, "no report written"
