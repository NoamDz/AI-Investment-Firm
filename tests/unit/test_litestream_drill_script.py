"""Tests for T23 firm/ops/litestream_drill.py — script-level behavior."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
SCRIPT = ROOT / "firm" / "ops" / "litestream_drill.py"


def test_drill_script_exists_and_executes():
    """Sanity: the script file exists and is at least syntactically valid Python."""
    assert SCRIPT.exists()
    compile(SCRIPT.read_text(encoding="utf-8"), str(SCRIPT), "exec")


def test_drill_wal_size_check_passes_when_no_wal(tmp_path, monkeypatch):
    """No data/firm.db-wal present → drill should NOT fail on the WAL check."""
    # Run the script from an isolated cwd with no data/ tree at all.
    env = dict(os.environ)
    env["FIRM_LITESTREAM_DRILL_SKIP_REPLICATE"] = "1"  # ← skip the docker part
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Exit 0 expected: no WAL file → vacuous pass; replicate phase skipped via env.
    assert result.returncode == 0, (result.stdout, result.stderr)


def test_drill_wal_size_check_flags_oversized_wal(tmp_path, monkeypatch):
    """If data/firm.db-wal exceeds 16 MB, drill must exit non-zero."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    big_wal = data_dir / "firm.db-wal"
    # Write a 17 MB file (>16 MB ceiling)
    big_wal.write_bytes(b"\x00" * (17 * 1024 * 1024))
    env = dict(os.environ)
    env["FIRM_LITESTREAM_DRILL_SKIP_REPLICATE"] = "1"
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "WAL" in (result.stderr + result.stdout)
