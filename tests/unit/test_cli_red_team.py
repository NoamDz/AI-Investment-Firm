"""Smoke test for the `firm red-team` CLI command."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).parent.parent.parent


def test_red_team_command_runs_and_reports_pass_count() -> None:
    """Invoke `python -m firm.cli red-team` end-to-end. Confirm:
      - exit code 0 (since all 50 corpus tests pass under the current harness)
      - stdout includes "X/50 passed" exactly
    """
    # Pin FIRM_VCR_MODE=replay so a developer/CI shell with FIRM_VCR_MODE=record
    # doesn't accidentally trigger live network calls and a misleading failure.
    env = {**os.environ, "FIRM_VCR_MODE": "replay"}
    result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "red-team"],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, f"red-team CLI exited non-zero:\n{combined}"
    assert "/50 passed" in combined, f"missing '/50 passed' summary:\n{combined}"
    # Stricter: the exact 50/50 line should be present.
    assert "50/50 passed" in combined, f"expected '50/50 passed' but got:\n{combined}"
