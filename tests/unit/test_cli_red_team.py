"""Smoke test for the `firm red-team` CLI command."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path



_REPO_ROOT = Path(__file__).parent.parent.parent
_CORPUS_PATH = _REPO_ROOT / "tests" / "red_team" / "corpus.jsonl"


def _expected_case_count() -> int:
    """Derive expected count from corpus to avoid hardcoding (T07.h added a case)."""
    return sum(
        1 for line in _CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def test_red_team_command_runs_and_reports_pass_count() -> None:
    """Invoke `python -m firm.cli red-team` end-to-end. Confirm:
      - exit code 0 (all corpus tests pass under the current harness)
      - stdout includes the "<N>/<N> passed" summary derived from the corpus
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
    n = _expected_case_count()
    assert result.returncode == 0, f"red-team CLI exited non-zero:\n{combined}"
    assert f"/{n} passed" in combined, f"missing '/{n} passed' summary:\n{combined}"
    assert f"{n}/{n} passed" in combined, f"expected '{n}/{n} passed' but got:\n{combined}"
