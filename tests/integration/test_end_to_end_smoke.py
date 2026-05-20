"""Plan 1 walking-skeleton smoke test.

This test exercises the full ``firm run --once`` subprocess path including
the broker, orchestrator, risk, HITL, execution, and reporter nodes.

Plan 2 changes (T29): the ``run`` command now constructs the full LLM/RAG
stack (Qdrant, sentence-transformers models, Anthropic client). When Qdrant
is unreachable or models are absent, the CLI falls back to the Plan 1
deterministic stub, but the PM voter still requires a valid LLM cache or a
live API key.

To keep the smoke test lightweight and green in CI without Qdrant/models, the
test is gated by ``@pytest.mark.requires_models``.  Environments that do NOT
have Qdrant running and models downloaded will skip.  T30 covers the full
grounded heartbeat with fixtures.

Skip strategy: check for Qdrant reachability before launching the subprocess.
If Qdrant is unreachable, ``pytest.skip`` is called so the test does not block
CI.  The marker also lets ``pytest -m 'not requires_models'`` exclude the test
explicitly.
"""
import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest


def _qdrant_reachable() -> bool:
    """Return True if the configured Qdrant endpoint accepts a TCP connection."""
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.mark.requires_models
def test_walking_skeleton_end_to_end(tmp_path: Path):
    """Full heartbeat smoke test (Plan 1 stub path).

    Skipped automatically when Qdrant is not reachable, so the test does not
    block lightweight CI environments.  Run 'docker compose up qdrant' to
    enable.
    """
    if not _qdrant_reachable():
        pytest.skip(
            "Qdrant not reachable — run 'docker compose up qdrant' or"
            " set QDRANT_URL. T30 covers the grounded path with fixtures."
        )

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
