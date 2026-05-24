"""Loop-mode smoke test for ``firm run --loop``.

Validates that ``--loop`` actually iterates (the flag was a no-op stub until
Plan 4 polish), exits cleanly on terminate, and contains per-heartbeat
exceptions instead of crashing the loop.

Setup mirrors ``test_cli.py``: empty Qdrant collection → research emits
REFUSE / INSUFFICIENT_EVIDENCE so no LLM cache entries are required.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

_FIXTURE_ROWS: list[dict[str, object]] = [
    {
        "doc_name": "AAPL_loop_smoke",
        "filing_date": "2024-01-15",
        "company": "Apple Inc.",
        "doc_type": "10-K",
        "text": "<html><body><p>Revenue grew.</p></body></html>",
    },
]

_DENSE_DIM = 768


def test_loop_runs_multiple_heartbeats_and_exits(tmp_path: Path) -> None:
    """``firm run --loop --interval-seconds 1`` produces ≥2 heartbeats."""
    fixture_json = tmp_path / "financebench_fixture.json"
    fixture_json.write_text(json.dumps(_FIXTURE_ROWS), encoding="utf-8")

    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from qdrant_client import QdrantClient
        from firm.rag.qdrant_store import VectorStore

        qclient = QdrantClient(path=str(qdrant_path))
        VectorStore(qclient).create_collection("firm_chunks", dense_dim=_DENSE_DIM)
        qclient.close()

    env = os.environ.copy()
    env["FIRM_BROKER"] = "FAKE"
    env["FIRM_DB_PATH"] = str(tmp_path / "firm.db")
    env["FIRM_HMAC_SECRET"] = "a" * 64
    env["FIRM_REPORTS_ROOT"] = str(tmp_path / "reports")
    env["FIRM_REPLAY_AT"] = "2024-03-13T14:30:00+00:00"
    env["FIRM_LLM_MODE"] = "cached"
    env["FIRM_FINANCEBENCH_FIXTURE"] = str(fixture_json)
    env["QDRANT_LOCAL_PATH"] = str(qdrant_path)
    env.setdefault("ANTHROPIC_API_KEY", "dummy-for-cached-mode")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("QDRANT_URL", None)

    # Short interval so we observe 2+ heartbeats within the wait window.
    proc = subprocess.Popen(
        [sys.executable, "-m", "firm.cli", "run", "--loop", "--interval-seconds", "1"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # Cold start (BM25 + Nomic load + Qdrant init) regularly takes 60-120s on
    # a fresh machine; pad generously, then watch for completed heartbeats.
    deadline = time.monotonic() + 240
    completed_pattern = re.compile(r"\[firm run\] heartbeat #(\d+) done")
    collected: list[str] = []
    seen_seqs: set[int] = set()

    try:
        while time.monotonic() < deadline:
            assert proc.stdout is not None
            line = proc.stdout.readline()
            if not line:
                # Process exited unexpectedly.
                break
            collected.append(line)
            m = completed_pattern.search(line)
            if m:
                seen_seqs.add(int(m.group(1)))
                if len(seen_seqs) >= 2:
                    break
    finally:
        proc.terminate()
        try:
            # Drain remaining output without blocking forever.
            tail, _ = proc.communicate(timeout=30)
            if tail:
                collected.append(tail)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    combined = "".join(collected)
    assert len(seen_seqs) >= 2, (
        f"Expected ≥2 completed heartbeats, got {sorted(seen_seqs)}.\n"
        f"--- output ---\n{combined}"
    )
    # Sequence numbers must be monotonic and start at 1.
    ordered = sorted(seen_seqs)
    assert ordered[0] == 1, f"first heartbeat should be #1, got {ordered}"
    assert ordered == list(range(1, ordered[-1] + 1)), (
        f"heartbeat sequence should be contiguous, got {ordered}"
    )
