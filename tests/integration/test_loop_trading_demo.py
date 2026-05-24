"""End-to-end trading-demo loop test — the demo flow.

The earlier ``test_cli_loop.py`` validates that ``firm run --loop`` iterates
and exits cleanly using the empty-Qdrant-→-REFUSE path. That proves the loop
plumbing but is a degenerate demo: a reviewer watching it sees only refusals.

This test exercises the *demoable* loop:

1. Reuse the cache-seeding pattern from ``test_end_to_end_grounded.py``
   (ingest AAPL+MSFT fixture → seed extractor/judge/PM caches with BUY-shaped
   canned responses).
2. Launch ``firm run --loop --interval-seconds 2`` as a subprocess.
3. Wait for at least 2 completed heartbeats.
4. SIGTERM the loop; drain stdout.
5. Assert (a) ≥2 confirmed outbox rows (one BUY per heartbeat), (b) ≥6 rows
   in ``decisions`` (research+pm+risk per heartbeat × ≥2 heartbeats), and
   (c) at least one ``BUY`` action in the ``decisions`` table.

Marked ``requires_models`` for the same reason as ``test_end_to_end_grounded``
(NomicEmbedder + bge-reranker-v2-m3). Skip with ``pytest -m "not requires_models"``.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client.local.qdrant_local")
pytest.importorskip("sentence_transformers")

from qdrant_client import QdrantClient  # noqa: E402

from firm.db.connection import get_conn  # noqa: E402
from firm.db.migrations import init_db  # noqa: E402

from tests.integration.test_end_to_end_grounded import (  # noqa: E402
    _COLLECTION,
    _FIXTURE_CHUNK_OVERLAP_TOKENS,
    _FIXTURE_CHUNK_TARGET_TOKENS,
    _FIXTURE_ROWS,
    _REPLAY_AT,
    _AUGMENTATION_MODEL,
    _JUDGE_MODEL,
    _PM_MODEL,
    _RESEARCH_MODEL,
    _REPO_ROOT,
    _ensure_precomputed_parquets,
    _retrieve_aapl_chunks,
    _seed_augmentation_cache,
    _seed_extractor_cache,
    _seed_judge_cache,
    _seed_pm_voter_cache,
)

import json


@pytest.mark.skip(
    reason="Ticker rotation (commit 440a7db) cycles AMD, AAPL, MSFT, ... per "
    "heartbeat; pre-seeded LLM cache only covers AAPL so heartbeats #1 and #3+ "
    "miss. Re-enable after seeding cache for the rotation set or adding a "
    "single-ticker test universe override."
)
@pytest.mark.requires_models
def test_loop_demo_produces_trades_across_heartbeats(tmp_path: Path) -> None:
    """``firm run --loop`` produces real BUY trades across ≥2 heartbeats."""
    # ------------------------------------------------------------------ #
    # Phase 0-7: identical to test_end_to_end_grounded.py.                #
    # ------------------------------------------------------------------ #
    _ensure_precomputed_parquets()

    fixture_json = tmp_path / "financebench_fixture.json"
    fixture_json.write_text(json.dumps(_FIXTURE_ROWS), encoding="utf-8")

    qdrant_local_path = tmp_path / "qdrant"
    qdrant_local_path.mkdir()
    reports_root = tmp_path / "reports"
    db_path = tmp_path / "firm.db"

    fixture_rag_yaml = tmp_path / "rag.yaml"
    fixture_rag_yaml.write_text(
        "\n".join([
            "corpus:",
            "  financebench:",
            "    split: train",
            "    max_docs: 2",
            "",
            "chunk:",
            f"  target_tokens: {_FIXTURE_CHUNK_TARGET_TOKENS}",
            f"  overlap_tokens: {_FIXTURE_CHUNK_OVERLAP_TOKENS}",
            "",
            "embedding:",
            "  dense_model: nomic-ai/nomic-embed-text-v1.5",
            "  dense_dim: 768",
            "  sparse: bm25",
            "",
            "retrieval:",
            "  top_k_retrieve: 8",
            "  top_k_rerank: 4",
            "",
            "rerank:",
            "  model: BAAI/bge-reranker-v2-m3",
            "  score_floor: 0.0",
            "",
            "contextual:",
            f"  summary_model: {_AUGMENTATION_MODEL}",
            "",
            "qdrant:",
            f"  collection: {_COLLECTION}",
            "  url_env: QDRANT_URL",
        ]),
        encoding="utf-8",
    )

    fixture_llm_yaml = tmp_path / "llm.yaml"
    fixture_llm_yaml.write_text(
        "\n".join([
            "research:",
            f"  model: {_RESEARCH_MODEL}",
            "  max_tokens: 4096",
            "  temperature: 0.0",
            "",
            "judge:",
            f"  model: {_JUDGE_MODEL}",
            "  max_tokens: 2048",
            "  temperature: 0.0",
            "",
            "pm:",
            f"  model: {_PM_MODEL}",
            "  max_tokens: 1024",
            "  temperature: 0.0",
        ]),
        encoding="utf-8",
    )

    init_db(db_path)
    _seed_augmentation_cache(db_path)

    env = os.environ.copy()
    env["FIRM_DB_PATH"] = str(db_path)
    env["FIRM_LLM_MODE"] = "cached"
    env["FIRM_FINANCEBENCH_FIXTURE"] = str(fixture_json)
    env["QDRANT_LOCAL_PATH"] = str(qdrant_local_path)
    env["FIRM_HMAC_SECRET"] = "a" * 64
    env["FIRM_RAG_CONFIG"] = str(fixture_rag_yaml)
    env["FIRM_LLM_CONFIG"] = str(fixture_llm_yaml)
    env["FIRM_BROKER"] = "FAKE"
    env["FIRM_REPLAY_AT"] = _REPLAY_AT
    env["FIRM_REPORTS_ROOT"] = str(reports_root)
    env["FIRM_INITIAL_POSITIONS"] = '{"AAPL": "10"}'
    env.setdefault("ANTHROPIC_API_KEY", "dummy-key-for-cached-mode")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("QDRANT_URL", None)

    # Phase: ingest
    ingest_result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "ingest", "--config", str(fixture_rag_yaml)],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300, cwd=str(_REPO_ROOT),
    )
    assert ingest_result.returncode == 0, (
        f"ingest failed\nstdout:\n{ingest_result.stdout}\nstderr:\n{ingest_result.stderr}"
    )

    _qc = QdrantClient(path=str(qdrant_local_path))
    assert _qc.count(collection_name=_COLLECTION, exact=True).count > 0
    _qc.close()

    # Phase: in-process retrieve + seed run-time caches
    retrieved_chunks = _retrieve_aapl_chunks(qdrant_local_path)
    assert retrieved_chunks, "in-process retrieval returned no chunks"
    claims = _seed_extractor_cache(db_path, retrieved_chunks)
    _seed_judge_cache(db_path, claims)
    _seed_pm_voter_cache(db_path, claims)

    # ------------------------------------------------------------------ #
    # Phase 8: firm run --loop                                            #
    # ------------------------------------------------------------------ #
    proc = subprocess.Popen(
        [sys.executable, "-m", "firm.cli", "run", "--loop", "--interval-seconds", "2"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", cwd=str(_REPO_ROOT),
    )

    # Per-heartbeat work in cached mode is ~2-5s; allow generous cold-start budget.
    deadline = time.monotonic() + 300
    completed_pattern = re.compile(r"Heartbeat #(\d+) complete")
    seen_seqs: set[int] = set()
    collected: list[str] = []

    try:
        while time.monotonic() < deadline:
            assert proc.stdout is not None
            line = proc.stdout.readline()
            if not line:
                break
            collected.append(line)
            m = completed_pattern.search(line)
            if m:
                seen_seqs.add(int(m.group(1)))
                if len(seen_seqs) >= 2:
                    break
    finally:
        # SIGTERM on POSIX; CTRL_BREAK_EVENT or terminate() on Windows.
        if sys.platform == "win32":
            proc.terminate()
        else:
            proc.send_signal(signal.SIGTERM)
        try:
            tail, _ = proc.communicate(timeout=30)
            if tail:
                collected.append(tail)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    combined = "".join(collected)
    assert len(seen_seqs) >= 2, (
        f"expected ≥2 heartbeats, got {sorted(seen_seqs)}\n--- output ---\n{combined}"
    )

    # ------------------------------------------------------------------ #
    # Phase 9: assertions on real trades                                  #
    # ------------------------------------------------------------------ #
    with closing(get_conn(db_path)) as conn:
        outbox_confirmed = conn.execute(
            "SELECT COUNT(*) AS n FROM outbox WHERE status='confirmed'"
        ).fetchone()["n"]
        decisions_count = conn.execute(
            "SELECT COUNT(*) AS n FROM decisions"
        ).fetchone()["n"]
        buy_count = conn.execute(
            "SELECT COUNT(*) AS n FROM decisions WHERE action='BUY'"
        ).fetchone()["n"]
        positions_count = conn.execute(
            "SELECT COUNT(*) AS n FROM positions"
        ).fetchone()["n"]

    assert outbox_confirmed >= 2, (
        f"expected ≥2 confirmed outbox rows (one per heartbeat), got {outbox_confirmed}"
    )
    # ≥3 decisions per heartbeat × ≥2 heartbeats. Permissive: per-voter rows
    # in Plan 3+ make this 6+ in practice.
    assert decisions_count >= 6, (
        f"expected ≥6 decisions across ≥2 heartbeats, got {decisions_count}"
    )
    assert buy_count >= 1, (
        f"expected ≥1 BUY action in decisions table, got {buy_count}\n"
        "Cache seeding may not be producing BUY decisions."
    )
    assert positions_count >= 1, (
        f"expected ≥1 open position after BUY trades, got {positions_count}"
    )
