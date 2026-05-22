"""CLI integration smoke-test (Plan 2 grounded path).

Runs ``firm run --once`` as a subprocess with:
- A minimal FinanceBench fixture (2 rows) so BM25 pre-pass has data.
- A local Qdrant path with an empty ``firm_chunks`` collection so retrieval
  returns 0 chunks → research emits REFUSE / INSUFFICIENT_EVIDENCE.
- ``FIRM_LLM_MODE=cached`` so no real Anthropic calls are made.

The PM agent pass-through path (research=REFUSE → no vote) guarantees no
LLM cache entries are required, so the test is fully deterministic and
network-free after the initial module imports.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Minimal FinanceBench fixture rows (flat array, same format as HF dataset)
# ---------------------------------------------------------------------------

_FIXTURE_ROWS: list[dict[str, object]] = [
    {
        "doc_name": "AAPL_2024_10K_cli_smoke",
        "filing_date": "2024-01-15",
        "company": "Apple Inc.",
        "doc_type": "10-K",
        "text": (
            "<html><body>"
            "<p>Apple Inc. reported strong quarterly performance. "
            "Revenue increased materially year over year.</p>"
            "</body></html>"
        ),
    },
    {
        "doc_name": "MSFT_2024_10K_cli_smoke",
        "filing_date": "2024-01-16",
        "company": "Microsoft Corporation",
        "doc_type": "10-K",
        "text": (
            "<html><body>"
            "<p>Microsoft Corporation delivered robust cloud revenue growth.</p>"
            "</body></html>"
        ),
    },
]

# Dense dimension must match what NomicEmbedder produces (768) and what the
# rag.yaml declares.  We use the same value here when creating the collection.
_DENSE_DIM = 768


def test_cli_run_produces_decision(tmp_path: Path) -> None:
    """firm run --once exits 0 and writes at least one JSONL report."""
    # ------------------------------------------------------------------ #
    # Step 1: write fixture JSON                                          #
    # ------------------------------------------------------------------ #
    fixture_json = tmp_path / "financebench_fixture.json"
    fixture_json.write_text(json.dumps(_FIXTURE_ROWS), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Step 2: create a local Qdrant store with an empty firm_chunks       #
    # collection so retrieval returns 0 results (no ingest needed).       #
    # ------------------------------------------------------------------ #
    qdrant_path = tmp_path / "qdrant"
    qdrant_path.mkdir()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from qdrant_client import QdrantClient
        from firm.rag.qdrant_store import VectorStore

        qclient = QdrantClient(path=str(qdrant_path))
        store = VectorStore(qclient)
        store.create_collection("firm_chunks", dense_dim=_DENSE_DIM)
        qclient.close()

    # ------------------------------------------------------------------ #
    # Step 3: subprocess environment                                       #
    # ------------------------------------------------------------------ #
    env = os.environ.copy()
    env["FIRM_BROKER"] = "FAKE"
    env["FIRM_DB_PATH"] = str(tmp_path / "firm.db")
    env["FIRM_HMAC_SECRET"] = "a" * 64
    env["FIRM_REPORTS_ROOT"] = str(tmp_path / "reports")
    env["FIRM_REPLAY_AT"] = "2024-03-13T14:30:00+00:00"
    env["FIRM_LLM_MODE"] = "cached"
    env["FIRM_FINANCEBENCH_FIXTURE"] = str(fixture_json)
    env["QDRANT_LOCAL_PATH"] = str(qdrant_path)
    # ANTHROPIC_API_KEY not needed in cached mode, but from_env() reads it.
    env.setdefault("ANTHROPIC_API_KEY", "dummy-for-cached-mode")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # Ensure QDRANT_URL is not set — QDRANT_LOCAL_PATH takes precedence.
    env.pop("QDRANT_URL", None)

    # T27a: cold-machine cost = BM25 pre-pass + NomicEmbedder._lazy_load
    # (sentence_transformers + torch warmup) + Qdrant init regularly tops 120s.
    # Bumped to 300s. Subprocess isolation means a session-scoped pytest
    # fixture wouldn't help (the warmup re-pays in every subprocess), so the
    # spec's fixture suggestion is dropped — timeout headroom is the lever.
    result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "run", "--once"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert result.returncode == 0, result.stderr

    # At least one report file must have been written.
    reports = list((tmp_path / "reports").rglob("*.jsonl"))
    assert reports, "no report written"
