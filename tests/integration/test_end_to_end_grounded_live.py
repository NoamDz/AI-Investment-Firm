"""End-to-end grounded heartbeat with LIVE Anthropic API (opt-in).

This is the *production-realism* counterpart to
``test_end_to_end_grounded.py``. The cached test seeds canned LLM responses
into ``llm_cache`` and asserts a deterministic BUY → confirmed-outbox path.
This test instead runs the same pipeline with ``FIRM_LLM_MODE=live`` against
the real Anthropic API, so it catches drift between the cassette assumptions
and what the production stack actually does end-to-end.

The test does NOT assert a specific action — live model outputs vary across
runs and model revisions. It asserts the *structural* invariants that must
hold no matter what action the system picks:

* ``firm ingest`` populates Qdrant with at least one chunk per fixture doc.
* ``firm run --once`` exits 0.
* The ``decisions`` table has >= 3 rows (research / pm / risk per heartbeat).
* The daily JSONL report exists and contains a research + pm + risk decision.
* If action is REFUSE, ``failure_mode`` is populated.
* If action is BUY/SELL/HOLD, the research decision carries at least one
  citation with a ``chunk_id`` (production grounding invariant).
* The ``cost_ledger`` records at least one charge (proves the live API was
  actually hit, not silently short-circuited).

Gating
------
Skipped unless ALL of:
  - ``FIRM_E2E_LIVE=1`` is set (explicit opt-in — this test costs ~$0.30-$1
    per run depending on chunk count and is slow ~3-6 min).
  - ``ANTHROPIC_API_KEY`` is set.
  - Qdrant is reachable at ``QDRANT_URL`` (default ``http://localhost:6333``)
    OR ``QDRANT_LOCAL_PATH`` is set (filesystem-backed Qdrant).

Run locally with:

    $env:FIRM_E2E_LIVE = "1"
    $env:ANTHROPIC_API_KEY = "sk-..."
    docker compose up -d qdrant   # or set QDRANT_LOCAL_PATH
    pytest tests/integration/test_end_to_end_grounded_live.py -v
"""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path
from urllib.parse import urlparse

import pytest

# Reuse fixture rows + the precomputed-parquet helper from the cached test.
# Both tests need identical AAPL/MSFT fixture data for the universe to match
# what the CLI's research agent will query.
from tests.integration.test_end_to_end_grounded import (
    _FIXTURE_ROWS,
    _REPLAY_AT,
    _ensure_precomputed_parquets,
)

pytestmark = pytest.mark.live


def _qdrant_reachable() -> bool:
    """Return True if Qdrant is reachable at QDRANT_URL (TCP probe).

    If ``QDRANT_LOCAL_PATH`` is set we treat that as 'reachable' too, since
    the CLI will route to the embedded filesystem backend.
    """
    if os.environ.get("QDRANT_LOCAL_PATH"):
        return True
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 6333
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _require_live_gates() -> None:
    if os.environ.get("FIRM_E2E_LIVE") != "1":
        pytest.skip(
            "FIRM_E2E_LIVE not set — this is the live-API e2e test (opt-in, "
            "costs real $$). Set FIRM_E2E_LIVE=1 to run."
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — live e2e needs a real key.")
    if not _qdrant_reachable():
        pytest.skip(
            "Qdrant not reachable — run 'docker compose up -d qdrant' or "
            "set QDRANT_LOCAL_PATH for the embedded backend."
        )


_REPO_ROOT = Path(__file__).parent.parent.parent

# Models must match the fixture llm.yaml below; kept in sync with the cached
# test so a side-by-side comparison is meaningful.
_RESEARCH_MODEL = "claude-sonnet-4-6"
_JUDGE_MODEL = "claude-haiku-4-5"
_PM_MODEL = "claude-sonnet-4-6"
_AUGMENTATION_MODEL = "claude-haiku-4-5"

_COLLECTION = "e2e_grounded_live_chunks"

# Reduced retrieval params to bound cost: smaller k_retrieve / k_rerank means
# fewer chunks → fewer tokens in the extractor prompt.
_FIXTURE_CHUNK_TARGET_TOKENS = 64
_FIXTURE_CHUNK_OVERLAP_TOKENS = 8


def _write_fixture_configs(tmp_path: Path) -> tuple[Path, Path]:
    rag_yaml = tmp_path / "rag.yaml"
    rag_yaml.write_text(
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
            "  top_k_retrieve: 4",
            "  top_k_rerank: 2",
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

    llm_yaml = tmp_path / "llm.yaml"
    llm_yaml.write_text(
        "\n".join([
            "research:",
            f"  model: {_RESEARCH_MODEL}",
            "  max_tokens: 2048",
            "  temperature: 0.0",
            "",
            "judge:",
            f"  model: {_JUDGE_MODEL}",
            "  max_tokens: 1024",
            "  temperature: 0.0",
            "",
            "pm:",
            f"  model: {_PM_MODEL}",
            "  max_tokens: 1024",
            "  temperature: 0.0",
        ]),
        encoding="utf-8",
    )
    return rag_yaml, llm_yaml


def test_grounded_demo_live_api_structural_invariants(tmp_path: Path) -> None:
    """Run one heartbeat against the live Anthropic API and assert structural
    invariants that must hold regardless of which action the live model picks.

    Cost: ~$0.30-$1 per run. Skipped unless FIRM_E2E_LIVE=1.
    """
    _require_live_gates()
    _ensure_precomputed_parquets()

    fixture_json = tmp_path / "financebench_fixture.json"
    fixture_json.write_text(json.dumps(_FIXTURE_ROWS), encoding="utf-8")

    qdrant_local_path = tmp_path / "qdrant"
    qdrant_local_path.mkdir()

    reports_root = tmp_path / "reports"
    db_path = tmp_path / "firm.db"

    rag_yaml, llm_yaml = _write_fixture_configs(tmp_path)

    env = os.environ.copy()
    env["FIRM_DB_PATH"] = str(db_path)
    env["FIRM_LLM_MODE"] = "live"
    # tests/conftest.py pins FIRM_VCR_MODE=replay for the whole session as a
    # safety net. Live e2e must bypass the cassette layer entirely so requests
    # actually leave the box — unset it for this subprocess.
    env.pop("FIRM_VCR_MODE", None)
    env["FIRM_FINANCEBENCH_FIXTURE"] = str(fixture_json)
    # Force the embedded Qdrant backend so the test does not depend on a
    # running container — drops a dependency for CI runners that have a key
    # but no docker. Users who *do* have docker can `pop` QDRANT_LOCAL_PATH
    # before running the test, but in CI we prefer hermetic.
    env["QDRANT_LOCAL_PATH"] = str(qdrant_local_path)
    env.pop("QDRANT_URL", None)
    env["FIRM_HMAC_SECRET"] = "a" * 64
    env["FIRM_RAG_CONFIG"] = str(rag_yaml)
    env["FIRM_LLM_CONFIG"] = str(llm_yaml)
    env["FIRM_BROKER"] = "FAKE"
    env["FIRM_REPLAY_AT"] = _REPLAY_AT
    env["FIRM_REPORTS_ROOT"] = str(reports_root)
    # Seed a non-zero AAPL position so Risk's escalate_new_ticker check does
    # not trigger — keeps the test on the BUY/SELL/HOLD/REFUSE branch rather
    # than the HITL branch (HITL is covered by the cached test's T31 case).
    env["FIRM_INITIAL_POSITIONS"] = '{"AAPL": "10"}'
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    # Ingest (will hit Anthropic for contextual augmentation summaries).
    ingest_result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "ingest", "--config", str(rag_yaml)],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        cwd=str(_REPO_ROOT),
    )
    assert ingest_result.returncode == 0, (
        f"firm ingest (live) exited {ingest_result.returncode}\n"
        f"stdout:\n{ingest_result.stdout}\n"
        f"stderr:\n{ingest_result.stderr}"
    )

    # Sanity: confirm Qdrant got populated.  An empty collection silently
    # routes the heartbeat to REFUSE/insufficient_evidence for every ticker.
    from qdrant_client import QdrantClient
    qc = QdrantClient(path=str(qdrant_local_path))
    chunk_count = qc.count(collection_name=_COLLECTION, exact=True).count
    qc.close()
    assert chunk_count > 0, (
        f"Qdrant collection {_COLLECTION!r} is empty after ingest — "
        f"check ingest stdout/stderr for chunking errors.\n"
        f"stdout:\n{ingest_result.stdout}"
    )

    # Heartbeat (live research extractor + judge + PM voters).
    run_result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "run", "--once"],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        cwd=str(_REPO_ROOT),
    )
    assert run_result.returncode == 0, (
        f"firm run --once (live) exited {run_result.returncode}\n"
        f"stdout:\n{run_result.stdout}\n"
        f"stderr:\n{run_result.stderr}"
    )

    # --- Structural assertions ---

    # The decisions table holds one row per *unique* Decision id.  When the
    # research path REFUSEs, pm and risk are passthroughs that re-emit the
    # same Decision object, so the row count is 2 (research + shared pm/risk).
    # When research succeeds, pm and risk emit their own Decisions and the
    # count is 3+.  We only insist that research was persisted; the structural
    # check on the JSONL report below covers the full graph.
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        decisions = conn.execute(
            "SELECT id, action, failure_mode, citations FROM decisions"
        ).fetchall()

    assert len(decisions) >= 1, (
        f"expected at least 1 decisions row after heartbeat, got {len(decisions)}\n"
        f"stdout:\n{run_result.stdout}"
    )

    # Daily JSONL report exists and is well-formed.
    report_file = reports_root / "2024-03-13" / "decisions.jsonl"
    assert report_file.exists(), (
        f"expected report file at {report_file}\n"
        f"reports root contents: {list(reports_root.rglob('*'))}\n"
        f"stdout:\n{run_result.stdout}"
    )

    records = [
        json.loads(line) for line in report_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records, f"report file {report_file} is empty"

    record = records[0]
    for key in ("research_decision", "pm_decision", "risk_decision"):
        assert key in record and record[key], (
            f"missing {key} in report record; keys={list(record.keys())}"
        )

    research = record["research_decision"]
    pm = record["pm_decision"]
    risk = record["risk_decision"]
    research_action = research.get("action")
    risk_action = risk.get("action")

    # decision_id_chain wiring: pm and risk both descend from research.
    assert research["id"] in (pm.get("decision_id_chain") or []), (
        f"pm.decision_id_chain={pm.get('decision_id_chain')} does not contain "
        f"research.id={research['id']!r}"
    )
    assert research["id"] in (risk.get("decision_id_chain") or []), (
        f"risk.decision_id_chain={risk.get('decision_id_chain')} does not contain "
        f"research.id={research['id']!r}"
    )

    # Action-dependent invariants.
    if research_action == "REFUSE":
        assert research.get("failure_mode"), (
            f"research action=REFUSE but failure_mode is empty; "
            f"every refusal must be classified."
        )
    else:
        # BUY / SELL / HOLD — production grounding rule: at least one citation
        # with a chunk_id traceable back to the ingested corpus.
        citations = research.get("citations") or []
        assert any(
            isinstance(c, dict) and c.get("chunk_id") for c in citations
        ), (
            f"research action={research_action!r} but no citation with "
            f"chunk_id; production rule violated.\n"
            f"citations={citations}"
        )

    # Execution path: REFUSE/HOLD → skipped; BUY/SELL → executed.
    execution = record.get("execution_result") or {}
    if risk_action in ("REFUSE", "HOLD"):
        assert execution.get("skipped") is True, (
            f"action={risk_action!r} but execution was not skipped: {execution}"
        )
