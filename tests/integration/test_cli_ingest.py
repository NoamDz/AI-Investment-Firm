"""Integration test for `firm ingest` CLI subcommand (T28).

Design notes
------------
The test uses the real NomicEmbedder backed by sentence-transformers, so it
requires the model to be downloaded (or cached).  It is therefore marked
``@pytest.mark.requires_models`` -- skip with ``pytest -m "not requires_models"``
if the environment has no network access or no disk cache.

The test avoids the HuggingFace dataset loader entirely by setting
``FIRM_FINANCEBENCH_FIXTURE`` to a local JSON fixture file.  The CLI ingest
command reads this env var and, when set, loads rows from the local file instead
of calling ``datasets.load_dataset``.

Qdrant is configured for local filesystem persistence via ``QDRANT_LOCAL_PATH``
(also a test-only env var).  Production uses ``QDRANT_URL``.

The LLM augmentation step is tested with ``FIRM_LLM_MODE=cached`` and a
pre-seeded SQLite ``llm_cache`` table.  Rather than duplicating the prompt
template string, we drive a real ``ContextualAugmenter`` with a capturing
stub client to extract the exact prompts (including en-dashes and whitespace)
and then hash-insert them into llm_cache.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("qdrant_client.local.qdrant_local")
pytest.importorskip("sentence_transformers")

from qdrant_client import QdrantClient  # noqa: E402

from firm.core.clock import ReplayClock  # noqa: E402
from firm.db.connection import get_conn  # noqa: E402
from firm.db.migrations import init_db  # noqa: E402
from firm.llm.cache import LlmCache, hash_prompt  # noqa: E402
from firm.llm.client import CompletionResponse  # noqa: E402
from firm.rag.chunk import Chunk, chunk_document  # noqa: E402
from firm.rag.contextual import ContextualAugmenter  # noqa: E402
from firm.rag.source import FilingDoc  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FIXTURE_CLOCK = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))

# Chunk parameters used in BOTH the fixture rag.yaml (the subprocess reads this
# config) AND in _seed_llm_cache's chunk_document call (the prompts the cache
# is keyed on depend on the chunks produced).  Keeping them in one place
# guarantees the two ends stay synchronized; otherwise a YAML edit alone would
# silently invalidate the cache hashes and the subprocess would cache-miss.
_FIXTURE_CHUNK_TARGET_TOKENS = 64
_FIXTURE_CHUNK_OVERLAP_TOKENS = 8

_FIXTURE_ROWS: list[dict[str, object]] = [
    {
        "doc_name": "AAPL_2024_10K_fixture",
        "filing_date": "2024-01-01",
        "company": "Apple Inc.",
        "doc_type": "10-K",
        "text": (
            "<html><body>"
            "<p>Apple Inc. reported strong quarterly performance across product lines. "
            "Revenue increased materially, and operating margin expanded year over year. "
            "Management reiterated full-year guidance and highlighted demand momentum. "
            "Segment results included growth in services and wearables, with hardware steady. "
            "The company continued capital returns through share repurchases and dividends.</p>"
            "</body></html>"
        ),
    },
    {
        "doc_name": "MSFT_2024_10K_fixture",
        "filing_date": "2024-01-02",
        "company": "Microsoft Corporation",
        "doc_type": "10-K",
        "text": (
            "<html><body>"
            "<p>Microsoft Corporation delivered robust cloud revenue growth in Azure. "
            "Commercial cloud revenues reached record levels driven by enterprise demand. "
            "Operating income grew alongside disciplined cost management and margin expansion. "
            "The intelligent cloud segment was a primary growth driver for the quarter. "
            "Management expects continued momentum in AI-driven productivity tools.</p>"
            "</body></html>"
        ),
    },
]


class _CapturingClient:
    """AnthropicClient stub that captures the exact call args without hitting the API.

    Implements the AnthropicClient protocol (T9/T16) so ContextualAugmenter
    accepts it.  Each call to complete() records system + messages so we can
    compute the canonical hash that CachedAnthropicClient will use.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        model: str,
        system: str | None,
        messages: Sequence[dict[str, object]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> CompletionResponse:
        # CachedAnthropicClient.complete() coerces None -> "" before hashing.
        effective_system = system if system is not None else ""
        self.calls.append({
            "system": effective_system,
            "messages": list(messages),
            "model": model,
        })
        return CompletionResponse(text="stub", input_tokens=0, output_tokens=0, cache_hit=False)


def _row_to_filing_doc(row: dict[str, object]) -> FilingDoc:
    """Mirror the logic in FinanceBenchSource / _row_to_filing_doc."""
    doc_name = str(row.get("doc_name", ""))
    filing_date_raw = str(row.get("filing_date", ""))
    dt = datetime.fromisoformat(filing_date_raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ticker = str(row.get("company", ""))
    doc_type = str(row.get("doc_type", ""))
    text = str(row.get("text") or row.get("pdf_content") or "")
    title = f"{ticker} {doc_type} {filing_date_raw}"
    return FilingDoc(
        doc_id=doc_name,
        ticker=ticker,
        filing_type=doc_type,
        published_at=dt,
        title=title,
        html=text,
    )


def _seed_llm_cache(db_path: Path, model: str) -> None:
    """Pre-seed llm_cache using a capturing stub driven through ContextualAugmenter.

    This avoids duplicating the prompt template string (including en-dashes,
    whitespace, etc.) in the test -- instead we run the real augmenter with a
    no-op client that records all (system, messages) pairs, then insert the
    exact prompt hashes into the cache table.
    """
    capturing_client: _CapturingClient = _CapturingClient()
    augmenter = ContextualAugmenter(
        client=capturing_client,
        model=model,
    )
    cache = LlmCache(db_path=db_path, clock=_FIXTURE_CLOCK)

    for row in _FIXTURE_ROWS:
        doc = _row_to_filing_doc(row)
        chunks = chunk_document(
            doc,
            target_tokens=_FIXTURE_CHUNK_TARGET_TOKENS,
            overlap_tokens=_FIXTURE_CHUNK_OVERLAP_TOKENS,
            source="financebench",
        )
        if not chunks:
            # Safety fallback: synthesize one chunk so augment() fires.
            chunks = [
                Chunk(
                    id=f"{doc.doc_id}-0",
                    doc_id=doc.doc_id,
                    ticker=doc.ticker,
                    section="body",
                    published_at=doc.published_at,
                    text=doc.html[:200],
                    char_span=(0, 200),
                    token_count=10,
                    source="financebench",
                )
            ]
        augmenter.augment(doc, list(chunks))

    for call in capturing_client.calls:
        prompt_hash = hash_prompt(
            system=call["system"],
            messages=call["messages"],
            tools=None,
        )
        response_dict: dict[str, object] = {
            "content": [{"type": "text", "text": "Synthetic fixture summary."}],
            "model": model,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }
        cache.put(
            prompt_hash=prompt_hash,
            model=call["model"],
            response_json=json.dumps(response_dict),
            input_tokens=10,
            output_tokens=10,
        )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.requires_models
def test_cli_ingest_runs_against_fixture_corpus_and_writes_collection(
    tmp_path: Path,
) -> None:
    """Invoke `firm ingest` as a subprocess against a 2-doc fixture corpus.

    Requires sentence-transformers (NomicEmbedder) to be installed and the
    nomic-embed-text-v1.5 model to be present in the local HF cache.  The
    test is marked requires_models and skipped when models are unavailable.
    """
    # --- fixture file layout ---
    fixture_json = tmp_path / "financebench_fixture.json"
    fixture_json.write_text(json.dumps(_FIXTURE_ROWS), encoding="utf-8")

    qdrant_local_path = tmp_path / "qdrant"
    qdrant_local_path.mkdir()

    db_path = tmp_path / "firm.db"

    # --- write fixture rag.yaml ---
    # dense_dim=768 matches the real NomicEmbedder.  Collection name is unique.
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
            "  score_floor: 0.3",
            "",
            "contextual:",
            "  summary_model: claude-haiku-4-5",
            "",
            "qdrant:",
            "  collection: test_chunks",
            "  url_env: QDRANT_URL",
        ]),
        encoding="utf-8",
    )

    # --- init DB and seed llm_cache ---
    init_db(db_path)
    _seed_llm_cache(db_path, model="claude-haiku-4-5")

    # --- build subprocess environment ---
    env = os.environ.copy()
    env["FIRM_DB_PATH"] = str(db_path)
    env["FIRM_LLM_MODE"] = "cached"
    env["FIRM_FINANCEBENCH_FIXTURE"] = str(fixture_json)
    env["QDRANT_LOCAL_PATH"] = str(qdrant_local_path)
    env["FIRM_HMAC_SECRET"] = "a" * 64
    # Prevent the CLI from trying to connect to any live Qdrant instance.
    env.pop("QDRANT_URL", None)

    # --- run the CLI ---
    result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "ingest", "--config", str(fixture_rag_yaml)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"firm ingest exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # --- assert ingest_runs row ---
    with closing(get_conn(db_path)) as conn:
        rows = conn.execute(
            "SELECT status, docs_completed, chunks_written FROM ingest_runs"
        ).fetchall()
    assert len(rows) == 1, f"expected 1 ingest_run row, got {len(rows)}"
    run_row = rows[0]
    assert run_row["status"] == "completed", f"ingest_run status={run_row['status']!r}"
    assert run_row["docs_completed"] > 0, "no docs completed"
    assert run_row["chunks_written"] > 0, "no chunks written"

    # --- assert chunks in Qdrant ---
    client = QdrantClient(path=str(qdrant_local_path))
    count = client.count(collection_name="test_chunks", exact=True).count
    assert count > 0, "Qdrant collection is empty after ingest"
    client.close()
