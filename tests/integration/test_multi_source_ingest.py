"""Integration tests for T21 multi-source ingest.

Verifies:
- Source payload presence: all chunks carry payload["source"] matching the source name.
- Order independence: two sources ingested in reverse order yield the same collection.
- Idempotency: a second ingest run on the same source adds zero new chunks.
"""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qdrant_client.local.qdrant_local")

from qdrant_client import QdrantClient  # noqa: E402

from firm.core.clock import ReplayClock  # noqa: E402
from firm.core.config import (  # noqa: E402
    ChunkConfig,
    ContextualConfig,
    CorpusConfig,
    EmbeddingConfig,
    FinanceBenchCorpusConfig,
    QdrantConfig,
    RagConfig,
    RerankConfig,
    RetrievalConfig,
)
from firm.db.migrations import init_db  # noqa: E402
from firm.rag.chunk import Chunk  # noqa: E402
from firm.rag.ingest import run_ingest  # noqa: E402
from firm.rag.qdrant_store import VectorStore  # noqa: E402
from firm.rag.source import FilingDoc  # noqa: E402


# ---------------------------------------------------------------------------
# Fake collaborators (mirrors test_ingest_pipeline.py pattern)
# ---------------------------------------------------------------------------


class FakeSource:
    """In-memory CorpusSource backed by an explicit list of docs."""

    def __init__(self, name: str, docs: list[FilingDoc]) -> None:
        self.name = name
        self._docs = docs

    def iter_docs(self) -> Iterator[FilingDoc]:
        yield from self._docs


class FakeAugmenter:
    def augment(self, doc: FilingDoc, chunks: list[Chunk]) -> list[Chunk]:
        for chunk in chunks:
            chunk.doc_summary = "<summary>"
        return chunks


class FakeEmbedder:
    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i in range(len(texts)):
            out[i, i % 4] = 1.0
        return out


class FakeSparse:
    def transform(self, text: str) -> dict[int, float]:
        return {0: 1.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rag_config(collection: str = "test_chunks") -> RagConfig:
    return RagConfig(
        corpus=CorpusConfig(
            financebench=FinanceBenchCorpusConfig(split="train", max_docs=None),
        ),
        chunk=ChunkConfig(target_tokens=64, overlap_tokens=8),
        embedding=EmbeddingConfig(dense_model="fake-dense", dense_dim=4, sparse="bm25"),
        retrieval=RetrievalConfig(top_k_retrieve=8, top_k_rerank=4),
        rerank=RerankConfig(model="fake-rerank", score_floor=0.3),
        contextual=ContextualConfig(summary_model="fake-haiku"),
        qdrant=QdrantConfig(collection=collection, url_env="QDRANT_URL"),
    )


def _make_doc(doc_id: str, ticker: str = "AAPL") -> FilingDoc:
    body = (
        "<html><body>"
        f"<p>{ticker} reported strong quarterly performance across product lines. "
        "Revenue increased materially, and operating margin expanded year over year. "
        "Management reiterated full-year guidance and highlighted demand momentum. "
        "Segment results included growth in services and wearables, with hardware steady. "
        "The company continued capital returns through share repurchases and dividends.</p>"
        "</body></html>"
    )
    return FilingDoc(
        doc_id=doc_id,
        ticker=ticker,
        filing_type="10-K",
        published_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
        title=f"{ticker} Annual Report",
        html=body,
    )


def _scroll_all(client: QdrantClient, name: str) -> list[dict]:
    points, _ = client.scroll(
        collection_name=name,
        limit=200,
        with_payload=True,
        with_vectors=False,
    )
    return [p.payload for p in points if p.payload is not None]


def _run(
    source: FakeSource,
    store: VectorStore,
    db_path: Path,
    rag_config: RagConfig,
) -> None:
    clock = ReplayClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))
    run_ingest(
        source=source,
        store=store,
        embedder=FakeEmbedder(),
        sparse=FakeSparse(),
        augmenter=FakeAugmenter(),
        db_path=db_path,
        clock=clock,
        rag_config=rag_config,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_two_source_ingest_writes_source_payload(tmp_path: Path) -> None:
    """Both sources land chunks in the same collection; payloads carry source."""
    db_path = tmp_path / "firm.db"
    init_db(db_path)

    client = QdrantClient(":memory:")
    store = VectorStore(client)
    rag_config = _make_rag_config()
    store.create_collection(rag_config.qdrant.collection, dense_dim=4)

    alpha = FakeSource("alpha", [_make_doc("doc-alpha-001", "AAPL")])
    beta = FakeSource("beta", [_make_doc("doc-beta-001", "MSFT")])

    _run(alpha, store, db_path, rag_config)
    _run(beta, store, db_path, rag_config)

    payloads = _scroll_all(client, rag_config.qdrant.collection)
    assert len(payloads) > 0, "Expected at least one chunk in collection"

    sources_found = {p["source"] for p in payloads}
    assert "alpha" in sources_found, "Expected at least one chunk with source='alpha'"
    assert "beta" in sources_found, "Expected at least one chunk with source='beta'"
    assert sources_found == {"alpha", "beta"}, f"Unexpected sources: {sources_found}"


def test_ingest_order_independent(tmp_path: Path) -> None:
    """Same final collection state regardless of source ingest order."""
    db1 = tmp_path / "firm1.db"
    db2 = tmp_path / "firm2.db"
    init_db(db1)
    init_db(db2)

    client1 = QdrantClient(":memory:")
    store1 = VectorStore(client1)
    rag_config1 = _make_rag_config("col1")
    store1.create_collection("col1", dense_dim=4)

    client2 = QdrantClient(":memory:")
    store2 = VectorStore(client2)
    rag_config2 = _make_rag_config("col2")
    store2.create_collection("col2", dense_dim=4)

    # Collection 1: alpha then beta
    alpha1 = FakeSource("alpha", [_make_doc("doc-alpha-001", "AAPL")])
    beta1 = FakeSource("beta", [_make_doc("doc-beta-001", "MSFT")])
    _run(alpha1, store1, db1, rag_config1)
    _run(beta1, store1, db1, rag_config1)

    # Collection 2: beta then alpha
    alpha2 = FakeSource("alpha", [_make_doc("doc-alpha-001", "AAPL")])
    beta2 = FakeSource("beta", [_make_doc("doc-beta-001", "MSFT")])
    _run(beta2, store2, db2, rag_config2)
    _run(alpha2, store2, db2, rag_config2)

    payloads1 = _scroll_all(client1, "col1")
    payloads2 = _scroll_all(client2, "col2")

    tuples1 = sorted((p.get("doc_id"), p.get("source"), p.get("chunk_id")) for p in payloads1)
    tuples2 = sorted((p.get("doc_id"), p.get("source"), p.get("chunk_id")) for p in payloads2)

    assert tuples1 == tuples2, (
        "Collection state must be order-independent: "
        f"col1 has {len(tuples1)} entries, col2 has {len(tuples2)}"
    )


def test_ingest_idempotent_second_run_no_new_chunks(tmp_path: Path) -> None:
    """Re-running ingest on the same source produces zero new chunks."""
    db_path = tmp_path / "firm.db"
    init_db(db_path)

    client = QdrantClient(":memory:")
    store = VectorStore(client)
    rag_config = _make_rag_config()
    store.create_collection(rag_config.qdrant.collection, dense_dim=4)

    source = FakeSource("alpha", [_make_doc("doc-alpha-001", "AAPL")])

    # First run.
    clock = ReplayClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))
    first = run_ingest(
        source=source,
        store=store,
        embedder=FakeEmbedder(),
        sparse=FakeSparse(),
        augmenter=FakeAugmenter(),
        db_path=db_path,
        clock=clock,
        rag_config=rag_config,
    )
    assert first.status == "completed"
    assert first.chunks_written > 0
    first_count = first.chunks_written

    payloads_after_first = _scroll_all(client, rag_config.qdrant.collection)
    assert len(payloads_after_first) == first_count

    # Second run with the same source.
    source2 = FakeSource("alpha", [_make_doc("doc-alpha-001", "AAPL")])
    second = run_ingest(
        source=source2,
        store=store,
        embedder=FakeEmbedder(),
        sparse=FakeSparse(),
        augmenter=FakeAugmenter(),
        db_path=db_path,
        clock=clock,
        rag_config=rag_config,
    )
    assert second.status == "completed"
    assert second.docs_completed == 0, "Second run must skip the already-indexed doc"
    assert second.chunks_written == 0, "Second run must write zero chunks"

    payloads_after_second = _scroll_all(client, rag_config.qdrant.collection)
    assert len(payloads_after_second) == first_count, (
        "Collection size must not change after a second ingest of the same docs"
    )
