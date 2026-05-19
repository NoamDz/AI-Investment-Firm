"""Integration tests for the RAG ingest pipeline orchestrator (T11).

A two-doc fixture is composed through the real preprocessor / chunker, with fake
augmenter, embedder, and sparse encoder so the pipeline exercises every seam
without touching the network.

The "writes nothing to Qdrant on failure" assertion is enforced by making the
FIRST doc raise — after that, the run is marked failed before any upsert lands,
so the in-memory Qdrant collection stays empty.
"""
from __future__ import annotations

import sqlite3
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
from firm.rag.qdrant_store import VectorStore  # noqa: E402
from firm.rag.source import FilingDoc  # noqa: E402


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------


class FakeSource:
    """In-memory CorpusSource backed by an explicit list of docs."""

    name = "fake_corpus"

    def __init__(self, docs: list[FilingDoc]) -> None:
        self._docs = docs

    def iter_docs(self) -> Iterator[FilingDoc]:
        yield from self._docs


class FakeAugmenter:
    """Sets chunk.doc_summary without touching any LLM client."""

    def __init__(self, summary: str = "<summary>") -> None:
        self._summary = summary
        self.calls: int = 0

    def augment(self, doc: FilingDoc, chunks: list[Chunk]) -> list[Chunk]:
        self.calls += 1
        for chunk in chunks:
            chunk.doc_summary = self._summary
        return chunks


class FakeEmbedder:
    """Deterministic dense embedder yielding (N, 4) float32 arrays."""

    def __init__(self, *, raise_on_call: int | None = None) -> None:
        self.calls: int = 0
        self._raise_on_call = raise_on_call
        self.last_batch_size: int = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        self.last_batch_size = len(texts)
        if self._raise_on_call is not None and self.calls == self._raise_on_call:
            raise RuntimeError("simulated embedder failure")
        # Deterministic non-zero vectors keyed by call index and text hash.
        out = np.zeros((len(texts), 4), dtype=np.float32)
        for i, _t in enumerate(texts):
            out[i, i % 4] = 1.0
        return out


class FakeSparse:
    """Sparse encoder returning a constant single-term vector."""

    def __init__(self) -> None:
        self.calls: int = 0

    def transform(self, text: str) -> dict[int, float]:
        self.calls += 1
        return {0: 1.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rag_config() -> RagConfig:
    return RagConfig(
        corpus=CorpusConfig(
            financebench=FinanceBenchCorpusConfig(split="train", max_docs=None),
        ),
        chunk=ChunkConfig(target_tokens=64, overlap_tokens=8),
        embedding=EmbeddingConfig(
            dense_model="fake-dense",
            dense_dim=4,
            sparse="bm25",
        ),
        retrieval=RetrievalConfig(top_k_retrieve=8, top_k_rerank=4),
        rerank=RerankConfig(model="fake-rerank", score_floor=0.3),
        contextual=ContextualConfig(summary_model="fake-haiku"),
        qdrant=QdrantConfig(collection="test_chunks", url_env="QDRANT_URL"),
    )


def _make_doc(doc_id: str, ticker: str = "AAPL") -> FilingDoc:
    # The chunker tokenises FilingDoc.html directly; supply enough plain prose
    # to yield at least one chunk through the real preprocessor.
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


def _qdrant_point_count(client: QdrantClient, name: str) -> int:
    # Scroll through all points (small fixtures only).
    points, _ = client.scroll(collection_name=name, limit=100)
    return len(points)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ingest_two_docs_end_to_end(tmp_path: Path) -> None:
    from firm.rag.ingest import run_ingest

    db_path = tmp_path / "firm.db"
    init_db(db_path)

    client = QdrantClient(":memory:")
    store = VectorStore(client)
    rag_config = _make_rag_config()
    store.create_collection(rag_config.qdrant.collection, dense_dim=rag_config.embedding.dense_dim)

    docs = [_make_doc("doc-a-001", "AAPL"), _make_doc("doc-b-002", "MSFT")]
    source = FakeSource(docs)
    augmenter = FakeAugmenter()
    embedder = FakeEmbedder()
    sparse = FakeSparse()
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))

    result = run_ingest(
        source=source,
        store=store,
        embedder=embedder,
        sparse=sparse,
        augmenter=augmenter,
        db_path=db_path,
        clock=clock,
        rag_config=rag_config,
    )

    assert result.status == "completed"
    assert result.corpus == "fake_corpus"
    assert result.docs_total == 2
    assert result.docs_completed == 2
    assert result.chunks_written > 0
    assert result.error is None
    assert augmenter.calls == 2
    assert embedder.calls == 2

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM ingest_runs ORDER BY id").fetchall()
    conn.close()
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"
    assert row["docs_total"] == 2
    assert row["docs_completed"] == 2
    assert row["chunks_written"] == result.chunks_written
    assert row["error"] is None
    assert row["finished_at"] is not None

    # Confirm chunks landed in Qdrant.
    assert _qdrant_point_count(client, rag_config.qdrant.collection) == result.chunks_written


def test_ingest_rolls_back_on_failure(tmp_path: Path) -> None:
    from firm.rag.ingest import run_ingest

    db_path = tmp_path / "firm.db"
    init_db(db_path)

    client = QdrantClient(":memory:")
    store = VectorStore(client)
    rag_config = _make_rag_config()
    store.create_collection(rag_config.qdrant.collection, dense_dim=rag_config.embedding.dense_dim)

    # Two docs; FIRST embed call raises so nothing ever reaches Qdrant.
    docs = [_make_doc("doc-fail-001", "AAPL"), _make_doc("doc-fail-002", "MSFT")]
    source = FakeSource(docs)
    augmenter = FakeAugmenter()
    embedder = FakeEmbedder(raise_on_call=1)
    sparse = FakeSparse()
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))

    result = run_ingest(
        source=source,
        store=store,
        embedder=embedder,
        sparse=sparse,
        augmenter=augmenter,
        db_path=db_path,
        clock=clock,
        rag_config=rag_config,
    )

    assert result.status == "failed"
    assert result.error is not None and "simulated embedder failure" in result.error
    assert result.docs_completed == 0
    assert result.chunks_written == 0

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM ingest_runs ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    assert row["status"] == "failed"
    assert row["chunks_written"] == 0
    assert row["docs_completed"] == 0
    assert row["error"] is not None and "simulated embedder failure" in row["error"]

    assert _qdrant_point_count(client, rag_config.qdrant.collection) == 0


def test_ingest_records_partial_progress_before_failure(tmp_path: Path) -> None:
    """When doc-1 succeeds and doc-2 raises, the ingest_runs row must show the
    partial counters from doc-1 — even though final status is 'failed' — so a
    crashed run is observable from the DB alone."""
    from firm.rag.ingest import run_ingest

    db_path = tmp_path / "firm.db"
    init_db(db_path)

    client = QdrantClient(":memory:")
    store = VectorStore(client)
    rag_config = _make_rag_config()
    store.create_collection(rag_config.qdrant.collection, dense_dim=rag_config.embedding.dense_dim)

    docs = [_make_doc("doc-partial-001", "AAPL"), _make_doc("doc-partial-002", "MSFT")]
    source = FakeSource(docs)
    augmenter = FakeAugmenter()
    # SECOND embed call raises — doc-1 fully succeeds first, doc-2 then fails.
    embedder = FakeEmbedder(raise_on_call=2)
    sparse = FakeSparse()
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))

    result = run_ingest(
        source=source,
        store=store,
        embedder=embedder,
        sparse=sparse,
        augmenter=augmenter,
        db_path=db_path,
        clock=clock,
        rag_config=rag_config,
    )

    assert result.status == "failed"
    assert result.docs_completed == 1
    assert result.chunks_written > 0
    doc1_chunk_count = result.chunks_written

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM ingest_runs ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()

    # The DB row must reflect doc-1's contribution even though the run failed.
    assert row["status"] == "failed"
    assert row["docs_completed"] == 1
    assert row["chunks_written"] == doc1_chunk_count
    assert row["chunks_written"] > 0
    assert row["error"] is not None and "simulated embedder failure" in row["error"]


def test_ingest_bumps_db_row_per_doc(tmp_path: Path) -> None:
    """Verify chunks_written/docs_completed are written to SQLite DURING the run
    (not just at the end). We peek the DB row from inside the fake augmenter's
    second call — before doc-2 has been embedded — and assert doc-1's counters
    are already visible."""
    from firm.rag.ingest import run_ingest

    db_path = tmp_path / "firm.db"
    init_db(db_path)

    client = QdrantClient(":memory:")
    store = VectorStore(client)
    rag_config = _make_rag_config()
    store.create_collection(rag_config.qdrant.collection, dense_dim=rag_config.embedding.dense_dim)

    captured: dict[str, int | None] = {"docs_completed": None, "chunks_written": None}

    class PeekingAugmenter(FakeAugmenter):
        def augment(self, doc: FilingDoc, chunks: list[Chunk]) -> list[Chunk]:
            self.calls += 1
            # On the SECOND doc, peek the DB row before this doc lands.
            if self.calls == 2:
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT docs_completed, chunks_written FROM ingest_runs "
                    "ORDER BY id DESC LIMIT 1"
                ).fetchone()
                conn.close()
                captured["docs_completed"] = int(row["docs_completed"])
                captured["chunks_written"] = int(row["chunks_written"])
            for chunk in chunks:
                chunk.doc_summary = "<summary>"
            return chunks

    docs = [_make_doc("doc-bump-001", "AAPL"), _make_doc("doc-bump-002", "MSFT")]
    source = FakeSource(docs)
    augmenter = PeekingAugmenter()
    embedder = FakeEmbedder()
    sparse = FakeSparse()
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))

    result = run_ingest(
        source=source,
        store=store,
        embedder=embedder,
        sparse=sparse,
        augmenter=augmenter,
        db_path=db_path,
        clock=clock,
        rag_config=rag_config,
    )

    assert result.status == "completed"
    assert captured["docs_completed"] == 1, (
        "DB row must show doc-1 completed before doc-2 starts"
    )
    assert captured["chunks_written"] is not None and captured["chunks_written"] > 0
    # The final value must be strictly greater than the mid-run snapshot.
    assert result.chunks_written > captured["chunks_written"]


def test_ingest_is_resumable(tmp_path: Path) -> None:
    from firm.rag.ingest import run_ingest

    db_path = tmp_path / "firm.db"
    init_db(db_path)

    client = QdrantClient(":memory:")
    store = VectorStore(client)
    rag_config = _make_rag_config()
    store.create_collection(rag_config.qdrant.collection, dense_dim=rag_config.embedding.dense_dim)

    docs = [_make_doc("doc-resume-001", "AAPL"), _make_doc("doc-resume-002", "MSFT")]
    source = FakeSource(docs)
    augmenter = FakeAugmenter()
    embedder = FakeEmbedder()
    sparse = FakeSparse()
    clock = ReplayClock(datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))

    first = run_ingest(
        source=source,
        store=store,
        embedder=embedder,
        sparse=sparse,
        augmenter=augmenter,
        db_path=db_path,
        clock=clock,
        rag_config=rag_config,
    )
    assert first.status == "completed"
    assert first.docs_completed == 2
    first_call_count = embedder.calls
    assert first_call_count == 2

    # Second run with the same docs: must skip both, perform zero new embed calls.
    second = run_ingest(
        source=FakeSource(docs),
        store=store,
        embedder=embedder,
        sparse=sparse,
        augmenter=augmenter,
        db_path=db_path,
        clock=clock,
        rag_config=rag_config,
    )

    assert second.status == "completed"
    assert second.docs_total == 2
    assert second.docs_completed == 0
    assert second.chunks_written == 0
    # Embedder must not have been called again.
    assert embedder.calls == first_call_count
