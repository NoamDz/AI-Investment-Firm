"""Permanent CI invariant: the retriever's PIT filter must exclude future chunks.

This test uses an in-memory Qdrant so it stays cheap (no LLM, no downloaded
model).  The spec (§6.4) elevates this assertion to a permanent CI invariant
because PIT discipline is the core lookahead-bias guard for the RAG layer.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("qdrant_client.local.qdrant_local")

from qdrant_client import QdrantClient  # noqa: E402

from firm.rag.chunk import Chunk  # noqa: E402
from firm.rag.qdrant_store import VectorStore  # noqa: E402
from firm.rag.retrieve import HybridRetriever  # noqa: E402


class _StaticEmbedder:
    """Deterministic dense embedder for tests: always returns the same unit vector."""

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 4), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class _StaticSparse:
    def transform(self, text: str) -> dict[int, float]:
        return {0: 1.0}


def _make_chunk(chunk_id: str, published_at: datetime, *, text: str = "body") -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=chunk_id.split("::")[0],
        ticker="AAPL",
        published_at=published_at,
        section="body",
        text=text,
        char_span=(0, len(text)),
        token_count=max(1, len(text.split())),
        source="test",
    )


def test_pit_filter_excludes_future_chunks() -> None:
    utc = timezone.utc
    client = QdrantClient(":memory:")
    store = VectorStore(client)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        store.create_collection("pit_retr", dense_dim=4)

    past = _make_chunk("past::0001", datetime(2024, 6, 1, tzinfo=utc), text="past chunk")
    boundary = _make_chunk(
        "boundary::0001", datetime(2024, 12, 31, tzinfo=utc), text="boundary chunk"
    )
    future = _make_chunk("future::0001", datetime(2025, 1, 1, tzinfo=utc), text="future chunk")

    dense_vecs: list[list[float]] = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    sparse_vecs: list[dict[int, float]] = [{0: 1.0}, {0: 1.0}, {0: 1.0}]

    store.upsert("pit_retr", [past, boundary, future], dense_vecs, sparse_vecs)

    retriever = HybridRetriever(
        store=store,
        embedder=_StaticEmbedder(),
        sparse=_StaticSparse(),
        collection="pit_retr",
        k_retrieve=50,
    )

    as_of = datetime(2024, 12, 31, tzinfo=utc)
    results = retriever.retrieve("anything", as_of=as_of)
    ids = {r.chunk.id for r in results}

    assert "past::0001" in ids, "past chunk must be returned"
    # Boundary chunk has published_at == as_of; PIT filter must be inclusive (<=).
    assert "boundary::0001" in ids, "boundary chunk (published_at == as_of) must be included"
    assert "future::0001" not in ids, "future chunk must be excluded by PIT filter"
