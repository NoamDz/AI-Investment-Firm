"""Unit tests for firm.rag.retrieve.HybridRetriever (T12).

These exercise the retriever through a stub VectorStore so they stay fast
(no Qdrant, no model loads).  The PIT invariant lives in
``tests/integration/test_retrieval_pit.py`` to keep this file pure unit.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest

from firm.rag.retrieve import HybridRetriever, RetrievedChunk

_RRF_K = 60


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _payload(
    chunk_id: str,
    *,
    text: str = "body text",
    doc_summary: str | None = None,
    published_at: datetime | None = None,
) -> dict[str, Any]:
    when = published_at or datetime(2024, 6, 1, tzinfo=timezone.utc)
    return {
        "chunk_id": chunk_id,
        "doc_id": chunk_id.split("::")[0],
        "ticker": "AAPL",
        "section": "body",
        "published_at": when.timestamp(),
        "text": text,
        "doc_summary": doc_summary,
        "char_span": [0, len(text)],
        "token_count": max(1, len(text.split())),
    }


@dataclass
class FakeVectorStore:
    """Stub for VectorStore: returns canned hit lists from search_dense/search_sparse."""

    dense_hits: list[dict[str, Any]] = field(default_factory=list)
    sparse_hits: list[dict[str, Any]] = field(default_factory=list)
    last_dense_k: int | None = None
    last_sparse_k: int | None = None
    last_published_before: datetime | None = None

    def search_dense(
        self,
        name: str,
        dense_vec: Sequence[float],
        k: int,
        *,
        published_before: datetime,
    ) -> list[dict[str, Any]]:
        self.last_dense_k = k
        self.last_published_before = published_before
        return self.dense_hits[:k]

    def search_sparse(
        self,
        name: str,
        sparse_vec: dict[int, float],
        k: int,
        *,
        published_before: datetime,
    ) -> list[dict[str, Any]]:
        self.last_sparse_k = k
        self.last_published_before = published_before
        return self.sparse_hits[:k]


class FakeEmbedder:
    """Deterministic dense embedder returning a (1, 4) array."""

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


class FakeSparseEncoder:
    def transform(self, text: str) -> dict[int, float]:
        return {0: 1.0}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_hybrid_retrieve_returns_top_50_max() -> None:
    """100 dense + 100 sparse with overlap -> retriever returns <= 50 hits."""
    dense = [
        {"chunk_id": f"d::{i:04d}", "score": 1.0 - i * 0.001, "payload": _payload(f"d::{i:04d}")}
        for i in range(100)
    ]
    sparse = [
        {"chunk_id": f"s::{i:04d}", "score": 1.0 - i * 0.001, "payload": _payload(f"s::{i:04d}")}
        for i in range(100)
    ]
    store = FakeVectorStore(dense_hits=dense, sparse_hits=sparse)

    retriever = HybridRetriever(
        store=store,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),
        sparse=FakeSparseEncoder(),
        collection="test",
        k_retrieve=50,
    )

    results = retriever.retrieve(
        "query", as_of=datetime(2024, 12, 31, tzinfo=timezone.utc)
    )

    assert len(results) <= 50
    assert all(isinstance(r, RetrievedChunk) for r in results)
    # Retriever should have asked the store for k_retrieve hits per list.
    assert store.last_dense_k == 50
    assert store.last_sparse_k == 50


def test_dense_and_sparse_results_merged_by_rrf() -> None:
    """Construct hand-crafted dense and sparse lists; verify RRF ranking and rank metadata."""
    # chunk_id "A": rank 0 dense, rank 2 sparse
    # chunk_id "B": rank 1 dense, rank 0 sparse
    # chunk_id "C": rank 2 dense, only          -> rank_sparse None
    # chunk_id "D": only sparse rank 1         -> rank_dense None
    dense = [
        {"chunk_id": "A", "score": 0.9, "payload": _payload("A")},
        {"chunk_id": "B", "score": 0.8, "payload": _payload("B")},
        {"chunk_id": "C", "score": 0.7, "payload": _payload("C")},
    ]
    sparse = [
        {"chunk_id": "B", "score": 0.5, "payload": _payload("B")},
        {"chunk_id": "D", "score": 0.4, "payload": _payload("D")},
        {"chunk_id": "A", "score": 0.3, "payload": _payload("A")},
    ]
    store = FakeVectorStore(dense_hits=dense, sparse_hits=sparse)

    retriever = HybridRetriever(
        store=store,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),
        sparse=FakeSparseEncoder(),
        collection="test",
        k_retrieve=10,
    )

    results = retriever.retrieve("q", as_of=datetime(2024, 12, 31, tzinfo=timezone.utc))

    # Manually-computed RRF scores (rank is 0-indexed; +1 in denominator).
    def rrf(rank: int) -> float:
        return 1.0 / (_RRF_K + rank + 1)

    expected = {
        "A": rrf(0) + rrf(2),
        "B": rrf(1) + rrf(0),
        "C": rrf(2),
        "D": rrf(1),
    }
    expected_order = [cid for cid, _ in sorted(expected.items(), key=lambda kv: kv[1], reverse=True)]
    actual_order = [r.chunk.id for r in results]

    assert actual_order == expected_order, (
        f"RRF ordering mismatch: expected {expected_order}, got {actual_order}"
    )

    by_id: dict[str, RetrievedChunk] = {r.chunk.id: r for r in results}
    assert by_id["A"].rank_dense == 0
    assert by_id["A"].rank_sparse == 2
    assert by_id["B"].rank_dense == 1
    assert by_id["B"].rank_sparse == 0
    assert by_id["C"].rank_dense == 2
    assert by_id["C"].rank_sparse is None
    assert by_id["D"].rank_dense is None
    assert by_id["D"].rank_sparse == 1

    # Score field should match the RRF score.
    assert by_id["A"].score == pytest.approx(expected["A"])
    assert by_id["B"].score == pytest.approx(expected["B"])


def test_retrieval_returns_chunks_with_doc_summary_prefix_attached() -> None:
    """Chunks with non-empty doc_summary must have it prepended to chunk.text on return.

    Chunks without doc_summary (None or empty) must come back with text unchanged.
    """
    dense = [
        {
            "chunk_id": "with-summary",
            "score": 0.9,
            "payload": _payload("with-summary", text="<ORIGINAL>", doc_summary="<SUMMARY>"),
        },
        {
            "chunk_id": "no-summary",
            "score": 0.8,
            "payload": _payload("no-summary", text="<PLAIN>", doc_summary=None),
        },
        {
            "chunk_id": "empty-summary",
            "score": 0.7,
            "payload": _payload("empty-summary", text="<PLAIN2>", doc_summary=""),
        },
    ]
    store = FakeVectorStore(dense_hits=dense, sparse_hits=[])

    retriever = HybridRetriever(
        store=store,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),
        sparse=FakeSparseEncoder(),
        collection="test",
        k_retrieve=10,
    )

    results = retriever.retrieve("q", as_of=datetime(2024, 12, 31, tzinfo=timezone.utc))
    by_id = {r.chunk.id: r for r in results}

    assert by_id["with-summary"].chunk.text == "<SUMMARY>\n\n<ORIGINAL>"
    assert by_id["no-summary"].chunk.text == "<PLAIN>"
    assert by_id["empty-summary"].chunk.text == "<PLAIN2>"

    # doc_summary on the reconstructed Chunk should reflect the payload value
    # (the retriever mutates text, not the stored doc_summary field).
    assert by_id["with-summary"].chunk.doc_summary == "<SUMMARY>"
    assert by_id["no-summary"].chunk.doc_summary is None


def test_retrieve_requires_timezone_aware_as_of() -> None:
    """as_of without tzinfo must raise ValueError — PIT semantics require a UTC anchor."""
    store = FakeVectorStore(dense_hits=[], sparse_hits=[])
    retriever = HybridRetriever(
        store=store,  # type: ignore[arg-type]
        embedder=FakeEmbedder(),
        sparse=FakeSparseEncoder(),
        collection="test",
    )

    with pytest.raises(ValueError, match="timezone-aware"):
        retriever.retrieve("q", as_of=datetime(2024, 12, 31))  # naive
