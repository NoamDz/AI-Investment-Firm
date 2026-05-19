"""Tests for firm.rag.qdrant_store — T10 spec compliance."""
from __future__ import annotations

import warnings
from datetime import datetime, timezone

import pytest

pytest.importorskip("qdrant_client.local.qdrant_local")

from qdrant_client import QdrantClient  # noqa: E402

from firm.rag.chunk import Chunk  # noqa: E402


def _make_client() -> QdrantClient:
    return QdrantClient(":memory:")


def _make_chunk(chunk_id: str, ticker: str, published_at: datetime) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id="test-doc",
        ticker=ticker,
        published_at=published_at,
        section="body",
        text=f"Text for {chunk_id}",
        char_span=(0, 20),
        token_count=5,
    )


# ---------------------------------------------------------------------------
# T10-test-1
# ---------------------------------------------------------------------------


def test_create_collection_with_named_vectors() -> None:
    from firm.rag.qdrant_store import VectorStore

    client = _make_client()
    store = VectorStore(client)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        store.create_collection("test_col", dense_dim=768)

    info = client.get_collection("test_col")
    vectors = info.config.params.vectors
    sparse_vectors = info.config.params.sparse_vectors

    assert isinstance(vectors, dict), "vectors_config must be a named-vector dict"
    assert "dense" in vectors, "dense vector not found"
    assert vectors["dense"].size == 768
    assert sparse_vectors is not None, "sparse_vectors must be configured"
    assert "sparse" in sparse_vectors, "sparse vector not found"


# ---------------------------------------------------------------------------
# T10-test-2
# ---------------------------------------------------------------------------


def test_upsert_then_search_returns_chunk_id() -> None:
    from firm.rag.qdrant_store import VectorStore

    client = _make_client()
    store = VectorStore(client)
    utc = timezone.utc
    published = datetime(2023, 6, 1, tzinfo=utc)

    chunks = [
        _make_chunk("aapl-10k::0001", "AAPL", published),
        _make_chunk("aapl-10k::0002", "AAPL", published),
        _make_chunk("aapl-10k::0003", "AAPL", published),
    ]

    dim = 4
    dense_vecs: list[list[float]] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    sparse_vecs: list[dict[int, float]] = [
        {0: 1.0},
        {1: 1.0},
        {2: 1.0},
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        store.create_collection("chunks", dense_dim=dim)
        store.upsert("chunks", chunks, dense_vecs, sparse_vecs)

    results = store.search_dense(
        "chunks",
        [1.0, 0.0, 0.0, 0.0],
        k=2,
        published_before=datetime(2024, 1, 1, tzinfo=utc),
    )

    assert len(results) > 0, "search returned no results"
    # Query [1,0,0,0] is identical to chunk-1's dense vec — it must rank #1 by cosine.
    assert results[0]["chunk_id"] == "aapl-10k::0001", (
        f"expected aapl-10k::0001 as top hit by cosine similarity; got {results[0]['chunk_id']}"
    )


# ---------------------------------------------------------------------------
# T10-test-3
# ---------------------------------------------------------------------------


def test_payload_filter_published_at_excludes_future() -> None:
    from firm.rag.qdrant_store import VectorStore

    client = _make_client()
    store = VectorStore(client)
    utc = timezone.utc

    past_chunk = _make_chunk("doc::0001", "AAPL", datetime(2023, 1, 1, tzinfo=utc))
    future_chunk = _make_chunk("doc::0002", "AAPL", datetime(2025, 1, 1, tzinfo=utc))

    dense_vecs: list[list[float]] = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ]
    sparse_vecs: list[dict[int, float]] = [
        {0: 1.0},
        {0: 1.0},
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        store.create_collection("pit_test", dense_dim=4)
        store.upsert("pit_test", [past_chunk, future_chunk], dense_vecs, sparse_vecs)

    cutoff = datetime(2024, 1, 1, tzinfo=utc)
    results = store.search_dense("pit_test", [1.0, 0.0, 0.0, 0.0], k=10, published_before=cutoff)

    returned_ids = {r["chunk_id"] for r in results}
    assert "doc::0001" in returned_ids, "past chunk should be returned"
    assert "doc::0002" not in returned_ids, "future chunk must be excluded by PIT filter"


# ---------------------------------------------------------------------------
# Length-mismatch guard
# ---------------------------------------------------------------------------


def test_upsert_rejects_length_mismatch() -> None:
    from firm.rag.qdrant_store import VectorStore

    client = _make_client()
    store = VectorStore(client)
    utc = timezone.utc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        store.create_collection("mismatch", dense_dim=4)

    chunks = [_make_chunk("doc::0001", "AAPL", datetime(2023, 1, 1, tzinfo=utc))]
    dense_vecs: list[list[float]] = []  # length 0, not 1
    sparse_vecs: list[dict[int, float]] = [{}]

    with pytest.raises(ValueError, match="length mismatch"):
        store.upsert("mismatch", chunks, dense_vecs, sparse_vecs)
