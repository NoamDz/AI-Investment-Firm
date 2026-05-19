"""Hybrid dense+sparse retriever with Point-In-Time filtering (T12).

Design decisions
----------------
* RRF (k=60) is computed locally over the per-list rankings returned by
  ``VectorStore.search_dense`` and ``VectorStore.search_sparse`` so that each
  ``RetrievedChunk`` can carry its independent ``rank_dense`` / ``rank_sparse``
  positions.  ``VectorStore.search_hybrid`` still exists for callers that do
  not need rank metadata; the retriever owns its own fusion because it needs
  the per-list ranks.

* The PIT filter is delegated to ``VectorStore.search_*`` via
  ``published_before=as_of``.  The retriever validates that ``as_of`` is
  timezone-aware (raises ``ValueError`` otherwise) — the PIT contract requires
  an unambiguous UTC anchor (see CI invariant in
  ``tests/integration/test_retrieval_pit.py``).

* ``doc_summary`` prefix attachment: Anthropic contextual retrieval pattern —
  prepend the doc-level summary to chunk text at retrieve time for downstream
  LLM consumption.  The stored payload ``text`` stays clean; the prefix is
  applied to the reconstructed Chunk only when ``doc_summary`` is non-empty.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from firm.rag.chunk import Chunk
from firm.rag.qdrant_store import VectorStore

if TYPE_CHECKING:
    from firm.rag.rerank import BgeReranker

_RRF_K = 60


# ---------------------------------------------------------------------------
# Injected-collaborator protocols (mypy --strict friendly)
# ---------------------------------------------------------------------------


@runtime_checkable
class DenseEmbedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class SparseEncoder(Protocol):
    def transform(self, text: str) -> dict[int, float]: ...


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rank_dense: int | None
    rank_sparse: int | None
    rerank_score: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _payload_to_chunk(payload: dict[str, Any]) -> Chunk:
    """Reconstruct a :class:`Chunk` from a Qdrant payload.

    Inverse of ``VectorStore.upsert`` payload construction.  ``char_span`` is
    round-tripped through ``list[int]`` because JSON has no tuple type.
    """
    raw_span = payload["char_span"]
    char_span: tuple[int, int] = (int(raw_span[0]), int(raw_span[1]))
    published_at = datetime.fromtimestamp(float(payload["published_at"]), tz=timezone.utc)
    return Chunk(
        id=str(payload["chunk_id"]),
        doc_id=str(payload["doc_id"]),
        ticker=str(payload["ticker"]),
        section=str(payload["section"]),
        published_at=published_at,
        text=str(payload["text"]),
        char_span=char_span,
        token_count=int(payload["token_count"]),
        doc_summary=payload.get("doc_summary"),
        metadata={},
    )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Dense + sparse retrieval with RRF fusion and PIT filtering.

    Parameters
    ----------
    store
        ``VectorStore`` already bound to its Qdrant client.
    embedder
        Dense-vector producer; called with ``[query]`` once per retrieve.
    sparse
        Sparse-vector producer; called with ``query`` once per retrieve.
    collection
        Qdrant collection name.
    k_retrieve
        Number of candidates pulled from each sub-search and (also) the upper
        bound on returned ``RetrievedChunk`` count.  Default 50 (spec §5.4).
    """

    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: DenseEmbedder,
        sparse: SparseEncoder,
        collection: str,
        k_retrieve: int = 50,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._sparse = sparse
        self._collection = collection
        self._k_retrieve = k_retrieve

    def retrieve(self, query: str, *, as_of: datetime) -> list[RetrievedChunk]:
        if as_of.tzinfo is None:
            raise ValueError(
                "as_of must be timezone-aware (PIT contract requires a UTC anchor)"
            )

        dense_arr = self._embedder.embed([query])
        # ``.tolist()`` collapses a 1-row ndarray to ``list[float]``; we guard
        # against an empty result just in case a stub returns shape (0, dim).
        dense_vec: Sequence[float] = (
            dense_arr[0].tolist() if dense_arr.shape[0] > 0 else []
        )
        sparse_vec: dict[int, float] = self._sparse.transform(query)

        dense_hits = self._store.search_dense(
            self._collection, dense_vec, self._k_retrieve, published_before=as_of
        )
        sparse_hits = self._store.search_sparse(
            self._collection, sparse_vec, self._k_retrieve, published_before=as_of
        )

        # Per-chunk-id: (rank_dense, rank_sparse, payload, rrf_score).
        rank_dense: dict[str, int] = {}
        rank_sparse: dict[str, int] = {}
        payload_by_id: dict[str, dict[str, Any]] = {}
        rrf_score: dict[str, float] = {}

        for rank, hit in enumerate(dense_hits):
            cid = str(hit["chunk_id"])
            rank_dense[cid] = rank
            payload_by_id.setdefault(cid, hit["payload"])
            rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)

        for rank, hit in enumerate(sparse_hits):
            cid = str(hit["chunk_id"])
            rank_sparse[cid] = rank
            payload_by_id.setdefault(cid, hit["payload"])
            rrf_score[cid] = rrf_score.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)

        # Sort by combined RRF score desc, truncate to k_retrieve.
        ranked = sorted(rrf_score.items(), key=lambda kv: kv[1], reverse=True)[
            : self._k_retrieve
        ]

        results: list[RetrievedChunk] = []
        for cid, score in ranked:
            payload = payload_by_id[cid]
            chunk = _payload_to_chunk(payload)

            # Anthropic contextual retrieval pattern: prepend doc-level summary
            # to chunk text at retrieve time for downstream LLM consumption.
            # The stored payload text stays clean.
            if chunk.doc_summary:
                chunk = chunk.model_copy(
                    update={"text": f"{chunk.doc_summary}\n\n{chunk.text}"}
                )

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    rank_dense=rank_dense.get(cid),
                    rank_sparse=rank_sparse.get(cid),
                )
            )

        return results


# ---------------------------------------------------------------------------
# GroundedRetriever facade (T14)
# ---------------------------------------------------------------------------


class GroundedRetriever:
    """Thin facade composing :class:`HybridRetriever` with a cross-encoder rerank step.

    The retrieve-then-rerank pipeline is wrapped in a single ``retrieve`` call so
    downstream agents do not need to know about the two-stage shape.  All filtering,
    PIT semantics, and contextual-summary attachment live in the underlying
    collaborators — this class is intentionally trivial.
    """

    def __init__(
        self,
        *,
        hybrid: HybridRetriever,
        reranker: BgeReranker,
        k_final: int = 8,
    ) -> None:
        self._hybrid = hybrid
        self._reranker = reranker
        self._k_final = k_final

    def retrieve(self, query: str, *, as_of: datetime) -> list[RetrievedChunk]:
        cands = self._hybrid.retrieve(query, as_of=as_of)
        return self._reranker.rerank(query, cands, k=self._k_final)
