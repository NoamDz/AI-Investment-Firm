"""Qdrant wrapper for hybrid dense+sparse retrieval with Point-In-Time (PIT) filtering.

Design decisions
----------------
* ``published_at`` is stored as a Unix timestamp (float) in the payload so that
  Qdrant's ``Range(lte=...)`` filter operates on a plain numeric field without
  any serialization ambiguity.

* Point IDs are derived via ``uuid.uuid5(uuid.NAMESPACE_URL, chunk.id)`` for
  stable, collision-resistant mapping from string chunk IDs to Qdrant-native UUIDs.
  The original ``chunk.id`` is also stored in the payload under ``"chunk_id"`` so
  callers never depend on UUID decoding.

* Hybrid search uses Reciprocal Rank Fusion (RRF) with the conventional constant
  k=60 (as referenced in the Plan 2 locked decisions for T10).  For each candidate
  list the score for a result at rank r (0-indexed) is ``1 / (60 + r + 1)``.
  Final scores are the sum of per-list RRF scores; the top-k by combined score are
  returned.  Oversample factor 4× (each sub-search fetches ``k*4`` results) gives
  good coverage before fusion.
"""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from firm.rag.chunk import Chunk

_RRF_K = 60


class VectorStore:
    """Qdrant wrapper for hybrid dense+sparse retrieval with PIT filtering."""

    DENSE_NAME = "dense"
    SPARSE_NAME = "sparse"

    def __init__(self, client: QdrantClient) -> None:
        self._client = client

    def create_collection(self, name: str, dense_dim: int, *, force: bool = False) -> None:
        """Create the named hybrid collection if it does not already exist.

        Idempotent by default — calling this on an existing collection is a no-op,
        which is what the ingest pipeline relies on for resumability (the per-doc
        ``doc_exists`` skip is meaningless if the collection itself gets wiped).
        Pass ``force=True`` to drop and recreate (test fixtures use this).
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if name in existing:
            if not force:
                return
            self._client.delete_collection(collection_name=name)
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                self.DENSE_NAME: models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                self.SPARSE_NAME: models.SparseVectorParams(),
            },
        )

    def upsert(
        self,
        name: str,
        chunks: Sequence[Chunk],
        dense_vecs: Sequence[Sequence[float]],
        sparse_vecs: Sequence[dict[int, float]],
    ) -> None:
        if not (len(chunks) == len(dense_vecs) == len(sparse_vecs)):
            raise ValueError("chunks/dense_vecs/sparse_vecs length mismatch")
        if not chunks:
            return
        points: list[models.PointStruct] = []
        for chunk, dvec, svec in zip(chunks, dense_vecs, sparse_vecs, strict=True):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.id))
            indices = list(svec.keys())
            values = list(svec.values())
            payload: dict[str, Any] = {
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "ticker": chunk.ticker,
                "section": chunk.section,
                "source": chunk.source,        # T21
                "published_at": chunk.published_at.timestamp(),
                "text": chunk.text,
                "doc_summary": chunk.doc_summary,
                # T12 reconstruction: Chunk requires char_span + token_count.
                # JSON has no tuple type, so we round-trip char_span as list[int].
                "char_span": [chunk.char_span[0], chunk.char_span[1]],
                "token_count": chunk.token_count,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        self.DENSE_NAME: list(dvec),
                        self.SPARSE_NAME: models.SparseVector(
                            indices=indices,
                            values=values,
                        ),
                    },
                    payload=payload,
                )
            )
        self._client.upsert(collection_name=name, points=points)

    def doc_exists(self, name: str, doc_id: str) -> bool:
        """Return True iff any point in *name* has payload ``doc_id == doc_id``.

        Used by the ingest pipeline (T11) to skip docs that are already indexed,
        enabling a second ``make ingest`` to resume without re-embedding.
        """
        points, _ = self._client.scroll(
            collection_name=name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(points) > 0

    def _pit_filter(self, published_before: datetime) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="published_at",
                    range=models.Range(lte=published_before.timestamp()),
                )
            ]
        )

    def search_dense(
        self,
        name: str,
        dense_vec: Sequence[float],
        k: int,
        *,
        published_before: datetime,
    ) -> list[dict[str, Any]]:
        response = self._client.query_points(
            collection_name=name,
            query=list(dense_vec),
            using=self.DENSE_NAME,
            limit=k,
            query_filter=self._pit_filter(published_before),
        )
        return [
            {
                "chunk_id": p.payload["chunk_id"] if p.payload else "",
                "score": p.score,
                "payload": p.payload or {},
            }
            for p in response.points
        ]

    def search_sparse(
        self,
        name: str,
        sparse_vec: dict[int, float],
        k: int,
        *,
        published_before: datetime,
    ) -> list[dict[str, Any]]:
        indices = list(sparse_vec.keys())
        values = list(sparse_vec.values())
        response = self._client.query_points(
            collection_name=name,
            query=models.SparseVector(indices=indices, values=values),
            using=self.SPARSE_NAME,
            limit=k,
            query_filter=self._pit_filter(published_before),
        )
        return [
            {
                "chunk_id": p.payload["chunk_id"] if p.payload else "",
                "score": p.score,
                "payload": p.payload or {},
            }
            for p in response.points
        ]

    def search_hybrid(
        self,
        name: str,
        dense_vec: Sequence[float],
        sparse_vec: dict[int, float],
        k: int,
        *,
        published_before: datetime,
    ) -> list[dict[str, Any]]:
        oversample = k * 4
        dense_results = self.search_dense(
            name, dense_vec, oversample, published_before=published_before
        )
        sparse_results = self.search_sparse(
            name, sparse_vec, oversample, published_before=published_before
        )

        rrf_scores: dict[str, float] = {}

        for rank, hit in enumerate(dense_results):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)

        for rank, hit in enumerate(sparse_results):
            cid = hit["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)

        all_hits: dict[str, dict[str, Any]] = {}
        for hit in dense_results + sparse_results:
            cid = hit["chunk_id"]
            if cid not in all_hits:
                all_hits[cid] = hit

        ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [
            {
                "chunk_id": cid,
                "score": score,
                "payload": all_hits[cid]["payload"],
            }
            for cid, score in ranked
        ]
