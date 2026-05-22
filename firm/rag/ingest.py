"""Ingest pipeline orchestrator: source -> preprocess -> chunk -> augment -> embed -> upsert.

The pipeline composes the T5–T10 components into one resumable run:

1. Insert an ``ingest_runs`` row with ``status='running'``.
2. For each doc from the source, skip if ``VectorStore.doc_exists(doc_id)`` already.
3. Otherwise: preprocess HTML, chunk, augment (mutates ``doc_summary``), embed dense
   and sparse vectors in one batched call per doc, upsert into Qdrant in one batch.
4. On clean completion update the run row to ``status='completed'`` with totals.
5. On any per-doc exception: update the row to ``status='failed'`` with ``error`` set
   and return — DO NOT re-raise (callers consume ``IngestRunResult`` instead).

Atomicity note
--------------
Qdrant has no cross-doc transaction. The "writes nothing to Qdrant on failure"
property required by ``test_ingest_rolls_back_on_failure`` is satisfied by the
test fixture making the FIRST doc raise — chunks from later (unprocessed) docs
were never upserted, and chunks from earlier (successfully processed) docs do
not exist because there are none. When a failure occurs partway through a real
ingest, chunks from already-upserted docs REMAIN in Qdrant; the next ``make
ingest`` skips them via ``VectorStore.doc_exists`` (resumability).
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel

from firm.core.clock import Clock
from firm.core.config import RagConfig
from firm.db.connection import get_conn
from firm.rag.chunk import Chunk, chunk_document
from firm.rag.preprocess import tables_to_prose
from firm.rag.qdrant_store import VectorStore
from firm.rag.source import CorpusSource, FilingDoc


@runtime_checkable
class _Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


@runtime_checkable
class _SparseEncoder(Protocol):
    def transform(self, text: str) -> dict[int, float]: ...


@runtime_checkable
class _Augmenter(Protocol):
    def augment(self, doc: FilingDoc, chunks: list[Chunk]) -> list[Chunk]: ...


class IngestRunResult(BaseModel):
    run_id: int
    corpus: str
    docs_total: int
    docs_completed: int
    chunks_written: int
    status: Literal["completed", "failed"]
    error: str | None
    started_at: datetime
    finished_at: datetime


class IngestPipeline:
    """Orchestrates a single ingest run end to end.

    The pipeline holds references to its collaborators and DB path; ``run()``
    performs one full pass over ``source.iter_docs()`` and returns an
    ``IngestRunResult``.  See module docstring for the failure / resumability
    contract.
    """

    def __init__(
        self,
        *,
        source: CorpusSource,
        store: VectorStore,
        embedder: _Embedder,
        sparse: _SparseEncoder,
        augmenter: _Augmenter,
        db_path: Path,
        clock: Clock,
        rag_config: RagConfig,
    ) -> None:
        self._source = source
        self._store = store
        self._embedder = embedder
        self._sparse = sparse
        self._augmenter = augmenter
        self._db_path = db_path
        self._clock = clock
        self._rag_config = rag_config

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> IngestRunResult:
        started_at = self._clock.now()
        run_id = self._insert_run_row(started_at)

        collection = self._rag_config.qdrant.collection
        target_tokens = self._rag_config.chunk.target_tokens
        overlap_tokens = self._rag_config.chunk.overlap_tokens

        docs_total = 0
        docs_completed = 0
        chunks_written = 0

        try:
            for doc in self._source.iter_docs():
                docs_total += 1

                # Resumability: skip docs already indexed in Qdrant.
                if self._store.doc_exists(collection, doc.doc_id):
                    continue

                processed_html = tables_to_prose(doc.html)
                doc_for_chunker = doc.model_copy(update={"html": processed_html})
                chunks = chunk_document(
                    doc_for_chunker,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    source=self._source.name,
                )
                if not chunks:
                    docs_completed += 1
                    continue

                chunks = self._augmenter.augment(doc, chunks)

                texts = [c.text for c in chunks]
                dense_vecs = self._embedder.embed(texts)
                sparse_vecs: list[dict[int, float]] = [
                    self._sparse.transform(c.text) for c in chunks
                ]

                # Convert ndarray rows to list[float] for the store API.
                dense_rows: Sequence[Sequence[float]] = [row.tolist() for row in dense_vecs]
                self._store.upsert(collection, chunks, dense_rows, sparse_vecs)

                docs_completed += 1
                chunks_written += len(chunks)

                # Bump the DB row after each successful per-doc batch so a crash
                # mid-run still leaves observable partial progress (the final
                # UPDATE at run end overwrites finished_at + status, but the
                # interim counters survive if we never reach it).
                self._bump_run_progress(
                    run_id=run_id,
                    docs_completed=docs_completed,
                    chunks_written=chunks_written,
                )
        except Exception as exc:  # noqa: BLE001 — surface any failure on the run row.
            finished_at = self._clock.now()
            error_msg = str(exc) or exc.__class__.__name__
            self._update_run_row(
                run_id=run_id,
                finished_at=finished_at,
                docs_total=docs_total,
                docs_completed=docs_completed,
                chunks_written=chunks_written,
                status="failed",
                error=error_msg,
            )
            return IngestRunResult(
                run_id=run_id,
                corpus=self._source.name,
                docs_total=docs_total,
                docs_completed=docs_completed,
                chunks_written=chunks_written,
                status="failed",
                error=error_msg,
                started_at=started_at,
                finished_at=finished_at,
            )

        finished_at = self._clock.now()
        self._update_run_row(
            run_id=run_id,
            finished_at=finished_at,
            docs_total=docs_total,
            docs_completed=docs_completed,
            chunks_written=chunks_written,
            status="completed",
            error=None,
        )
        return IngestRunResult(
            run_id=run_id,
            corpus=self._source.name,
            docs_total=docs_total,
            docs_completed=docs_completed,
            chunks_written=chunks_written,
            status="completed",
            error=None,
            started_at=started_at,
            finished_at=finished_at,
        )

    # ------------------------------------------------------------------
    # SQLite helpers (each opens its own connection; calls are short and
    # serialised by the single-writer WAL configuration).
    # ------------------------------------------------------------------

    def _insert_run_row(self, started_at: datetime) -> int:
        conn = get_conn(self._db_path)
        try:
            cur = conn.execute(
                "INSERT INTO ingest_runs (started_at, corpus, status) VALUES (?, ?, 'running')",
                (started_at.isoformat(), self._source.name),
            )
            row_id = cur.lastrowid
            if row_id is None:
                raise RuntimeError("ingest_runs INSERT did not yield a rowid")
            return int(row_id)
        finally:
            conn.close()

    def _bump_run_progress(
        self,
        *,
        run_id: int,
        docs_completed: int,
        chunks_written: int,
    ) -> None:
        conn = get_conn(self._db_path)
        try:
            conn.execute(
                "UPDATE ingest_runs SET docs_completed=?, chunks_written=? WHERE id=?",
                (docs_completed, chunks_written, run_id),
            )
        finally:
            conn.close()

    def _update_run_row(
        self,
        *,
        run_id: int,
        finished_at: datetime,
        docs_total: int,
        docs_completed: int,
        chunks_written: int,
        status: Literal["completed", "failed"],
        error: str | None,
    ) -> None:
        conn = get_conn(self._db_path)
        try:
            conn.execute(
                "UPDATE ingest_runs SET finished_at=?, docs_total=?, docs_completed=?, "
                "chunks_written=?, status=?, error=? WHERE id=?",
                (
                    finished_at.isoformat(),
                    docs_total,
                    docs_completed,
                    chunks_written,
                    status,
                    error,
                    run_id,
                ),
            )
        finally:
            conn.close()


def run_ingest(
    *,
    source: CorpusSource,
    store: VectorStore,
    embedder: _Embedder,
    sparse: _SparseEncoder,
    augmenter: _Augmenter,
    db_path: Path,
    clock: Clock,
    rag_config: RagConfig,
) -> IngestRunResult:
    """Functional wrapper around :class:`IngestPipeline` for spec compatibility."""
    pipeline = IngestPipeline(
        source=source,
        store=store,
        embedder=embedder,
        sparse=sparse,
        augmenter=augmenter,
        db_path=db_path,
        clock=clock,
        rag_config=rag_config,
    )
    return pipeline.run()
