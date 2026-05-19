"""Integration test for the grounded research agent (T19).

Wires a real ``GroundedRetriever`` (in-memory Qdrant + a stub reranker) to a
stub ``CitedClaimExtractor`` that returns one pre-baked Claim. Asserts the
research agent produces a Decision with at least one Citation and surfaces
the retrieved chunks on state.

The reranker is stubbed (per T13's pattern) to keep the test network-free —
loading the real ``bge-reranker-v2-m3`` would download a model.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("qdrant_client.local.qdrant_local")

from qdrant_client import QdrantClient  # noqa: E402

from firm.agents.research import make_research  # noqa: E402
from firm.broker.fake_broker import FakeBroker  # noqa: E402
from firm.core.clock import ReplayClock  # noqa: E402
from firm.core.config import load_universe  # noqa: E402
from firm.core.models import Claim  # noqa: E402
from firm.rag.chunk import Chunk  # noqa: E402
from firm.rag.qdrant_store import VectorStore  # noqa: E402
from firm.rag.rerank import BgeReranker  # noqa: E402
from firm.rag.retrieve import GroundedRetriever, HybridRetriever  # noqa: E402


class _StaticEmbedder:
    """Deterministic dense embedder: emits a constant unit vector."""

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 4), dtype=np.float32)
        out[:, 0] = 1.0
        return out


class _StaticSparse:
    def transform(self, text: str) -> dict[int, float]:
        return {0: 1.0}


class _PassThroughCrossEncoder:
    """CrossEncoder stub that returns a constant high score for every pair."""

    def predict(self, pairs: list[list[str]]) -> list[float]:
        return [0.9 for _ in pairs]


class _OneClaimExtractor:
    """Stub extractor returning a single grounded Claim against the first chunk."""

    def extract(
        self, *, query: str, chunks: list[Chunk], as_of: datetime
    ) -> list[Claim]:
        if not chunks:
            return []
        chunk = chunks[0]
        return [
            Claim(
                text="Apple reported strong revenue growth in the most recent quarter.",
                source_chunk_id=chunk.id,
                source_span=(0, min(50, len(chunk.text))),
            )
        ]


def test_research_end_to_end_produces_decision_with_citation() -> None:
    utc = timezone.utc
    client = QdrantClient(":memory:")
    store = VectorStore(client)
    collection = "research_e2e"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        store.create_collection(collection, dense_dim=4)

    # Seed two chunks: both before the as_of, so PIT passes both through.
    chunk_a = Chunk(
        id="doc-aapl-001::0001",
        doc_id="doc-aapl-001",
        ticker="AAPL",
        section="body",
        published_at=datetime(2024, 6, 1, tzinfo=utc),
        text="Apple revenue grew 8% year-over-year driven by services and wearables.",
        char_span=(0, 70),
        token_count=12,
    )
    chunk_b = Chunk(
        id="doc-aapl-002::0001",
        doc_id="doc-aapl-002",
        ticker="AAPL",
        section="body",
        published_at=datetime(2024, 5, 1, tzinfo=utc),
        text="Management reiterated full-year guidance and capital return plans.",
        char_span=(0, 66),
        token_count=10,
    )
    dense_vecs: list[list[float]] = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    sparse_vecs: list[dict[int, float]] = [{0: 1.0}, {0: 1.0}]
    store.upsert(collection, [chunk_a, chunk_b], dense_vecs, sparse_vecs)

    hybrid = HybridRetriever(
        store=store,
        embedder=_StaticEmbedder(),
        sparse=_StaticSparse(),
        collection=collection,
        k_retrieve=8,
    )
    reranker = BgeReranker(
        model_id="stub",
        score_floor=0.0,
        model=_PassThroughCrossEncoder(),
    )
    retriever: GroundedRetriever = GroundedRetriever(
        hybrid=hybrid, reranker=reranker, k_final=4
    )
    extractor = _OneClaimExtractor()

    broker = FakeBroker(initial_cash=Decimal("100000"))
    universe = load_universe(Path("config/universe.yaml"))
    clock = ReplayClock(datetime(2024, 9, 15, tzinfo=utc))

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=retriever,
        extractor=extractor,
    )
    out = research({"heartbeat_at": clock.now().isoformat()})

    decision = out["research_decision"]
    assert len(decision.citations) >= 1
    # The cited chunk must correspond to one of the seeded chunks.
    cited_chunk_ids = {c.chunk_id for c in decision.citations}
    assert cited_chunk_ids.issubset({chunk_a.id, chunk_b.id})
    # The retrieved_chunks slot should carry the underlying Chunks (not RetrievedChunks).
    retrieved: list[Chunk] = out["retrieved_chunks"]
    assert len(retrieved) >= 1
    assert all(isinstance(c, Chunk) for c in retrieved)
    # oldest_filing_age_days should be present and equal to the days from
    # the *oldest* seeded chunk's published_at to clock.now() (chunk_b is older).
    expected_age = (clock.now().date() - chunk_b.published_at.date()).days
    assert decision.metadata["oldest_filing_age_days"] == expected_age
