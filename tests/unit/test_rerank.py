"""Unit tests for firm.rag.rerank.BgeReranker (T13).

The reranker is exercised through a stub CrossEncoder that scores by lexical
overlap, so these tests are network-free and model-free.  A single smoke test
against the real bge-reranker-v2-m3 lives at the bottom guarded by
``@pytest.mark.requires_models`` and is excluded from default runs via
``-m "not requires_models"``.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from firm.rag.chunk import Chunk
from firm.rag.rerank import BgeReranker
from firm.rag.retrieve import RetrievedChunk


# ---------------------------------------------------------------------------
# Stubs and fixtures
# ---------------------------------------------------------------------------


class StubCrossEncoder:
    """CrossEncoder stand-in scoring by query/doc token-overlap fraction."""

    def predict(self, pairs: list[list[str]]) -> list[float]:
        scores: list[float] = []
        for query, doc in pairs:
            q_tokens = set(query.lower().split())
            d_tokens = set(doc.lower().split())
            if not q_tokens:
                scores.append(0.0)
            else:
                scores.append(len(q_tokens & d_tokens) / len(q_tokens))
        return scores


class FixedScoreCrossEncoder:
    """Returns a pre-set list of scores in pair order (validates filter/sort)."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs: list[list[str]]) -> list[float]:
        if len(pairs) != len(self._scores):
            raise AssertionError(
                f"FixedScoreCrossEncoder expected {len(self._scores)} pairs, got {len(pairs)}"
            )
        return list(self._scores)


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=chunk_id.split("::")[0],
        ticker="AAPL",
        section="body",
        published_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        text=text,
        char_span=(0, len(text)),
        token_count=max(1, len(text.split())),
        source="test",
    )


def _make_retrieved(chunk_id: str, text: str, *, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=_make_chunk(chunk_id, text),
        score=score,
        rank_dense=0,
        rank_sparse=0,
    )


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------


def test_rerank_returns_top_k_in_descending_score() -> None:
    """5 candidates -> top 3 returned in descending rerank_score order."""
    query = "apple revenue 90 billion"
    candidates = [
        _make_retrieved("d::0001", "completely unrelated text about weather"),
        _make_retrieved("d::0002", "apple revenue 90 billion quarterly report"),  # full overlap
        _make_retrieved("d::0003", "apple revenue announced last quarter"),  # 2/4 overlap
        _make_retrieved("d::0004", "tesla revenue surged in china"),  # 1/4 overlap
        _make_retrieved("d::0005", "apple 90 billion in quarterly revenue figures"),  # 3/4 overlap
    ]

    reranker = BgeReranker(
        model_id="stub",
        score_floor=0.0,
        model=StubCrossEncoder(),
    )
    result = reranker.rerank(query, candidates, k=3)

    assert len(result) == 3
    assert result[0].rerank_score is not None
    assert result[1].rerank_score is not None
    assert result[2].rerank_score is not None
    assert result[0].rerank_score >= result[1].rerank_score >= result[2].rerank_score
    # The full-overlap chunk should be on top.
    assert result[0].chunk.id == "d::0002"


def test_rerank_filters_below_score_floor() -> None:
    """score_floor=0.5 drops candidates with rerank_score < 0.5."""
    candidates = [
        _make_retrieved("c0", "alpha"),
        _make_retrieved("c1", "bravo"),
        _make_retrieved("c2", "charlie"),
        _make_retrieved("c3", "delta"),
        _make_retrieved("c4", "echo"),
    ]
    # Fixed scores in candidate (pair) order: [0.9, 0.6, 0.4, 0.2, 0.7]
    scores = [0.9, 0.6, 0.4, 0.2, 0.7]
    reranker = BgeReranker(
        model_id="stub",
        score_floor=0.5,
        model=FixedScoreCrossEncoder(scores),
    )

    result = reranker.rerank("query", candidates, k=10)

    # Only candidates with score >= 0.5 survive: 0.9 (c0), 0.7 (c4), 0.6 (c1).
    assert [r.chunk.id for r in result] == ["c0", "c4", "c1"]
    assert [r.rerank_score for r in result] == [0.9, 0.7, 0.6]


def test_rerank_is_deterministic_for_same_input() -> None:
    """Same query+candidates+k must produce identical rerank output across calls."""
    query = "apple revenue 90 billion"
    candidates = [
        _make_retrieved("d::0001", "apple revenue 90 billion quarterly report"),
        _make_retrieved("d::0002", "tesla revenue surged in china"),
        _make_retrieved("d::0003", "apple 90 billion in quarterly revenue figures"),
        _make_retrieved("d::0004", "completely unrelated text about weather"),
    ]

    reranker = BgeReranker(
        model_id="stub",
        score_floor=0.0,
        model=StubCrossEncoder(),
    )

    first = reranker.rerank(query, candidates, k=4)
    second = reranker.rerank(query, candidates, k=4)

    assert [r.chunk.id for r in first] == [r.chunk.id for r in second]
    assert [r.rerank_score for r in first] == [r.rerank_score for r in second]


def test_rerank_empty_candidates_returns_empty() -> None:
    """Empty candidates short-circuits without ever calling the model."""

    class ExplodingEncoder:
        def predict(self, pairs: list[list[str]]) -> list[float]:
            raise AssertionError("predict() must not be called for empty candidates")

    reranker = BgeReranker(
        model_id="stub",
        score_floor=0.0,
        model=ExplodingEncoder(),
    )
    assert reranker.rerank("q", [], k=5) == []


# ---------------------------------------------------------------------------
# Real-model smoke test (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.requires_models
def test_real_bge_model_loads_and_scores() -> None:
    """Smoke-test against the real bge-reranker-v2-m3 — requires a local model."""
    reranker = BgeReranker(model_id="BAAI/bge-reranker-v2-m3")
    rc = _make_retrieved("d::0001", "Apple reported Q3 revenue of $90B")
    result = reranker.rerank("Apple Q3 revenue", [rc], k=1)

    assert len(result) == 1
    assert result[0].rerank_score is not None
    # bge-reranker-v2-m3 scores are roughly in [0, 1] after sigmoid; the
    # SentenceTransformers CrossEncoder applies the default activation.
    assert 0.0 <= result[0].rerank_score <= 1.0
