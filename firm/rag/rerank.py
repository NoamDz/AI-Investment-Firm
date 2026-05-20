"""Cross-encoder reranker (bge-reranker-v2-m3) for the RAG pipeline (T13).

Design decisions
----------------
* The cross-encoder is lazy-loaded on the first call to ``rerank()`` so that
  test collection and module import stay cheap.  A pre-built model may be
  injected via the constructor (mirrors :class:`firm.rag.embed.NomicEmbedder`)
  so unit tests can exercise the algorithm with a stub scorer.

* Scoring is a single batched ``predict(pairs)`` call: the cross-encoder is
  invoked once per ``rerank()`` regardless of candidate count.

* Filtering follows the design spec §7.4 floor of 0.3 by default.  Candidates
  with ``rerank_score < score_floor`` are dropped before truncation to ``k``.

* Sorting is stable (Python's ``sorted`` is stable), so candidates tied on
  ``rerank_score`` preserve the upstream RRF order — useful when the
  cross-encoder cannot discriminate between two passages and we fall back to
  the hybrid retriever's ranking.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from firm.rag.retrieve import RetrievedChunk


# ---------------------------------------------------------------------------
# Injected-collaborator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class _CrossEncoderProtocol(Protocol):
    """Structural type for ``sentence_transformers.CrossEncoder``.

    Only the ``predict`` method is required.  The real CrossEncoder accepts
    ``list[list[str]]`` (or list of tuples) and returns an iterable of floats;
    we coerce to ``list[float]`` in ``rerank()``.
    """

    def predict(self, pairs: list[list[str]]) -> Iterable[float]: ...


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class BgeReranker:
    """Cross-encoder reranker backed by ``BAAI/bge-reranker-v2-m3``.

    Parameters
    ----------
    model_id
        HuggingFace model identifier passed to ``CrossEncoder(...)`` when the
        real model is lazy-loaded.
    score_floor
        Minimum ``rerank_score`` to retain a candidate.  Defaults to 0.3 per
        design spec §7.4.
    model
        Optional pre-built cross-encoder to inject (e.g. a stub in unit
        tests).  When ``None`` the real ``CrossEncoder(model_id)`` is loaded
        on the first call to :meth:`rerank`.
    """

    def __init__(
        self,
        *,
        model_id: str,
        score_floor: float = 0.3,
        model: _CrossEncoderProtocol | None = None,
    ) -> None:
        self._model_id = model_id
        self._score_floor = score_floor
        self._model: _CrossEncoderProtocol | None = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lazy_load(self) -> _CrossEncoderProtocol:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_id)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        *,
        k: int,
    ) -> list[RetrievedChunk]:
        """Score, filter, and return the top-``k`` candidates by cross-encoder.

        Each returned :class:`RetrievedChunk` carries a populated
        ``rerank_score``.  Candidates scoring below ``score_floor`` are
        dropped before truncation.  When ``candidates`` is empty the model is
        never touched.
        """
        if not candidates:
            return []

        model = self._lazy_load()
        pairs: list[list[str]] = [[query, c.chunk.text] for c in candidates]
        raw_scores = model.predict(pairs)
        scores: list[float] = [float(s) for s in raw_scores]

        annotated: list[RetrievedChunk] = [
            c.model_copy(update={"rerank_score": score})
            for c, score in zip(candidates, scores, strict=True)
        ]

        # Filter by score floor.  ``rerank_score`` is set above, so the
        # ``is not None`` check is for the type-checker.
        filtered: list[RetrievedChunk] = [
            rc for rc in annotated
            if rc.rerank_score is not None and rc.rerank_score >= self._score_floor
        ]

        # Stable sort by rerank_score descending.  ``sorted`` is stable so
        # ties preserve the original RRF order.
        ordered = sorted(
            filtered,
            key=lambda rc: rc.rerank_score if rc.rerank_score is not None else float("-inf"),
            reverse=True,
        )

        return ordered[:k]
