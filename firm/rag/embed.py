"""Dense and sparse embedders for the RAG pipeline.

- NomicEmbedder: lazy-loads nomic-ai/nomic-embed-text-v1.5 from sentence-transformers;
  produces L2-normalised float32 vectors of shape (N, 768).
- BM25Sparse: fits BM25Okapi over a corpus; tokenises every string through
  firm.rag.preprocess.ticker_aware_tokens so that $AAPL, BRK.B, 10-K survive
  as single atoms in the sparse index.

Neither class imports sentence-transformers at module level — deferred to first use so
test collection and import remain cheap.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

from firm.rag.preprocess import ticker_aware_tokens

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


_EMBED_DIM: int = 768
_MODEL_NAME: str = "nomic-ai/nomic-embed-text-v1.5"


class NomicEmbedder:
    """Dense embedder backed by nomic-embed-text-v1.5.

    Accepts an optional pre-built *model* kwarg so unit tests can inject a stub
    without triggering a model download.  When *model* is None the real
    SentenceTransformer is loaded on the first call to embed().
    """

    def __init__(self, model: SentenceTransformer | None = None) -> None:
        self._model: SentenceTransformer | None = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lazy_load(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer as _ST

            self._model = _ST(_MODEL_NAME, trust_remote_code=True)
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (N, 768) float32 array of L2-normalised embeddings.

        Empty input returns a zero-row array of the correct shape and dtype
        without ever touching the model.
        """
        if not texts:
            return np.zeros((0, _EMBED_DIM), dtype=np.float32)

        model = self._lazy_load()
        vectors: np.ndarray = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # Defensive L2-normalise in case the injected stub or a model version
        # does not guarantee unit norms.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1.0, norms)
        return vectors.astype(np.float32)


class BM25Sparse:
    """BM25Okapi-backed sparse embedder that preserves finance tokens.

    Vocabulary and IDF weights are built during .fit(); .transform() emits
    Qdrant sparse-vector dicts of the form {token_id: weight}.

    Tokenisation is always routed through ticker_aware_tokens so that dollar-
    prefixed tickers ($AAPL), dotted tickers (BRK.B), and filing form names
    (10-K) survive as single atoms in both the index and the query.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self._bm25: Any = None  # rank_bm25.BM25Okapi, imported lazily
        self._idf: dict[str, float] = {}

    def fit(self, corpus: Iterable[str]) -> None:
        """Tokenise each document with ticker_aware_tokens and build BM25Okapi."""
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

        tokenised: list[list[str]] = [ticker_aware_tokens(doc) for doc in corpus]

        self._bm25 = BM25Okapi(tokenised)

        # Build a stable token → int mapping over the union of all tokens.
        all_tokens: list[str] = sorted({tok for doc in tokenised for tok in doc})
        self.vocab = {tok: idx for idx, tok in enumerate(all_tokens)}

        # Cache IDF values for fast transform().
        # BM25Okapi stores idf as a dict keyed by token string after fitting.
        raw_idf: dict[str, float] = self._bm25.idf
        self._idf = {tok: float(raw_idf[tok]) for tok in self.vocab if tok in raw_idf}

    def transform(self, text: str) -> dict[int, float]:
        """Return a Qdrant sparse vector dict for *text*.

        Scores are IDF × term-frequency within *text* (TF counted over the
        tokenised query, not the corpus).  Tokens not in the fitted vocabulary
        are silently ignored; zero-weight entries are omitted.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Sparse.fit() must be called before transform()")

        tokens = ticker_aware_tokens(text)

        # Count TF in the query text.
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1

        result: dict[int, float] = {}
        for tok, count in tf.items():
            if tok not in self.vocab:
                continue
            idf = self._idf.get(tok, 0.0)
            weight = idf * count
            if weight != 0.0:
                result[self.vocab[tok]] = weight

        return result
