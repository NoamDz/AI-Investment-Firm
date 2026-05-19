"""Tests for firm.rag.embed — T8 spec compliance.

Run fast tests with: pytest tests/unit/test_embed.py -v -m "not requires_models"
"""
from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_stub_encode(seed: int = 42) -> Callable[..., np.ndarray]:
    """Return a deterministic encode function backed by a seeded RNG."""
    rng = np.random.RandomState(seed)  # noqa: NPY002 — intentional seeded state

    def _encode(
        texts: list[str],
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        n = len(texts)
        raw = rng.randn(n, 768).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            raw = raw / np.where(norms == 0, 1.0, norms)
        return raw

    return _encode


@pytest.fixture()
def stub_dense_model() -> MagicMock:
    """A tiny SentenceTransformer stub that returns deterministic (N, 768) arrays."""
    mock = MagicMock()
    mock.encode.side_effect = _make_stub_encode(seed=42)
    return mock


# ---------------------------------------------------------------------------
# Dense embedder tests
# ---------------------------------------------------------------------------


def test_dense_embedder_returns_768_dim_unit_vectors(stub_dense_model: MagicMock) -> None:
    from firm.rag.embed import NomicEmbedder

    embedder = NomicEmbedder(model=stub_dense_model)
    texts = ["Apple reported strong quarterly results.", "Tesla delivered record vehicles."]
    result = embedder.embed(texts)

    assert result.shape == (2, 768), f"Expected shape (2, 768), got {result.shape}"
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)


def test_dense_embedder_is_deterministic(stub_dense_model: MagicMock) -> None:
    """Identical inputs must produce bit-identical outputs to 1e-6 tolerance."""
    from firm.rag.embed import NomicEmbedder

    # Two separate embedders sharing the same stub (same underlying encode side_effect
    # seed is reset per-fixture call, so use the same instance).
    embedder = NomicEmbedder(model=stub_dense_model)
    texts = ["$AAPL earnings beat consensus estimates."]

    # Call encode twice; the stub returns the same deterministic array because
    # sentence-transformers is deterministic for the same model weights + input.
    # We reset the side_effect to a fresh seeded RNG to simulate determinism.
    stub_dense_model.encode.side_effect = _make_stub_encode(seed=42)
    first = embedder.embed(texts)
    stub_dense_model.encode.side_effect = _make_stub_encode(seed=42)
    second = embedder.embed(texts)

    np.testing.assert_allclose(first, second, atol=1e-6)


def test_batch_embed_handles_empty_input(stub_dense_model: MagicMock) -> None:
    from firm.rag.embed import NomicEmbedder

    embedder = NomicEmbedder(model=stub_dense_model)
    result = embedder.embed([])

    assert result.shape == (0, 768), f"Expected shape (0, 768), got {result.shape}"
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# Sparse BM25 tests
# ---------------------------------------------------------------------------

_FIXTURE_CORPUS = [
    "$AAPL announced record iPhone sales in Q3 2024.",
    "BRK.B subsidiary GEICO reported underwriting gains.",
    "The 10-K filing disclosed significant capital expenditures.",
    "Microsoft Azure revenue grew 29 percent year-over-year.",
]


def test_sparse_embedder_preserves_ticker_tokens() -> None:
    """Vocabulary built from ticker-aware tokenization must retain $AAPL and BRK.B."""
    from firm.rag.embed import BM25Sparse

    sparse = BM25Sparse()
    sparse.fit(_FIXTURE_CORPUS)

    assert "$AAPL" in sparse.vocab, "$AAPL missing from BM25Sparse vocabulary"
    assert "BRK.B" in sparse.vocab, "BRK.B missing from BM25Sparse vocabulary"


def test_sparse_transform_returns_dict_int_float() -> None:
    """transform() must return dict[int, float] with non-zero entries only."""
    from firm.rag.embed import BM25Sparse

    sparse = BM25Sparse()
    sparse.fit(_FIXTURE_CORPUS)
    result = sparse.transform("$AAPL quarterly results")

    assert isinstance(result, dict), "transform() must return a dict"
    for k, v in result.items():
        assert isinstance(k, int), f"key {k!r} is not int"
        assert isinstance(v, float), f"value {v!r} is not float"
        assert v != 0.0, "transform() must omit zero-weight entries"


def test_sparse_transform_unknown_token_is_ignored() -> None:
    """Tokens absent from the fitted vocabulary must not appear in transform output."""
    from firm.rag.embed import BM25Sparse

    sparse = BM25Sparse()
    sparse.fit(_FIXTURE_CORPUS)
    # "xyzzy" was never in the corpus; its id would not exist in vocab.
    result = sparse.transform("xyzzy randomtoken999")
    assert result == {}


# ---------------------------------------------------------------------------
# Integration test — real model (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.requires_models
def test_nomic_embedder_real_model_shape() -> None:
    """Smoke-test against the real nomic model — requires a local model download."""
    from firm.rag.embed import NomicEmbedder

    embedder = NomicEmbedder()
    result = embedder.embed(["Test sentence for shape verification."])
    assert result.shape[1] == 768
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, np.ones(1), atol=1e-5)
