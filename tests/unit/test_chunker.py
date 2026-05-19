"""Tests for firm.rag.chunk — T4 spec compliance."""
from __future__ import annotations

import types
from datetime import datetime, timezone

import pytest
import tiktoken

from firm.rag.chunk import Chunk, chunk_document
from firm.rag.source import FilingDoc

_ENC = tiktoken.encoding_for_model("gpt-4")

_TZ = timezone.utc
_PUBLISHED_AT = datetime(2024, 1, 15, tzinfo=_TZ)

TARGET = 512
OVERLAP = 64


def _make_doc(text: str, published_at: datetime | None = _PUBLISHED_AT) -> FilingDoc:
    return FilingDoc(
        doc_id="TEST-001",
        ticker="AAPL",
        filing_type="10-K",
        published_at=published_at,  # type: ignore[arg-type]
        title="Annual Report",
        html=text,
    )


def _synthetic_text(approx_tokens: int) -> str:
    unit = "This is sentence number {n}. "
    unit_tokens = len(_ENC.encode(unit.format(n=999)))
    repeats = (approx_tokens // unit_tokens) + 1
    return "".join(unit.format(n=i) for i in range(repeats))


def test_chunk_target_size_within_tolerance() -> None:
    text = _synthetic_text(3000)
    doc = _make_doc(text)
    chunks = chunk_document(doc, target_tokens=TARGET, overlap_tokens=OVERLAP)

    assert len(chunks) >= 2, "Expected multiple chunks for a ~3000-token document"

    non_last = chunks[:-1]
    for chunk in non_last:
        assert chunk.token_count <= TARGET * 1.2, (
            f"Chunk token_count {chunk.token_count} exceeds 120% of target {TARGET}"
        )
        assert chunk.token_count >= TARGET * 0.8, (
            f"Chunk token_count {chunk.token_count} is below 80% of target {TARGET}"
        )


def test_chunk_overlap_present() -> None:
    text = _synthetic_text(3000)
    doc = _make_doc(text)
    chunks = chunk_document(doc, target_tokens=TARGET, overlap_tokens=OVERLAP)

    assert len(chunks) >= 2

    full_text = doc.html
    for i in range(len(chunks) - 1):
        tokens_i = _ENC.encode(chunks[i].text)
        tokens_next = _ENC.encode(chunks[i + 1].text)

        tail_tokens = tokens_i[-OVERLAP:]
        head_tokens = tokens_next[:OVERLAP]
        assert tail_tokens == head_tokens, (
            f"Overlap mismatch between chunk {i} and chunk {i + 1}"
        )

    _ = full_text


def test_chunk_preserves_published_at_and_metadata() -> None:
    text = _synthetic_text(3000)
    extra_meta = {"source": "edgar", "form": "10-K"}
    doc = FilingDoc(
        doc_id="META-002",
        ticker="MSFT",
        filing_type="10-K",
        published_at=_PUBLISHED_AT,
        title="Annual Report",
        html=text,
        metadata=extra_meta,
    )
    chunks = chunk_document(doc, target_tokens=TARGET, overlap_tokens=OVERLAP)

    assert len(chunks) >= 1
    for idx, chunk in enumerate(chunks):
        assert isinstance(chunk, Chunk), f"chunk {idx} is not a Chunk instance"
        assert chunk.published_at == _PUBLISHED_AT, f"chunk {idx} published_at mismatch"
        assert chunk.ticker == "MSFT", f"chunk {idx} ticker mismatch"
        assert chunk.doc_id == "META-002", f"chunk {idx} doc_id mismatch"
        assert chunk.section == "body", f"chunk {idx} section should be 'body'"
        assert chunk.id == f"META-002::{idx:04d}", f"chunk {idx} id format mismatch"


def test_chunker_rejects_doc_without_published_at() -> None:
    text = _synthetic_text(100)
    duck = types.SimpleNamespace(
        doc_id="BAD-001",
        ticker="TSLA",
        filing_type="10-K",
        published_at=None,
        title="Bad Doc",
        html=text,
        metadata={},
    )
    with pytest.raises(ValueError, match="published_at"):
        chunk_document(duck, target_tokens=TARGET, overlap_tokens=OVERLAP)  # type: ignore[arg-type]
