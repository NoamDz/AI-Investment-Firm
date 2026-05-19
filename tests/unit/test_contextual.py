"""Tests for contextual augmentation: per-doc Haiku summaries attached to chunks."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from collections.abc import Sequence

from firm.llm.client import CompletionResponse
from firm.rag.chunk import Chunk
from firm.rag.contextual import ContextualAugmenter
from firm.rag.source import FilingDoc


# ---------------------------------------------------------------------------
# Fake client
# ---------------------------------------------------------------------------


class FakeAnthropicClient:
    """In-memory stub that honours the AnthropicClient Protocol.

    Tracks real API hits vs. cache hits via a simple in-memory dict keyed by
    a hash of (model, messages_repr).  When the same prompt hash is seen a
    second time the response is served from the local dict and api_calls is
    NOT incremented.
    """

    def __init__(self, *, summary_text: str = "Fake summary.") -> None:
        self._summary_text = summary_text
        self._cache: dict[str, CompletionResponse] = {}
        self.api_calls: int = 0
        self.last_call: dict[str, object] = {}

    @staticmethod
    def _key(model: str, messages: Sequence[dict[str, object]]) -> str:
        raw = f"{model}|{messages!r}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def complete(
        self,
        *,
        model: str,
        system: str | None,
        messages: Sequence[dict[str, object]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> CompletionResponse:
        self.last_call = {"model": model, "system": system, "messages": messages}
        key = self._key(model, messages)
        if key in self._cache:
            return CompletionResponse(
                text=self._cache[key].text,
                cache_hit=True,
            )
        response = CompletionResponse(text=self._summary_text, cache_hit=False)
        self._cache[key] = response
        self.api_calls += 1
        return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_doc(
    html: str = "<p>Apple Inc. is a technology company that designs consumer electronics.</p>",
) -> FilingDoc:
    return FilingDoc(
        doc_id="AAPL-2024-10K",
        ticker="AAPL",
        filing_type="10-K",
        published_at=datetime(2024, 1, 31, tzinfo=timezone.utc),
        title="Apple Inc. Annual Report 2024",
        html=html,
    )


def _make_chunks(doc: FilingDoc, n: int = 8) -> list[Chunk]:
    return [
        Chunk(
            id=f"{doc.doc_id}::{i:04d}",
            doc_id=doc.doc_id,
            ticker=doc.ticker,
            published_at=doc.published_at,
            section="body",
            text=f"Chunk text {i}.",
            char_span=(i * 10, i * 10 + 10),
            token_count=3,
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_summary_generated_once_per_doc_and_reused() -> None:
    """8 chunks from the same doc → exactly 1 Haiku complete() call."""
    client = FakeAnthropicClient()
    augmenter = ContextualAugmenter(
        client=client,
        model="claude-haiku-4-5",
    )
    doc = _make_doc()
    chunks = _make_chunks(doc, n=8)

    result = augmenter.augment(doc, chunks)

    assert len(result) == 8
    assert client.api_calls == 1


def test_summary_attached_to_each_chunk() -> None:
    """doc_summary is non-empty on every chunk returned by augment()."""
    client = FakeAnthropicClient(summary_text="Apple Inc. 2024 10-K annual filing context.")
    augmenter = ContextualAugmenter(
        client=client,
        model="claude-haiku-4-5",
    )
    doc = _make_doc()
    chunks = _make_chunks(doc, n=4)

    result = augmenter.augment(doc, chunks)

    for chunk in result:
        assert chunk.doc_summary, "doc_summary must be non-empty on every chunk"
        assert "Apple Inc." in (chunk.doc_summary or "")


def test_summary_uses_llm_cache() -> None:
    """Second augment() on the same doc returns cached response; real_api_calls stays 1."""
    client = FakeAnthropicClient(summary_text="Cached summary text.")
    augmenter = ContextualAugmenter(
        client=client,
        model="claude-haiku-4-5",
    )
    doc = _make_doc()
    chunks = _make_chunks(doc, n=3)

    first_result = augmenter.augment(doc, chunks)
    assert client.api_calls == 1

    second_result = augmenter.augment(doc, chunks)
    # The cache (inside fake client) should have served the second call without a new API hit.
    assert client.api_calls == 1, "Second invocation must not increment real API calls"

    # Both results must have doc_summary populated.
    for chunk in first_result + second_result:
        assert chunk.doc_summary == "Cached summary text."


def test_augment_strips_html_before_calling_haiku() -> None:
    """HTML tags must be absent from the <doc>...</doc> excerpt sent to Haiku."""
    client = FakeAnthropicClient()
    augmenter = ContextualAugmenter(client=client, model="claude-haiku-4-5", max_tokens=512)
    doc = _make_doc("<html><body><p>Apple Inc. quarterly revenue grew.</p></body></html>")
    chunks = _make_chunks(doc, n=2)

    augmenter.augment(doc, chunks)

    messages = client.last_call["messages"]
    assert isinstance(messages, list)
    user_content = str(messages[0]["content"])
    # Extract the text between <doc> and </doc> and assert it contains no HTML tags.
    doc_start = user_content.index("<doc>") + len("<doc>")
    doc_end = user_content.index("</doc>")
    doc_section = user_content[doc_start:doc_end]
    assert "<" not in doc_section, f"HTML tags found in doc excerpt: {doc_section!r}"
