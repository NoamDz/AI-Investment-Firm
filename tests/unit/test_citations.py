"""Tests for the Anthropic Citations API extractor (Plan 2 §T18).

The extractor takes a query plus a list of retrieved chunks, calls the
Anthropic Citations API via an injectable client, and emits one
:class:`firm.core.models.Claim` per citation entry in the response. Text
blocks lacking ``citations`` are dropped and counted on
``last_uncited_count`` so the upstream agent can surface a
``UNCITED_CLAIM`` failure mode.

These tests use a recording stub client (no real SDK) and inline canned
response dicts in the exact shape produced by the Anthropic Citations API.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from firm.llm.citations import AnthropicCitationsExtractor
from firm.llm.prompts import RESEARCH_SYSTEM
from firm.rag.chunk import Chunk


class _StubClient:
    """Recording stub satisfying the ``AnthropicMessagesClient`` Protocol."""

    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.last_kwargs: dict[str, object] | None = None

    def messages_create(self, **kwargs: object) -> dict[str, object]:
        self.last_kwargs = kwargs
        return self.response


def _chunk(idx: int, doc_id: str, text: str) -> Chunk:
    return Chunk(
        id=f"{doc_id}::{idx:04d}",
        doc_id=doc_id,
        ticker="AAPL",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        section="body",
        text=text,
        char_span=(0, len(text)),
        token_count=len(text.split()),
    )


def _three_citations_response() -> dict[str, Any]:
    """Three citation-anchored content blocks, one citation each."""
    return {
        "content": [
            {
                "type": "text",
                "text": "Apple reported revenue of $383.3B in fiscal 2023.",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "Total net sales were $383.285 billion",
                        "document_index": 0,
                        "document_title": "AAPL 10-K FY2023",
                        "start_char_index": 100,
                        "end_char_index": 138,
                    }
                ],
            },
            {
                "type": "text",
                "text": "Gross margin expanded to 44.1%.",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "Gross margin was 44.1%",
                        "document_index": 0,
                        "document_title": "AAPL 10-K FY2023",
                        "start_char_index": 200,
                        "end_char_index": 222,
                    }
                ],
            },
            {
                "type": "text",
                "text": "Services revenue hit $85.2B.",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "Services revenue of $85.2 billion",
                        "document_index": 1,
                        "document_title": "AAPL 10-Q Q1 FY2024",
                        "start_char_index": 50,
                        "end_char_index": 83,
                    }
                ],
            },
        ],
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


def test_extractor_emits_one_claim_per_citation() -> None:
    chunks = [
        _chunk(0, "AAPL-10K-2023", "Total net sales were $383.285 billion"),
        _chunk(0, "AAPL-10Q-Q1-2024", "Services revenue of $85.2 billion"),
    ]
    stub = _StubClient(_three_citations_response())
    extractor = AnthropicCitationsExtractor(
        client=stub, model="claude-sonnet-4-6", max_tokens=1024
    )

    claims = extractor.extract(
        query="What were Apple's FY2023 financials?",
        chunks=chunks,
        as_of=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )

    assert len(claims) == 3
    for c in claims:
        assert c.source_chunk_id is not None

    # The first two claims map back to the first chunk (document_index 0),
    # the third to the second chunk (document_index 1).
    assert claims[0].source_chunk_id == chunks[0].id
    assert claims[1].source_chunk_id == chunks[0].id
    assert claims[2].source_chunk_id == chunks[1].id

    # Claim text comes from the block-level text (the model's assertion),
    # NOT the verbatim cited_text quote.
    assert claims[0].text == "Apple reported revenue of $383.3B in fiscal 2023."


def test_extractor_rejects_uncited_claim() -> None:
    response: dict[str, Any] = {
        "content": [
            {
                "type": "text",
                "text": "Apple is a great company.",
                # no citations -> must be dropped
            },
            {
                "type": "text",
                "text": "Revenue grew 5% YoY.",
                "citations": [
                    {
                        "type": "char_location",
                        "cited_text": "Revenue increased 5% year-over-year",
                        "document_index": 0,
                        "document_title": "AAPL 10-K",
                        "start_char_index": 0,
                        "end_char_index": 35,
                    }
                ],
            },
        ],
        "usage": {"input_tokens": 50, "output_tokens": 30},
    }
    chunks = [_chunk(0, "AAPL-10K", "Revenue increased 5% year-over-year")]
    stub = _StubClient(response)
    extractor = AnthropicCitationsExtractor(
        client=stub, model="claude-sonnet-4-6", max_tokens=1024
    )

    claims = extractor.extract(
        query="How did revenue change?",
        chunks=chunks,
        as_of=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )

    assert len(claims) == 1
    assert claims[0].text == "Revenue grew 5% YoY."
    assert extractor.last_uncited_count == 1


def test_extractor_carries_source_span() -> None:
    chunks = [
        _chunk(0, "AAPL-10K-2023", "Total net sales were $383.285 billion"),
        _chunk(0, "AAPL-10Q-Q1-2024", "Services revenue of $85.2 billion"),
    ]
    stub = _StubClient(_three_citations_response())
    extractor = AnthropicCitationsExtractor(
        client=stub, model="claude-sonnet-4-6", max_tokens=1024
    )

    claims = extractor.extract(
        query="What were Apple's FY2023 financials?",
        chunks=chunks,
        as_of=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )

    for c in claims:
        assert c.source_span is not None
    assert claims[0].source_span == (100, 138)
    assert claims[1].source_span == (200, 222)
    assert claims[2].source_span == (50, 83)


def test_extractor_passes_documents_with_citations_enabled() -> None:
    chunks = [
        _chunk(0, "AAPL-10K-2023", "Total net sales were $383.285 billion"),
        _chunk(0, "AAPL-10Q-Q1-2024", "Services revenue of $85.2 billion"),
    ]
    stub = _StubClient(_three_citations_response())
    extractor = AnthropicCitationsExtractor(
        client=stub, model="claude-sonnet-4-6", max_tokens=1024
    )

    extractor.extract(
        query="What were Apple's FY2023 financials?",
        chunks=chunks,
        as_of=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )

    assert stub.last_kwargs is not None
    kwargs = stub.last_kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["system"] == RESEARCH_SYSTEM
    assert kwargs["max_tokens"] == 1024
    assert kwargs["temperature"] == 0.0

    messages = kwargs["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 1
    msg = messages[0]
    assert isinstance(msg, dict)
    assert msg["role"] == "user"

    content = msg["content"]
    assert isinstance(content, list)
    # Two documents + one user-text block.
    document_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "document"]
    text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
    assert len(document_blocks) == 2
    assert len(text_blocks) == 1

    for i, doc_block in enumerate(document_blocks):
        assert doc_block["citations"] == {"enabled": True}
        source = doc_block["source"]
        assert isinstance(source, dict)
        assert source["type"] == "text"
        assert source["media_type"] == "text/plain"
        assert source["data"] == chunks[i].text
        assert doc_block["title"] == chunks[i].doc_id


def test_extractor_resets_uncited_count_per_call() -> None:
    """Regression guard: ``last_uncited_count`` must reset on every ``extract``."""
    uncited_response: dict[str, Any] = {
        "content": [
            {"type": "text", "text": "Uncited prose here."},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    chunks = [_chunk(0, "AAPL-10K", "Some evidence text")]
    stub = _StubClient(uncited_response)
    extractor = AnthropicCitationsExtractor(
        client=stub, model="claude-sonnet-4-6", max_tokens=1024
    )

    extractor.extract(
        query="q1",
        chunks=chunks,
        as_of=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    assert extractor.last_uncited_count == 1

    # Second call returns a fully cited response; count must reset to zero.
    stub.response = _three_citations_response()
    extractor.extract(
        query="q2",
        chunks=[
            _chunk(0, "AAPL-10K-2023", "Total net sales were $383.285 billion"),
            _chunk(0, "AAPL-10Q-Q1-2024", "Services revenue of $85.2 billion"),
        ],
        as_of=datetime(2024, 6, 1, tzinfo=timezone.utc),
    )
    assert extractor.last_uncited_count == 0
