"""Cited claim extractor backed by the Anthropic Citations API. See Plan 2 §T18.

This module turns a query + a list of retrieved :class:`firm.rag.chunk.Chunk`
objects into a list of :class:`firm.core.models.Claim` objects, each anchored
to a specific retrieved chunk via ``source_chunk_id`` and ``source_span``.

The extractor builds the user message as a sequence of ``document`` content
blocks (one per chunk, each with ``citations: {enabled: true}``) followed by a
single ``text`` block carrying the question. It then iterates the response
content: every text block that carries a non-empty ``citations`` array produces
one :class:`Claim` per citation entry; text blocks without citations are
DROPPED, with the count surfaced on :attr:`AnthropicCitationsExtractor.last_uncited_count`
so the upstream agent can raise an ``UNCITED_CLAIM`` failure mode.

Claim ``text`` is set from the block-level ``text`` (the model's natural-
language assertion). The verbatim ``cited_text`` from each citation is
reserved for the :class:`firm.core.models.Citation` object that T19 will
attach alongside the claim.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from firm.core.models import Claim
from firm.llm.prompts import RESEARCH_SYSTEM
from firm.rag.chunk import Chunk


@runtime_checkable
class AnthropicMessagesClient(Protocol):
    """Narrow Protocol over ``messages_create`` for dependency injection.

    Both :class:`firm.llm.anthropic_client.CachedAnthropicClient` and a hand-
    rolled test stub satisfy this Protocol. The full T9
    :class:`firm.llm.client.AnthropicClient` Protocol returns a flattened
    :class:`firm.llm.client.CompletionResponse` and therefore cannot expose
    the Citations API content blocks that this extractor depends on.
    """

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, object]],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, object]: ...


@runtime_checkable
class CitedClaimExtractor(Protocol):
    """Abstract extractor boundary so agents can swap implementations.

    Implementations are responsible for ensuring every emitted ``Claim``
    carries provenance (``source_chunk_id``, ``source_span``) and for
    accounting any text the LLM produced without a citation so the caller
    can react.
    """

    def extract(
        self,
        *,
        query: str,
        chunks: list[Chunk],
        as_of: datetime,
    ) -> list[Claim]: ...


class AnthropicCitationsExtractor:
    """Cited claim extractor backed by the Anthropic Citations API.

    Attributes
    ----------
    last_uncited_count:
        Number of text blocks in the most recent ``extract`` call that the
        model emitted without any citations. Reset to ``0`` at the start of
        every ``extract`` call. Upstream agents read this attribute to decide
        whether to raise an ``UNCITED_CLAIM`` failure mode.
    """

    def __init__(
        self,
        *,
        client: AnthropicMessagesClient,
        model: str,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self.last_uncited_count: int = 0

    def extract(
        self,
        *,
        query: str,
        chunks: list[Chunk],
        as_of: datetime,  # noqa: ARG002 -- reserved for T19 question template
    ) -> list[Claim]:
        self.last_uncited_count = 0

        # One ``document`` content block per chunk, each opted-in to citations.
        document_blocks: list[dict[str, object]] = [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": chunk.text,
                },
                "title": chunk.doc_id,
                "citations": {"enabled": True},
            }
            for chunk in chunks
        ]
        user_content: list[dict[str, object]] = [
            *document_blocks,
            {"type": "text", "text": query},
        ]
        messages: list[dict[str, object]] = [
            {"role": "user", "content": user_content},
        ]

        response = self._client.messages_create(
            model=self._model,
            system=RESEARCH_SYSTEM,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

        claims: list[Claim] = []
        content = response.get("content", [])
        if not isinstance(content, list):
            return claims

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "text":
                continue

            text_val = block.get("text", "")
            if not isinstance(text_val, str):
                continue

            citations_raw = block.get("citations")
            if not isinstance(citations_raw, list) or len(citations_raw) == 0:
                self.last_uncited_count += 1
                continue

            emitted_before = len(claims)
            for citation in citations_raw:
                if not isinstance(citation, dict):
                    continue
                doc_index_raw = citation.get("document_index")
                start_raw = citation.get("start_char_index")
                end_raw = citation.get("end_char_index")
                if (
                    not isinstance(doc_index_raw, int)
                    or not isinstance(start_raw, int)
                    or not isinstance(end_raw, int)
                ):
                    continue
                if doc_index_raw < 0 or doc_index_raw >= len(chunks):
                    continue
                source_chunk_id = chunks[doc_index_raw].id
                claims.append(
                    Claim(
                        text=text_val,
                        source_chunk_id=source_chunk_id,
                        source_span=(start_raw, end_raw),
                    )
                )

            # Defensive belt: a non-empty citations list whose every entry
            # failed our parsing guards still represents evidence we cannot
            # ground. Surface it on last_uncited_count rather than silently
            # discarding the block-level text.
            if len(claims) == emitted_before:
                self.last_uncited_count += 1

        return claims


__all__ = [
    "AnthropicCitationsExtractor",
    "AnthropicMessagesClient",
    "CitedClaimExtractor",
]
