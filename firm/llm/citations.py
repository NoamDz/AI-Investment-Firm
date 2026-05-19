"""Cited claim extractor backed by the Anthropic Citations API. See Plan 2 §T18/§T24.

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

Tool dispatch (§T24)
--------------------
When ``tools`` is provided to the constructor, the extractor passes the
Anthropic-shaped ``tools=`` payload on every ``messages_create`` call. If
the first response contains ``tool_use`` blocks, the extractor executes each
matching tool (``tool.run(**block["input"])``), assembles ``tool_result``
content blocks, and makes a second ``messages_create`` call. Text-block
iteration (and Claim emission) then uses the *second* response. If no tool
calls were made, the first response is used directly.

Tool-derived Claims carry ``tool_call_id`` (set from the Anthropic block id)
and ``value`` (the Decimal returned by the tool). They satisfy the Claim
validator since ``tool_call_id is not None``.
"""
from __future__ import annotations

import copy
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, runtime_checkable

from firm.core.models import Claim
from firm.llm.prompts import RESEARCH_SYSTEM
from firm.rag.chunk import Chunk
from firm.tools import Tool

_log = logging.getLogger(__name__)


@runtime_checkable
class AnthropicMessagesClient(Protocol):
    """Narrow Protocol over ``messages_create`` for dependency injection.

    Both :class:`firm.llm.anthropic_client.CachedAnthropicClient` and a hand-
    rolled test stub satisfy this Protocol. The full T9
    :class:`firm.llm.client.AnthropicClient` Protocol returns a flattened
    :class:`firm.llm.client.CompletionResponse` and therefore cannot expose
    the Citations API content blocks that this extractor depends on.

    The ``tools`` parameter is optional (``None`` when no tools are configured)
    so that stubs using ``**kwargs`` continue to satisfy the Protocol.
    """

    def messages_create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
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

    The ``last_tool_call_ids`` attribute is part of the Protocol contract
    (Plan 2 §T24): implementations must surface the tool-use block ids
    executed during the most recent ``extract`` call so the research agent
    can write them to working state.
    """

    last_tool_call_ids: list[str]

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
    last_tool_call_ids:
        List of tool-use block ids that were executed during the most recent
        ``extract`` call. Reset to ``[]`` at the start of every call. The
        research agent reads this to surface ``tool_call_ids`` on the working
        state.
    last_tool_error_count:
        Number of tool-use blocks where the tool raised an exception during
        the most recent ``extract`` call. Skipped defensively (no re-raise).
    """

    def __init__(
        self,
        *,
        client: AnthropicMessagesClient,
        model: str,
        max_tokens: int = 1024,
        tools: list[Tool] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self.last_uncited_count: int = 0
        self.last_tool_call_ids: list[str] = []
        self.last_tool_error_count: int = 0

        if tools:
            self._tools_by_name: dict[str, Tool] = {t.tool_def.name: t for t in tools}
            self._tool_payload: list[dict[str, object]] | None = [
                {
                    "name": t.tool_def.name,
                    "description": t.tool_def.description,
                    # deepcopy: ``input_schema`` is a MappingProxyType wrapping
                    # nested dicts/lists. A shallow dict(...) copy would still
                    # share nested ``properties``/``required``/``enum`` refs,
                    # so a downstream mutation could leak back into the tool's
                    # immutable schema.
                    "input_schema": copy.deepcopy(dict(t.tool_def.input_schema)),
                }
                for t in tools
            ]
        else:
            self._tools_by_name = {}
            self._tool_payload = None

    def extract(
        self,
        *,
        query: str,
        chunks: list[Chunk],
        as_of: datetime,  # noqa: ARG002 -- reserved for T19 question template
    ) -> list[Claim]:
        self.last_uncited_count = 0
        self.last_tool_call_ids = []
        self.last_tool_error_count = 0

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

        # First API call — pass tools= if configured.
        first_response = self._client.messages_create(
            model=self._model,
            system=RESEARCH_SYSTEM,
            messages=messages,
            tools=self._tool_payload,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

        # Collect tool_use blocks and execute them.
        tool_records: list[tuple[str, str, Any, Decimal]] = []  # (id, name, input, value)
        first_content = first_response.get("content", [])
        if isinstance(first_content, list):
            for block in first_content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "tool_use":
                    continue
                tool_id = block.get("id")
                tool_name = block.get("name")
                tool_input = block.get("input", {})
                if not isinstance(tool_id, str) or not isinstance(tool_name, str):
                    continue
                tool = self._tools_by_name.get(tool_name)
                if tool is None:
                    _log.debug("Unknown tool %r — skipping tool_use block %r", tool_name, tool_id)
                    continue
                if not isinstance(tool_input, dict):
                    tool_input = {}
                try:
                    result_value: Decimal = tool.run(**tool_input)
                except Exception:
                    _log.warning(
                        "Tool %r raised during extract; skipping block %r",
                        tool_name,
                        tool_id,
                        exc_info=True,
                    )
                    self.last_tool_error_count += 1
                    continue
                tool_records.append((tool_id, tool_name, tool_input, result_value))
                self.last_tool_call_ids.append(tool_id)

        # Determine which response to use for text-block iteration.
        if tool_records:
            # Build tool_result content blocks for the second turn.
            tool_result_blocks: list[dict[str, object]] = [
                {
                    "type": "tool_result",
                    "tool_use_id": rec_id,
                    "content": str(rec_value),
                }
                for rec_id, _name, _inp, rec_value in tool_records
            ]
            second_messages: list[dict[str, object]] = [
                *messages,
                {"role": "assistant", "content": first_content},
                {"role": "user", "content": tool_result_blocks},
            ]
            active_response = self._client.messages_create(
                model=self._model,
                system=RESEARCH_SYSTEM,
                messages=second_messages,
                tools=self._tool_payload,
                max_tokens=self._max_tokens,
                temperature=0.0,
            )
        else:
            active_response = first_response

        # Emit Claims from the active response's text blocks.
        claims: list[Claim] = []
        content = active_response.get("content", [])
        if not isinstance(content, list):
            # Still emit tool-derived Claims even if text content is malformed.
            claims.extend(self._build_tool_claims(tool_records))
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

        # Append tool-derived Claims after text-block Claims.
        claims.extend(self._build_tool_claims(tool_records))
        return claims

    def _build_tool_claims(
        self, tool_records: list[tuple[str, str, Any, Decimal]]
    ) -> list[Claim]:
        """Build one Claim per executed tool call."""
        result: list[Claim] = []
        for tool_id, tool_name, _tool_input, value in tool_records:
            result.append(
                Claim(
                    text=f"Tool {tool_name} returned {value}",
                    value=value,
                    unit=None,
                    source_chunk_id=None,
                    source_span=None,
                    tool_call_id=tool_id,
                )
            )
        return result


__all__ = [
    "AnthropicCitationsExtractor",
    "AnthropicMessagesClient",
    "CitedClaimExtractor",
]
