"""Chunk model and finance-aware document chunker. See design spec §5.3.

Input text is taken directly from FilingDoc.html (raw HTML string). T5 will replace
this with a proper finance-aware HTML→text extractor; for now the raw HTML is the
corpus text. char_span offsets are therefore into FilingDoc.html.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import tiktoken
from pydantic import BaseModel, Field

from firm.rag.source import FilingDoc

_ENCODING = tiktoken.encoding_for_model("gpt-4")


class Chunk(BaseModel):
    id: str
    doc_id: str
    ticker: str
    published_at: datetime
    section: str
    text: str
    char_span: tuple[int, int]
    token_count: int
    doc_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def chunk_document(
    doc: FilingDoc,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    if doc.published_at is None:
        raise ValueError("FilingDoc.published_at is required; got None")

    text = doc.html
    all_tokens = _ENCODING.encode(text)
    total = len(all_tokens)

    if total == 0:
        return []

    step = target_tokens - overlap_tokens
    chunks: list[Chunk] = []
    start_tok = 0

    while start_tok < total:
        end_tok = min(start_tok + target_tokens, total)
        window_tokens = all_tokens[start_tok:end_tok]
        window_text = _ENCODING.decode(window_tokens)

        # Locate char_span by finding the decoded window inside the source text.
        # We search from approximately where the previous chunk ended to keep
        # find() efficient and avoid false matches for repeated phrases.
        search_from = 0 if not chunks else chunks[-1].char_span[0]
        char_start = text.find(window_text, search_from)
        if char_start == -1:
            # Fallback: search from the beginning (should not normally happen).
            char_start = text.find(window_text)
        char_end = char_start + len(window_text) if char_start != -1 else len(window_text)

        idx = len(chunks)
        chunks.append(
            Chunk(
                id=f"{doc.doc_id}::{idx:04d}",
                doc_id=doc.doc_id,
                ticker=doc.ticker,
                published_at=doc.published_at,
                section="body",
                text=window_text,
                char_span=(char_start, char_end),
                token_count=len(window_tokens),
                metadata=dict(doc.metadata),
            )
        )

        if end_tok == total:
            break
        start_tok += step

    return chunks
