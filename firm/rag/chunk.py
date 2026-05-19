"""Chunk model and finance-aware document chunker. See design spec §5.3.

Input text is taken directly from FilingDoc.html (raw HTML string). T5 will replace
this with a proper finance-aware HTML→text extractor; for now the raw HTML is the
corpus text. char_span offsets are therefore into FilingDoc.html.

T5 will populate `section` based on filing structure (`item_1A`, `risk_factors`, etc.);
for T4 every chunk is tagged `body`.
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
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be < target_tokens")

    text = doc.html
    all_tokens = _ENCODING.encode(text)
    total = len(all_tokens)

    if total == 0:
        return []

    step = target_tokens - overlap_tokens
    chunks: list[Chunk] = []

    # char_cursor tracks the character offset of the first token of the current window.
    # We advance it by the decoded length of the step prefix after each window so that
    # it always points to the exact start of the next window in the source text.
    char_cursor = 0

    for window_start_token_idx in range(0, total, step):
        window_tokens = all_tokens[window_start_token_idx : window_start_token_idx + target_tokens]
        window_text = _ENCODING.decode(window_tokens)

        char_start = char_cursor
        char_end = char_start + len(window_text)

        idx = len(chunks)

        # Verify the BPE round-trip is exact so we never emit a corrupt span.
        if text[char_start:char_end] != window_text:
            raise RuntimeError(f"char_span resolution failed at chunk {idx}")

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

        # Advance cursor by the decoded length of the step prefix of this window.
        # This is the portion not re-emitted in the next (overlapping) chunk.
        step_prefix = _ENCODING.decode(window_tokens[:step])
        char_cursor += len(step_prefix)

        # If this window consumed the last tokens, we are done.
        if window_start_token_idx + target_tokens >= total:
            break

    return chunks
