"""Contextual augmentation: attach a per-doc summary to every chunk via Haiku.

One LLM call is made per augment() invocation, regardless of the number of chunks.
Caching across invocations is the responsibility of the injected AnthropicClient
(T15 will supply a SQLite-backed CachedAnthropicClient).
"""
from __future__ import annotations

from firm.llm.client import AnthropicClient
from firm.rag.chunk import Chunk
from firm.rag.preprocess import normalize_text
from firm.rag.source import FilingDoc

_DOC_EXCERPT_CHARS: int = 2000


class ContextualAugmenter:
    """Generate one doc-level summary per filing and attach it to every chunk."""

    def __init__(self, *, client: AnthropicClient, model: str, max_tokens: int = 512) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def augment(self, doc: FilingDoc, chunks: list[Chunk]) -> list[Chunk]:
        """Generate one doc-level summary and populate chunk.doc_summary on every chunk.

        A single complete() call is made per invocation. If the injected client
        implements local caching (T15), identical prompts across invocations will
        be served from cache without hitting the API.
        """
        doc_excerpt = normalize_text(doc.html)[:_DOC_EXCERPT_CHARS]
        summary_user_prompt = (
            "Here is the document this chunk belongs to:\n"
            "<doc>\n"
            f"Title: {doc.title}\n\n"
            f"{doc_excerpt}\n"
            "</doc>\n\n"
            "Provide a 1–2 sentence summary that situates this document for "
            "retrieval. Do not paraphrase content; describe context only."
        )
        messages: list[dict[str, object]] = [{"role": "user", "content": summary_user_prompt}]
        response = self._client.complete(
            model=self._model,
            system=None,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )
        summary = response.text
        for chunk in chunks:
            chunk.doc_summary = summary
        return chunks
