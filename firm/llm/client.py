"""AnthropicClient Protocol and CompletionResponse model.

T15 will add a concrete ``CachedAnthropicClient`` class here that:
- wraps the Anthropic SDK,
- caches responses in the ``llm_cache`` SQLite table (keyed by prompt hash),
- sets ``cache_hit=True`` on CompletionResponse when served from local cache.

For T9, only the Protocol and response model are defined so that the contextual
augmenter and tests can depend on a stable interface without the real SDK client.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class CompletionResponse(BaseModel):
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_hit: bool = False  # True if served from local cache (T15)


@runtime_checkable
class AnthropicClient(Protocol):
    def complete(
        self,
        *,
        model: str,
        system: str | None,
        messages: Sequence[dict[str, object]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> CompletionResponse: ...
