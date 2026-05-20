"""FilingDoc source model and CorpusSource protocol. See design spec §5.2."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator


class FilingDoc(BaseModel):
    doc_id: str
    ticker: str
    filing_type: str        # 10-K | 10-Q | 8-K
    published_at: datetime  # tz-aware, REQUIRED
    title: str
    html: str
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_tz_aware_published_at(self) -> "FilingDoc":
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")
        return self


@runtime_checkable
class CorpusSource(Protocol):
    name: str

    def iter_docs(self) -> Iterator[FilingDoc]: ...
