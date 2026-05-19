"""FilingDoc source model. See design spec §5.2. CorpusSource Protocol added by T6."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class FilingDoc(BaseModel):
    doc_id: str
    ticker: str
    filing_type: str
    published_at: datetime
    title: str
    html: str
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_tz_aware_published_at(self) -> "FilingDoc":
        if self.published_at.tzinfo is None:
            raise ValueError("published_at must be timezone-aware")
        return self
