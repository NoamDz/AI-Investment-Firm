"""firm.obs — OpenTelemetry observability spine (spec §10.1)."""
from __future__ import annotations

from firm.obs.spans import (
    agent_span,
    llm_span,
    retrieval_span,
    stamp_decision,
    tool_span,
)

__all__ = [
    "agent_span",
    "llm_span",
    "retrieval_span",
    "stamp_decision",
    "tool_span",
]
