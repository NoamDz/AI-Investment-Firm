"""LangGraph shared state. See spec §3.1, §4.1."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages

from firm.core.models import Claim, Decision
from firm.rag.chunk import Chunk


class WorkingState(TypedDict, total=False):
    """State flowing through the LangGraph workflow.

    Each agent reads upstream Decision(s) from this state and appends its own.
    Decisions chain via Decision.decision_id_chain.
    """
    heartbeat_at: str                  # ISO 8601
    research_decision: Decision
    pm_decision: Decision
    risk_decision: Decision
    # Plan 2 §T19: research agent surfaces retrieved chunks + extracted claims
    # on the working state so downstream nodes (sufficiency, PM) can read them.
    retrieved_chunks: list[Chunk]
    claims: list[Claim]
    hitl_required: bool
    hitl_approved: bool | None
    execution_result: dict[str, Any]   # OrderResult-as-dict
    report_path: str
    notes: Annotated[list[str], add_messages]
