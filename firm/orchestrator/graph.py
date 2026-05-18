"""LangGraph topology. See spec §3.1."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from firm.core.models import ActionEnum
from firm.orchestrator.state import WorkingState


class NodeCallable(Protocol):
    """Protocol for LangGraph node callables operating on WorkingState."""

    def __call__(self, state: WorkingState, /) -> dict[str, Any]: ...


def build_graph(
    *,
    db_path: Path,
    monitor_node: NodeCallable,
    research_node: NodeCallable,
    pm_node: NodeCallable,
    risk_node: NodeCallable,
    hitl_node: NodeCallable,
    execution_node: NodeCallable,
    reporter_node: NodeCallable,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Compose the firm's workflow.

    Edges: monitor → research → pm → risk → (hitl|execution) → reporter
    Conditional after risk: if hitl_required → hitl → execution; else → execution.
    """
    g = StateGraph(WorkingState)
    # LangGraph stubs constrain NodeInputT to TypedDictLike/Dataclass/BaseModel via TypeVar
    # inference from concrete callables; passing a Protocol type alias trips the overload
    # resolver even though NodeCallable is structurally identical to _Node[WorkingState].
    g.add_node("monitor", monitor_node)  # type: ignore[call-overload]
    g.add_node("research", research_node)  # type: ignore[call-overload]
    g.add_node("pm", pm_node)  # type: ignore[call-overload]
    g.add_node("risk", risk_node)  # type: ignore[call-overload]
    g.add_node("hitl", hitl_node)  # type: ignore[call-overload]
    g.add_node("execution", execution_node)  # type: ignore[call-overload]
    g.add_node("reporter", reporter_node)  # type: ignore[call-overload]

    g.set_entry_point("monitor")
    g.add_edge("monitor", "research")
    g.add_edge("research", "pm")
    g.add_edge("pm", "risk")

    def route_after_risk(state: WorkingState) -> str:
        decision = state.get("risk_decision")
        if decision is None:
            # Risk node did not produce a decision (e.g., crashed mid-run). Skip HITL and
            # let execution short-circuit on the missing/non-actionable decision.
            return "execution"
        return "hitl" if decision.action == ActionEnum.ESCALATE else "execution"

    g.add_conditional_edges("risk", route_after_risk, {"hitl": "hitl", "execution": "execution"})
    g.add_edge("hitl", "execution")
    g.add_edge("execution", "reporter")
    g.add_edge("reporter", END)

    import sqlite3
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    return g.compile(checkpointer=saver, interrupt_before=["hitl"])
