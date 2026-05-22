"""Reporter — minimal JSONL summary for Plan 1. Markdown+XLSX in Plan 3."""
from __future__ import annotations

import json
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from opentelemetry import trace

from firm.core.clock import Clock
from firm.core.models import Decision
from firm.db.connection import get_conn
from firm.obs import agent_span
from firm.orchestrator.state import WorkingState


def _json_default(obj: Any) -> Any:
    """``json.dumps`` fallback for non-primitive types in WorkingState values.

    LangGraph state may contain plain datetime/date objects nested inside
    ``model_dump()`` output (which does not stringify datetimes by default).
    Convert them to ISO 8601 strings so the JSONL stays serialisable; other
    unknown types fall back to ``repr()`` rather than raising.
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return repr(obj)


def _persist_decisions_from_state(state: WorkingState | dict[str, Any], db_path: Path, clock: Clock) -> None:
    """Persist all top-level Decision values in `state` to the decisions table.

    Plan 1 assumption: WorkingState stores Decisions as scalar top-level values
    (research_decision, pm_decision, risk_decision). If Plan 2/3 introduces
    nested Decisions (e.g., inside metadata or lists), revisit this scan.
    """
    decisions: list[Decision] = [v for v in state.values() if isinstance(v, Decision)]
    if not decisions:
        return
    with closing(get_conn(db_path)) as conn:
        for d in decisions:
            conn.execute(
                "INSERT OR IGNORE INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    d.id, json.dumps(d.decision_id_chain), d.action.value,
                    d.payload.model_dump_json(), d.rationale, d.confidence,
                    json.dumps([c.model_dump(mode="json") for c in d.citations]),
                    d.falsification_condition, d.escalation_reason,
                    d.failure_mode.value if d.failure_mode else None,
                    json.dumps(d.metadata), d.nonce, clock.now().isoformat(),
                ),
            )


def _cost_today_usd(db_path: Path, clock: Clock) -> float:
    """Return total cost_usd from cost_ledger for today (UTC).

    Uses ``clock.now()`` for determinism in tests — no wall-clock dependency.
    Returns 0.0 when the table is empty or no rows exist for today.
    """
    today_utc = clock.now().astimezone(timezone.utc).strftime("%Y-%m-%d")
    midnight_iso = f"{today_utc}T00:00:00+00:00"
    with closing(get_conn(db_path)) as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM cost_ledger WHERE created_at >= ?",
            (midnight_iso,),
        ).fetchone()
    return float(row[0]) if row else 0.0


def _serialize_value(v: Any) -> Any:
    """Serialize a state value for JSONL output.

    Pydantic models are dumped to structured dicts (mode='json') so that
    the JSONL is round-trippable. Other values are left as-is; json.dumps
    will handle primitives and will raise on any remaining non-serializable
    objects rather than silently calling str() on them.
    """
    if hasattr(v, "model_dump"):
        return v.model_dump(mode="json")
    return v


def make_reporter(
    *, reports_root: Path, clock: Clock, db_path: Path | None = None
) -> Callable[[WorkingState], dict[str, Any]]:
    def reporter(state: WorkingState) -> dict[str, Any]:
        # Wrap in ``agent.reporter`` so (a) the per-heartbeat span trail is
        # complete and (b) ``get_current_span()`` below reports the trace_id
        # we want to embed in the JSONL row.
        with agent_span("reporter"):
            now = clock.now()
            date_dir = reports_root / now.strftime("%Y-%m-%d")
            date_dir.mkdir(parents=True, exist_ok=True)
            path = date_dir / "decisions.jsonl"
            payload: dict[str, Any] = {"ts": now.isoformat()}
            for k, v in state.items():
                payload[k] = _serialize_value(v)
            # T03 trace pointer: stamp the current OTel trace_id onto each
            # row so an operator reading ``decisions.jsonl`` can ``jq`` against
            # the matching ``traces/<date>/run-<run_id>.jsonl`` to recover the
            # full span tree.  ``get_current_span()`` always returns a Span
            # (INVALID_SPAN sentinel when no provider/span is active), so the
            # only thing to check is ``trace_id != 0`` (0 == INVALID_SPAN).
            # Falls back to "" so the JSONL schema stays stable when nothing
            # is active (shouldn't happen in production).
            ctx = trace.get_current_span().get_span_context()
            if ctx.trace_id:
                payload["trace_id"] = format(ctx.trace_id, "032x")
            else:
                payload["trace_id"] = ""
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=_json_default) + "\n")
            if db_path is not None:
                _persist_decisions_from_state(state, db_path, clock)
            # T26: capture cost inside the span so the DB read is traced.
            cost = _cost_today_usd(db_path, clock) if db_path is not None else 0.0
        # Print after the span closes so stdout is not captured by the tracer.
        print(f"Cost so far today: ${cost:.3f}")
        return {"report_path": str(path)}
    return reporter
