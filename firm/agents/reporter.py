"""Reporter — minimal JSONL summary for Plan 1. Markdown+XLSX in Plan 3."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Any, Callable

from firm.core.clock import Clock
from firm.core.models import Decision
from firm.db.connection import get_conn
from firm.orchestrator.state import WorkingState


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


def make_reporter(
    *, reports_root: Path, clock: Clock, db_path: Path | None = None
) -> Callable[[WorkingState], dict[str, Any]]:
    def reporter(state: WorkingState) -> dict[str, Any]:
        now = clock.now()
        date_dir = reports_root / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        path = date_dir / "decisions.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now.isoformat(), **state}, default=str) + "\n")
        if db_path is not None:
            _persist_decisions_from_state(state, db_path, clock)
        return {"report_path": str(path)}
    return reporter
