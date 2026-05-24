"""Daily HTML report renderer. See Plan 4 §T1 / PLAN_reports_overhaul.md §4.

Mirrors ``firm.reports.daily.render_daily_report`` but emits a self-contained
HTML document. Determinism is critical: this file is part of the eval-pipeline
diff in ``firm/ops/check_reports_clean.sh``, so the writer never embeds
wall-clock timestamps and all iteration is in a stable order.
"""
from __future__ import annotations

import json
from contextlib import closing
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from firm.broker.protocol import Broker
from firm.db.connection import get_conn
from firm.reports.daily import _label_sort_key, _model_label

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Same palette as firm.dashboard.ACTION_COLORS — duplicated rather than imported
# to avoid pulling Streamlit (a dashboard-only dep) into the report writer.
_ACTION_COLORS = {
    "BUY": "#1b8a4d",
    "SELL": "#c0392b",
    "HOLD": "#7f8c8d",
    "REFUSE": "#8e44ad",
    "ESCALATE": "#e67e22",
}

# Actions whose payload carries (ticker, shares); mirror xlsx.py._TRADE_ACTIONS.
_TRADE_ACTIONS = {"BUY", "SELL"}

# Display order for the executive-summary action histogram.
_ACTION_ORDER = ["BUY", "SELL", "HOLD", "REFUSE", "ESCALATE"]

_RATIONALE_MAX = 120


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _truncate_rationale(rationale: str) -> str:
    """Truncate to _RATIONALE_MAX chars, appending an ellipsis when cut."""
    if rationale is None:
        return ""
    if len(rationale) <= _RATIONALE_MAX:
        return rationale
    return rationale[:_RATIONALE_MAX] + "…"  # … (U+2026)


def _trade_fields(action: str, payload_json: str) -> tuple[str, str]:
    """Return ('ticker', 'shares') string pair for trade actions; ('','') else."""
    if action.upper() not in _TRADE_ACTIONS:
        return "", ""
    try:
        payload = json.loads(payload_json) if payload_json else {}
        ticker = payload.get("ticker") or ""
        shares_raw = payload.get("shares")
        shares = "" if shares_raw is None or shares_raw == "" else str(shares_raw)
        return str(ticker), shares
    except (json.JSONDecodeError, ValueError):
        return "", ""


def _citation_count(citations_json: str | None) -> int:
    """JSON-array length, or 0 for empty/null."""
    if not citations_json:
        return 0
    try:
        parsed = json.loads(citations_json)
        return len(parsed) if isinstance(parsed, list) else 0
    except (json.JSONDecodeError, ValueError):
        return 0


def _build_decision_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    """Project decision DB rows into template-friendly dicts (ordered)."""
    out: list[dict[str, str]] = []
    for r in rows:
        action = str(r["action"])
        ticker, shares = _trade_fields(action, str(r.get("payload") or ""))
        out.append(
            {
                "ts": str(r["created_at"]),
                "action": action,
                "color": _ACTION_COLORS.get(action, "#444"),
                "ticker": ticker,
                "shares": shares,
                "confidence": f"{float(r['confidence']):.2f}",
                "citations": str(_citation_count(r.get("citations"))),  # type: ignore[arg-type]
                "failure_mode": str(r["failure_mode"] or ""),
                "rationale": _truncate_rationale(str(r.get("rationale") or "")),
            }
        )
    return out


def _build_cost_groups(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, str]], int, str]:
    """Return (groups, cached_pct, total_cost_str) ready for the template."""
    if not rows:
        return [], 0, "0.000"

    groups: dict[str, list[float]] = {}
    total_cost = 0.0
    cached_count = 0
    for row in rows:
        label = _model_label(str(row["model"]))
        cost = float(row["cost_usd"])  # type: ignore[arg-type]
        groups.setdefault(label, []).append(cost)
        total_cost += cost
        if row["cached_tokens"] is not None:
            cached_count += 1

    cached_pct = round(cached_count / len(rows) * 100)

    sorted_labels = sorted(groups, key=_label_sort_key)
    out: list[dict[str, str]] = []
    for label in sorted_labels:
        costs = groups[label]
        count = len(costs)
        avg = sum(costs) / count
        out.append(
            {
                "label": label,
                "calls": str(count),
                "avg": f"{avg:.4f}",
                "total": f"{sum(costs):.3f}",
            }
        )
    return out, cached_pct, f"{total_cost:.3f}"


def _build_summary_text(
    date_str: str,
    decisions: list[dict[str, object]],
    cost_groups: list[dict[str, str]],
    cached_pct: int,
    total_cost: str,
    total_calls: int,
) -> str:
    """One-paragraph executive summary derived from counts only."""
    if not decisions:
        return f"No decisions recorded for {date_str}."

    counts = {a: 0 for a in _ACTION_ORDER}
    for d in decisions:
        action = str(d["action"])
        if action in counts:
            counts[action] += 1

    parts = [f"{counts[a]} {a}" for a in _ACTION_ORDER if counts[a] > 0]
    histogram = ", ".join(parts)
    total = len(decisions)
    spend_clause = (
        f" Total LLM spend ${total_cost} across {total_calls} calls "
        f"({cached_pct}% cache hit)."
        if cost_groups
        else ""
    )
    return (
        f"On {date_str} the firm took {total} decision{'s' if total != 1 else ''}: "
        f"{histogram}.{spend_clause}"
    )


def _build_trace_grep_cmds(decision_rows: list[dict[str, object]]) -> list[str]:
    """Produce one ``grep ... | jq .name`` line per BUY/SELL decision_id.

    Ordering follows the input rows (already ``created_at ASC``); using the same
    order as the table guarantees byte-stable output across runs.
    """
    cmds: list[str] = []
    for r in decision_rows:
        action = str(r["action"]).upper()
        if action not in _TRADE_ACTIONS:
            continue
        decision_id = str(r["id"])
        cmds.append(
            f"grep '\"decision_id\":\"{decision_id}\"' trace.jsonl | jq .name"
        )
    return cmds


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_daily_html(
    *,
    date: date,
    db_path: Path,
    broker: Broker,  # noqa: ARG001 — reserved for symmetry with daily.py / future use
    traces_path: Path,  # noqa: ARG001 — reserved; not used in T1
    reports_root: Path,
    reconcile_block: str,
) -> Path:
    """Render daily_report.html and return the path written.

    Sections (top-to-bottom): header bar, executive summary, decisions table,
    cost summary, reconciliation (pre-formatted block), trace pointer footer.
    """
    date_str = date.isoformat()
    date_prefix = date_str

    with closing(get_conn(db_path)) as conn:
        decision_db_rows = conn.execute(
            "SELECT id, created_at, action, payload, confidence, citations, "
            "failure_mode, rationale "
            "FROM decisions WHERE created_at LIKE ? ORDER BY created_at ASC",
            (f"{date_prefix}%",),
        ).fetchall()

        cost_db_rows = conn.execute(
            "SELECT model, cached_tokens, cost_usd "
            "FROM cost_ledger WHERE created_at LIKE ?",
            (f"{date_prefix}%",),
        ).fetchall()

    decisions_dicts: list[dict[str, object]] = [dict(r) for r in decision_db_rows]
    costs_dicts: list[dict[str, object]] = [dict(r) for r in cost_db_rows]

    decision_rows = _build_decision_rows(decisions_dicts)
    cost_groups, cached_pct, total_cost = _build_cost_groups(costs_dicts)
    summary_text = _build_summary_text(
        date_str=date_str,
        decisions=decisions_dicts,
        cost_groups=cost_groups,
        cached_pct=cached_pct,
        total_cost=total_cost,
        total_calls=len(costs_dicts),
    )
    trace_grep_cmds = _build_trace_grep_cmds(decisions_dicts)

    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=True,  # rationale/failure_mode must be HTML-escaped
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("daily_report.html.j2")

    content = tmpl.render(
        date_str=date_str,
        summary_text=summary_text,
        decision_rows=decision_rows,
        cost_groups=cost_groups,
        cached_pct=cached_pct,
        total_cost=total_cost,
        reconcile_block=reconcile_block,
        trace_grep_cmds=trace_grep_cmds,
    )

    out_dir = reports_root / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "daily_report.html"
    out_path.write_text(content, encoding="utf-8", newline="\n")
    return out_path
