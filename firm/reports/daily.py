"""Daily Markdown report renderer. See Plan 3 §T16."""
from __future__ import annotations

from contextlib import closing
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from firm.broker.protocol import Broker
from firm.db.connection import get_conn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed action display order (spec §T16 decision summary).
_ACTION_ORDER = ["BUY", "SELL", "HOLD", "REFUSE", "ESCALATE"]

# Display label priority for model name grouping — checked via `in model.lower()`.
_MODEL_LABEL_ORDER = ["haiku", "sonnet", "opus"]
_MODEL_DISPLAY = {"haiku": "Haiku", "sonnet": "Sonnet", "opus": "Opus"}

# Width for label+colon field so counts align at the same column (see spec §10.2).
# "Sonnet:" is the widest standard label at 7 chars; total before count = 9.
_COST_LABEL_COL = 9  # label+colon is left-justified in this field; count right in 2


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _model_label(model: str) -> str:
    """Derive display label from raw model string.

    Haiku → "Haiku", Sonnet → "Sonnet", Opus → "Opus", else raw model string.
    """
    lower = model.lower()
    for key in _MODEL_LABEL_ORDER:
        if key in lower:
            return _MODEL_DISPLAY[key]
    return model


def _label_sort_key(label: str) -> tuple[int, str]:
    """Sort key: Haiku < Sonnet < Opus < (others lex)."""
    ordered = [_MODEL_DISPLAY[k] for k in _MODEL_LABEL_ORDER]
    try:
        return (ordered.index(label), "")
    except ValueError:
        return (len(ordered), label)


def _render_decision_block(date_str: str, rows: list[dict[str, object]]) -> str:
    """Build the DECISION SUMMARY section as a plain-text block."""
    # Count actions
    action_counts: dict[str, int] = {a: 0 for a in _ACTION_ORDER}
    failure_mode_counts: dict[str, int] = {}

    for row in rows:
        action = str(row["action"])
        if action in action_counts:
            action_counts[action] += 1
        fm = row["failure_mode"]
        if fm is not None:
            fm_str = str(fm)
            failure_mode_counts[fm_str] = failure_mode_counts.get(fm_str, 0) + 1

    total = len(rows)
    lines: list[str] = [
        f"DECISION SUMMARY ({date_str})",
        f"Total: {total}",
    ]

    # Histogram: all 5 actions, zero-count included.
    # Label "ESCALATE:" is 9 chars — widest; count right-aligned in 2.
    # Format: "  {action}:{spaces}{count:>2}"
    # Widest action name: "ESCALATE" (8) + ":" = 9. Field width = 9+1 = 10 before count.
    # We pad action+colon to 10 chars left-justified, then count right in 2.
    for action in _ACTION_ORDER:
        label_colon = f"{action}:"
        # Pad label_colon to 10 so count lands at same column for all 5 actions.
        # "BUY:" = 4, "SELL:" = 5, "HOLD:" = 5, "REFUSE:" = 7, "ESCALATE:" = 9
        lines.append(f"  {label_colon:<10}{action_counts[action]:>2}")

    # failure_mode breakdown: only observed modes, sorted for determinism.
    for fm in sorted(failure_mode_counts):
        lines.append(f"  {fm}: {failure_mode_counts[fm]}")

    return "\n".join(lines)


def _render_cost_block(date_str: str, rows: list[dict[str, object]]) -> str:
    """Build the COST SUMMARY section as a plain-text block (spec §10.2)."""
    if not rows:
        return f"COST SUMMARY ({date_str})\n  (no LLM calls recorded)"

    # Group by display label.
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

    total_rows = len(rows)
    cached_pct = round(cached_count / total_rows * 100)

    # Sort labels: Haiku, Sonnet, Opus, then others lex.
    sorted_labels = sorted(groups, key=_label_sort_key)

    lines: list[str] = [f"COST SUMMARY ({date_str})"]
    for label in sorted_labels:
        costs = groups[label]
        count = len(costs)
        avg = sum(costs) / count
        total_g = sum(costs)
        label_colon = f"{label}:"
        # Align count at column _COST_LABEL_COL (label+colon left-pad to 9), count right in 2.
        lines.append(
            f"  {label_colon:<{_COST_LABEL_COL}}{count:>2} × avg ${avg:.4f} = ${total_g:.3f}"
        )

    lines.append(f"  Cached:  {cached_pct}%")
    lines.append(f"  Total:   ${total_cost:.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_daily_report(
    *,
    date: date,
    db_path: Path,
    broker: Broker,  # reserved for T17 (reconcile) and T18 (CLI symmetry); not used in T16 itself
    traces_path: Path,  # reserved for T18 wiring (CLI passes through); not rendered in T16
    reports_root: Path,
    reconcile_block: str,
) -> Path:
    """Render daily_report.md and return the path written.

    Sections:
      1. Decision summary (action histogram + failure_mode breakdown)
      2. Cost summary (grouped by model, spec §10.2 format)
      3. Reconciliation block (pre-rendered string from T17)
    """
    date_str = date.isoformat()  # YYYY-MM-DD
    date_prefix = date_str  # used for LIKE 'YYYY-MM-DD%' filter

    with closing(get_conn(db_path)) as conn:
        # Pull decisions for the day.
        # Use Python-side date prefix filter (not SQL date()) for timezone-stable behaviour —
        # created_at is stored as ISO 8601 text; SQL date() strips the timezone offset,
        # which can shift rows across midnight depending on server locale.
        decision_rows = conn.execute(
            "SELECT action, failure_mode FROM decisions WHERE created_at LIKE ?",
            (f"{date_prefix}%",),
        ).fetchall()

        cost_rows = conn.execute(
            "SELECT model, cached_tokens, cost_usd FROM cost_ledger WHERE created_at LIKE ?",
            (f"{date_prefix}%",),
        ).fetchall()

    # Convert sqlite3.Row to plain dicts for helper functions.
    decisions: list[dict[str, object]] = [dict(r) for r in decision_rows]
    costs: list[dict[str, object]] = [dict(r) for r in cost_rows]

    decision_block = _render_decision_block(date_str, decisions)
    cost_block = _render_cost_block(date_str, costs)

    # Use FileSystemLoader so we don't need an __init__.py in the templates dir,
    # and the loader works regardless of how the package is installed.
    templates_dir = Path(__file__).parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,  # output is plain Markdown, not HTML
        keep_trailing_newline=True,
    )
    tmpl = env.get_template("md_template.j2")

    content = tmpl.render(
        decision_block=decision_block,
        cost_block=cost_block,
        reconcile_block=reconcile_block,
    )

    out_dir = reports_root / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "daily_report.md"
    # Write with explicit UTF-8 + Unix line endings to keep output portable
    # and handle × (U+00D7) and ✓ (U+2713) safely on Windows.
    out_path.write_text(content, encoding="utf-8", newline="\n")
    return out_path
