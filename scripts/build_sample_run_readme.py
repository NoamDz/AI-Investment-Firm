"""Generate a per-date ``README.md`` for a ``sample_runs/<date>/`` directory.

Reads ``decisions.jsonl`` and (optionally) ``trace.jsonl`` from
``sample_runs/<date>/`` and writes ``sample_runs/<date>/README.md`` in the
layout spec'd by ``docs/PLAN_reports_overhaul.md`` §5 / §6 T5.

Determinism: no wall-clock reads, no randomness. Running twice produces
byte-identical output. UTF-8, ``\n`` line endings, trailing newline.

CLI:
    python scripts/build_sample_run_readme.py --date YYYY-MM-DD \
        [--sample-runs-root sample_runs]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Heuristic regime labels chosen to match the existing committed fixture
# content. Easy to extend: add another (date -> label) row when new
# sample-run dates are committed under ``sample_runs/``.
_REGIME_BY_DATE: dict[str, str] = {
    "2024-03-13": "earnings_heavy",  # AAPL earnings window in fixture data
    "2024-08-07": "vol_spike",  # Aug VIX spike
    "2023-11-08": "pre_news",  # pre-CPI day
}

_RATIONALE_MAX = 80


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts. Skips blank lines."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _truncate_rationale(text: str) -> str:
    """Truncate rationale to _RATIONALE_MAX chars, append '...' if longer.

    Also escapes any pipe characters so the surrounding markdown table stays
    valid.
    """
    if text is None:
        text = ""
    if len(text) > _RATIONALE_MAX:
        text = text[:_RATIONALE_MAX] + "…"
    return text.replace("|", r"\|")


def _setup_sentence(decisions: list[dict[str, Any]], regime: str) -> str:
    """Build the one-line 'Setup' description from the first decision."""
    if not decisions:
        return "No decisions recorded."
    first = decisions[0].get("research_decision", {}) or {}
    action = first.get("action", "")
    payload = first.get("payload", {}) or {}
    if action == "BUY":
        ticker = payload.get("ticker", "")
        shares = payload.get("shares", "")
        return f"Opens with BUY {ticker} x {shares}; {regime} regime."
    return f"Opens flat; {regime} regime - no immediate trade."


def _what_to_look_for(decisions: list[dict[str, Any]]) -> str:
    """Pick the single most interesting decision to point the reader at."""
    if not decisions:
        return "Watch the decision table for action variety."
    # 1. ESCALATE wins.
    for d in decisions:
        rd = d.get("research_decision", {}) or {}
        if rd.get("action") == "ESCALATE":
            return (
                f"ESCALATE row triggered by `{rd.get('failure_mode') or 'policy_gate'}` "
                f"- see decision `{rd.get('id', '')}`."
            )
    # 2. First non-null failure_mode.
    for d in decisions:
        rd = d.get("research_decision", {}) or {}
        fm = rd.get("failure_mode")
        if fm:
            return f"Failure mode `{fm}` flagged on decision `{rd.get('id', '')}`."
    # 3. Highest-confidence BUY.
    buys = [
        d for d in decisions
        if (d.get("research_decision", {}) or {}).get("action") == "BUY"
    ]
    if buys:
        # Stable sort: descending confidence, tiebreak earliest ts.
        buys_sorted = sorted(
            buys,
            key=lambda d: (
                -float((d.get("research_decision", {}) or {}).get("confidence", 0.0)),
                d.get("ts", ""),
            ),
        )
        rd = buys_sorted[0].get("research_decision", {}) or {}
        return (
            f"Highest-confidence BUY is `{rd.get('id', '')}` "
            f"({float(rd.get('confidence', 0.0)):.2f})."
        )
    return "Watch the decision table for action variety."


def _citation_counts_by_decision(spans: list[dict[str, Any]]) -> dict[str, int]:
    """Sum the ``citations`` counter from ``llm.call`` spans, grouped by decision_id.

    Only ``llm.call`` spans count — other spans (research/pm/risk) double-count
    the same citation set because they wrap an underlying llm.call. Keeping
    this in lockstep with ``scripts/hydrate_sample_db.py`` so the per-decision
    counts in this README match what the HTML report shows.
    """
    totals: dict[str, int] = {}
    for span in spans:
        if span.get("operation") != "llm.call":
            continue
        did = span.get("decision_id") or ""
        if not did:
            continue
        c = span.get("citations") or 0
        try:
            c_int = int(c)
        except (TypeError, ValueError):
            c_int = 0
        totals[did] = totals.get(did, 0) + c_int
    return totals


def _pick_walk_decision(
    decisions: list[dict[str, Any]],
    citation_totals: dict[str, int],
) -> str | None:
    """Pick decision_id with highest total citations; tiebreak by earliest ts."""
    if not decisions:
        return None
    # Build (decision_id, ts) pairs preserving order from decisions.jsonl.
    pairs = []
    for d in decisions:
        rd = d.get("research_decision", {}) or {}
        did = rd.get("id")
        if not did:
            continue
        pairs.append((did, d.get("ts", "")))
    if not pairs:
        return None
    # Sort by (-citations, ts ascending). Stable sort preserves original order
    # for additional determinism.
    pairs_sorted = sorted(
        pairs,
        key=lambda t: (-citation_totals.get(t[0], 0), t[1]),
    )
    return pairs_sorted[0][0]


def _decision_table_rows(
    decisions: list[dict[str, Any]],
    citation_totals: dict[str, int],
) -> list[str]:
    """Render one markdown table row per decision."""
    rows: list[str] = []
    for d in decisions:
        rd = d.get("research_decision", {}) or {}
        payload = rd.get("payload", {}) or {}
        action = rd.get("action", "") or ""
        ts = d.get("ts", "") or ""
        if action in ("BUY", "SELL"):
            ticker = payload.get("ticker", "") or ""
            shares = payload.get("shares", "") or ""
        else:
            ticker = ""
            shares = ""
        conf_val = rd.get("confidence")
        try:
            conf = f"{float(conf_val):.2f}"
        except (TypeError, ValueError):
            conf = ""
        # Citations: prefer the trace-derived total; fall back to a top-level
        # ``citations`` field on the decision; else 0.
        did = rd.get("id") or ""
        if did in citation_totals:
            citations = citation_totals[did]
        else:
            raw = rd.get("citations")
            if isinstance(raw, list):
                citations = len(raw)
            elif isinstance(raw, int):
                citations = raw
            else:
                citations = 0
        failure_mode = rd.get("failure_mode") or ""
        rationale = _truncate_rationale(rd.get("rationale", "") or "")
        rows.append(
            f"| {ts} | {action} | {ticker} | {shares} | {conf} "
            f"| {citations} | {failure_mode} | {rationale} |"
        )
    return rows


def _walk_trade_block(
    decision_id: str | None,
    date: str,
    spans: list[dict[str, Any]],
    raw_trace_lines: list[str],
) -> list[str]:
    """Build the 'Walking one trade' section as a list of lines (no trailing \\n)."""
    lines: list[str] = ["## Walking one trade", ""]
    if not spans or decision_id is None:
        lines.append("_(no trace data available for this run)_")
        return lines

    lines.append(
        f"The most-cited decision this day is `{decision_id}`. "
        "Reproduce its full chain with:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append(
        f'grep \'"decision_id":"{decision_id}"\' '
        f"sample_runs/{date}/trace.jsonl | jq ."
    )
    lines.append("```")
    lines.append("")
    lines.append("```jsonl")
    # Iterate raw lines in file order, keep ones matching decision_id, cap 10.
    kept = 0
    for raw in raw_trace_lines:
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if obj.get("decision_id") != decision_id:
            continue
        # Emit verbatim (rstrip just in case of \r\n).
        lines.append(raw.rstrip("\r\n"))
        kept += 1
        if kept >= 10:
            break
    lines.append("```")
    return lines


def _render_readme(date: str, sample_runs_root: Path) -> str:
    date_dir = sample_runs_root / date
    decisions_path = date_dir / "decisions.jsonl"
    trace_path = date_dir / "trace.jsonl"

    if not decisions_path.exists():
        raise FileNotFoundError(f"missing decisions file: {decisions_path}")

    decisions = _load_jsonl(decisions_path)

    if trace_path.exists():
        spans = _load_jsonl(trace_path)
        # Re-read the raw lines (preserves file order + exact JSON formatting).
        with trace_path.open("r", encoding="utf-8") as f:
            raw_trace_lines = f.readlines()
    else:
        spans = []
        raw_trace_lines = []

    regime = _REGIME_BY_DATE.get(date, "unknown")
    setup = _setup_sentence(decisions, regime)
    look_for = _what_to_look_for(decisions)

    citation_totals = _citation_counts_by_decision(spans)
    walk_decision_id = _pick_walk_decision(decisions, citation_totals)
    table_rows = _decision_table_rows(decisions, citation_totals)

    lines: list[str] = []
    lines.append(f"# Sample run - {date}")
    lines.append("")
    lines.append("## What this day demonstrates")
    lines.append("")
    lines.append(f"- **Regime:** {regime}")
    lines.append(f"- **Setup:** {setup}")
    lines.append(f"- **What to look for:** {look_for}")
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    lines.append(
        "| ts | action | ticker | shares | conf | citations | failure_mode | rationale |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    if table_rows:
        lines.extend(table_rows)
    lines.append("")
    lines.extend(_walk_trade_block(walk_decision_id, date, spans, raw_trace_lines))
    lines.append("")
    lines.append("## Bundle")
    lines.append("")
    lines.append("- [`daily_report.md`](daily_report.md) - legacy plain-text summary")
    lines.append(
        "- [`daily_report.html`](daily_report.html) - rendered report (open in browser)"
    )
    lines.append(
        "- [`positions.xlsx`](positions.xlsx) - Positions / P&L / Decisions sheets"
    )
    lines.append("- [`decisions.jsonl`](decisions.jsonl) - raw decisions")
    lines.append("- [`trace.jsonl`](trace.jsonl) - raw spans")
    dashboard_png = (sample_runs_root / date / "dashboard.png")
    if dashboard_png.exists():
        lines.append(
            "- [`dashboard.png`](dashboard.png) - Tab 1 (Today's Report) screenshot"
        )
    else:
        lines.append(
            "- `dashboard.png` - _not committed; run "
            "`scripts/capture_dashboard_png.py` after "
            "`pip install playwright && playwright install chromium` to generate_"
        )

    # Trailing newline.
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate sample_runs/<date>/README.md from per-date "
            "decisions.jsonl and trace.jsonl."
        )
    )
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--sample-runs-root",
        default="sample_runs",
        help="Root of sample_runs/ (defaults to ./sample_runs).",
    )
    args = parser.parse_args(argv)

    root = Path(args.sample_runs_root)
    date_dir = root / args.date
    if not date_dir.is_dir():
        print(f"error: {date_dir} is not a directory", file=sys.stderr)
        return 2

    try:
        content = _render_readme(args.date, root)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out_path = date_dir / "README.md"
    out_path.write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
