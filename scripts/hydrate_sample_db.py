"""Hydrate a temporary ``firm.db`` from a committed sample-run bundle.

A one-shot helper used by the ``T6`` regeneration workflow (see
``docs/PLAN_reports_overhaul.md`` §5 / §6 T6). Reads
``sample_runs/<date>/decisions.jsonl`` + ``trace.jsonl`` and writes the
``decisions`` + ``cost_ledger`` (+ ``positions``) rows the downstream
``firm.cli report --date <date>`` command needs to render the HTML / XLSX
bundle.

The committed JSONL files were captured before Plan 4's per-decision
``citations`` JSON was added; this script writes a placeholder ``citations``
JSON array of the correct length (one stub object per source citation
counted from matching ``llm.call`` spans) so the HTML report and the
per-date README agree on per-decision citation counts.

CLI:
    python scripts/hydrate_sample_db.py \\
        --sample-runs-root sample_runs --date YYYY-MM-DD --out PATH/TO/firm.db
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any

# Local imports — keep at module-top so ``python scripts/hydrate_sample_db.py``
# fails fast if the firm package isn't importable rather than at first-use.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from firm.db.connection import get_conn  # noqa: E402
from firm.db.migrations import init_db  # noqa: E402


# Static fixture data for the "positions at end of day" tables. The committed
# sample-run daily_report.md files document the broker view; we replicate it
# here so the rendered HTML + XLSX show a non-empty Positions sheet.
#
# Keys:  date string (YYYY-MM-DD).
# Value: list of (ticker, shares_str, avg_cost_str). avg_cost defaults to
# "0.0" when not derivable from the committed BUY rationale — the Positions
# sheet still renders, just with $0 unrealized PnL.
_POSITIONS_BY_DATE: dict[str, list[tuple[str, str, str]]] = {
    "2024-03-13": [("AAPL", "100", "0.0")],
    "2024-08-07": [("AAPL", "60", "0.0")],
    "2023-11-08": [("AAPL", "100", "0.0"), ("MSFT", "20", "0.0")],
}

# Starting cash for the hydrated broker view. Matches FakeBroker default so
# reconcile_on_boot sees a consistent picture.
_DEFAULT_CASH = Decimal("100000")


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


def _citation_counts_by_decision(spans: list[dict[str, Any]]) -> dict[str, int]:
    """Sum the ``citations`` counter from ``llm.call`` spans, grouped by decision_id.

    Mirrors the same computation done by ``scripts/build_sample_run_readme.py``
    so the per-decision citation count rendered in the HTML report
    (which reads ``decisions.citations`` JSON length) matches what the
    README's decisions table shows.
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


def _insert_decisions(
    conn: Any,
    decisions: list[dict[str, Any]],
    spans: list[dict[str, Any]],
    date_str: str,
) -> int:
    """INSERT one row into ``decisions`` per decision in the JSONL. Returns count.

    The ``citations`` column is populated with placeholder JSON objects whose
    *length* equals the sum of ``citations`` counters across matching
    ``llm.call`` spans in the trace. We don't seed real citation objects here
    — just match the count so the HTML report's ``len(citations)`` lines up
    with the README's citation column.
    """
    citation_totals = _citation_counts_by_decision(spans)
    n = 0
    for row in decisions:
        rd = row.get("research_decision") or {}
        decision_id = rd.get("id")
        if not decision_id:
            continue
        action = rd.get("action") or ""
        rationale = rd.get("rationale") or ""
        confidence = float(rd.get("confidence", 0.0))
        failure_mode = rd.get("failure_mode")  # may be None
        payload = rd.get("payload") or {}
        created_at = row.get("ts") or f"{date_str}T00:00:00+00:00"

        n_citations = citation_totals.get(decision_id, 0)
        citations_json = json.dumps(
            [{"placeholder": i} for i in range(n_citations)]
        )

        conn.execute(
            """
            INSERT OR REPLACE INTO decisions (
                id, parent_chain, action, payload, rationale,
                confidence, citations, falsification, escalation,
                failure_mode, metadata, nonce, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                "[]",  # parent_chain — empty (no chain info in committed JSONL)
                action,
                json.dumps(payload),
                rationale,
                confidence,
                citations_json,
                "",  # falsification — empty by design (T5 data shape)
                None,  # escalation
                failure_mode,
                "{}",  # metadata — empty JSON object
                "",  # nonce — empty (no broker submission for hydrated rows)
                created_at,
            ),
        )
        n += 1
    return n


def _insert_cost_ledger(
    conn: Any, spans: list[dict[str, Any]], date_str: str
) -> int:
    """INSERT one row into ``cost_ledger`` per ``llm.call`` span. Returns count."""
    n = 0
    for i, span in enumerate(spans):
        if span.get("operation") != "llm.call":
            continue
        decision_id = span.get("decision_id") or ""
        agent_raw = span.get("agent") or ""
        agent = agent_raw or "research"
        model = span.get("model") or ""
        input_tokens = int(span.get("input_tokens") or 0)
        output_tokens = int(span.get("output_tokens") or 0)
        cached_raw = int(span.get("cached_tokens") or 0)
        cached_tokens: int | None = cached_raw if cached_raw > 0 else None
        cost_usd = float(span.get("cost_usd") or 0.0)
        # Deterministic timestamp so the daily.py LIKE filter matches. Ordering
        # of cost_ledger rows doesn't affect the COST SUMMARY block since it
        # groups by model.
        created_at = f"{date_str}T00:00:0{i % 10}+00:00"

        conn.execute(
            """
            INSERT INTO cost_ledger (
                decision_id, agent, model,
                input_tokens, output_tokens, cached_tokens,
                cost_usd, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                agent,
                model,
                input_tokens,
                output_tokens,
                cached_tokens,
                cost_usd,
                created_at,
            ),
        )
        n += 1
    return n


def _insert_positions(conn: Any, date_str: str) -> int:
    """Seed the ``positions`` + ``cash`` tables for the date.

    Source data comes from the committed daily_report.md "Broker positions"
    line via the :data:`_POSITIONS_BY_DATE` lookup. For unknown dates this is
    a no-op (Positions sheet renders empty + CASH row only).
    """
    positions = _POSITIONS_BY_DATE.get(date_str, [])
    now = f"{date_str}T23:59:59+00:00"

    # Cash row first (idempotent — uses INSERT OR REPLACE because PK is fixed).
    conn.execute(
        "INSERT OR REPLACE INTO cash (id, amount, updated_at) VALUES (1, ?, ?)",
        (str(_DEFAULT_CASH), now),
    )

    n = 0
    for ticker, shares, avg_cost in positions:
        conn.execute(
            "INSERT OR REPLACE INTO positions (ticker, shares, avg_cost, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (ticker, shares, avg_cost, now),
        )
        n += 1
    return n


def hydrate(sample_runs_root: Path, date_str: str, out: Path) -> dict[str, int]:
    """Create + populate ``out`` from the sample bundle. Returns row counts."""
    date_dir = sample_runs_root / date_str
    if not date_dir.is_dir():
        raise FileNotFoundError(f"sample-run directory not found: {date_dir}")

    decisions_path = date_dir / "decisions.jsonl"
    trace_path = date_dir / "trace.jsonl"
    if not decisions_path.is_file():
        raise FileNotFoundError(f"missing decisions.jsonl: {decisions_path}")

    decisions = _load_jsonl(decisions_path)
    spans = _load_jsonl(trace_path) if trace_path.is_file() else []

    # Make sure the parent directory exists; init_db creates the file itself.
    out.parent.mkdir(parents=True, exist_ok=True)
    # If the target file already exists, delete it so a re-run starts from a
    # blank slate (hydration is intended to be one-shot).
    if out.exists():
        out.unlink()

    init_db(out)

    with closing(get_conn(out)) as conn:
        n_decisions = _insert_decisions(conn, decisions, spans, date_str)
        n_costs = _insert_cost_ledger(conn, spans, date_str)
        n_positions = _insert_positions(conn, date_str)

    return {
        "decisions": n_decisions,
        "cost_ledger": n_costs,
        "positions": n_positions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Hydrate a temp firm.db from sample_runs/<date>/decisions.jsonl "
            "+ trace.jsonl. Used by the T6 regeneration workflow."
        )
    )
    parser.add_argument(
        "--sample-runs-root",
        default="sample_runs",
        help="Root of sample_runs/ (defaults to ./sample_runs).",
    )
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--out", required=True, help="Path to the firm.db file to create."
    )
    args = parser.parse_args(argv)

    try:
        counts = hydrate(
            sample_runs_root=Path(args.sample_runs_root),
            date_str=args.date,
            out=Path(args.out),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"hydrated {args.out}: "
        f"decisions={counts['decisions']} "
        f"cost_ledger={counts['cost_ledger']} "
        f"positions={counts['positions']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
