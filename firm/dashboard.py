"""Streamlit dashboard — primary report channel + live observability.

Three-tab layout (Plan 4 / PLAN_reports_overhaul.md §3):

* **Tab 1 — Today's Report** (default). Embeds ``daily_report.html`` for the
  selected date and exposes the full download bundle. No auto-refresh.
* **Tab 2 — Live Desk**. The legacy live view that refreshes every 5 s while
  ``firm run --loop`` is producing heartbeats — reads ``firm.db`` directly.
* **Tab 3 — Trace**. Span-level lookup by ``decision_id`` against either
  ``data/traces/<date>/run-*.jsonl`` (live) or ``sample_runs/<date>/trace.jsonl``.

Run:
    pip install -e ".[dashboard]"
    streamlit run firm/dashboard.py

Env:
    FIRM_DB_PATH           sqlite path (default: data/firm.db)
    FIRM_REPORTS_ROOT      reports root for per-date bundles (default: data/reports)
    FIRM_SAMPLE_RUNS_ROOT  committed sample-run snapshots (default: sample_runs)
    FIRM_TRACES_ROOT       live trace root (default: data/traces)
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

DB_PATH = Path(os.environ.get("FIRM_DB_PATH", "data/firm.db"))
REPORTS_ROOT = Path(os.environ.get("FIRM_REPORTS_ROOT", "data/reports"))
SAMPLE_RUNS_ROOT = Path(os.environ.get("FIRM_SAMPLE_RUNS_ROOT", "sample_runs"))
TRACES_ROOT = Path(os.environ.get("FIRM_TRACES_ROOT", "data/traces"))
REFRESH_SECONDS = 5

ACTION_COLORS = {
    "BUY": "#1b8a4d",
    "SELL": "#c0392b",
    "HOLD": "#7f8c8d",
    "REFUSE": "#8e44ad",
    "ESCALATE": "#e67e22",
}

# Download-button MIME types per artifact name.
_BUNDLE_MIME = {
    "daily_report.html": "text/html",
    "daily_report.md": "text/markdown",
    "positions.xlsx": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ),
    "decisions.jsonl": "application/x-ndjson",
}

# Order matters: shown top-to-bottom under "Download bundle".
_BUNDLE_FILES = (
    "daily_report.html",
    "daily_report.md",
    "positions.xlsx",
    "decisions.jsonl",
)


# ---------------------------------------------------------------------------
# DB readers (unchanged from previous revision — read live firm.db)
# ---------------------------------------------------------------------------


def _connect() -> sqlite3.Connection | None:
    if not DB_PATH.exists():
        return None
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def _read_cash(conn: sqlite3.Connection) -> Decimal | None:
    if not _table_exists(conn, "cash"):
        return None
    row = conn.execute("SELECT amount FROM cash WHERE id=1").fetchone()
    return Decimal(row["amount"]) if row else None


def _read_positions(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read positions and mark them to the current deterministic quote.

    Re-uses ``_deterministic_price`` from the FakeBroker so the dashboard
    agrees with what the firm actually quotes — no drift between the broker's
    fill price and the dashboard's mark. ``gross_value`` is current mark, not
    cost basis; ``unrealized_pnl`` = (mark − avg_cost) × shares.
    """
    if not _table_exists(conn, "positions"):
        return pd.DataFrame()
    rows = conn.execute(
        "SELECT ticker, shares, avg_cost, updated_at FROM positions ORDER BY ticker"
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    # Lazy import — keeps dashboard importable without the broker package on
    # the path during static checks.
    from firm.broker.fake_broker import _deterministic_price

    df = pd.DataFrame([dict(r) for r in rows])
    df["shares"] = df["shares"].astype(float)
    df["avg_cost"] = df["avg_cost"].astype(float)
    df["mark"] = df["ticker"].map(lambda t: float(_deterministic_price(t)))
    df["gross_value"] = df["shares"].abs() * df["mark"]
    df["unrealized_pnl"] = (df["mark"] - df["avg_cost"]) * df["shares"]
    return df


def _read_decisions(conn: sqlite3.Connection, limit: int = 25) -> pd.DataFrame:
    if not _table_exists(conn, "decisions"):
        return pd.DataFrame()
    rows = conn.execute(
        "SELECT id, action, rationale, confidence, failure_mode, created_at, payload, citations "
        "FROM decisions ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    out = []
    for r in rows:
        payload = json.loads(r["payload"]) if r["payload"] else {}
        citations = json.loads(r["citations"]) if r["citations"] else []
        out.append(
            {
                "ts": r["created_at"],
                "action": r["action"],
                "ticker": payload.get("ticker", ""),
                "shares": payload.get("shares", ""),
                "confidence": float(r["confidence"]),
                "failure_mode": r["failure_mode"] or "",
                "citations": len(citations),
                "rationale": r["rationale"],
                "id": r["id"],
            }
        )
    return pd.DataFrame(out)


def _read_hitl(conn: sqlite3.Connection) -> pd.DataFrame:
    if not _table_exists(conn, "hitl_queue"):
        return pd.DataFrame()
    rows = conn.execute(
        "SELECT decision_id, queued_at, status, approver, decided_at "
        "FROM hitl_queue ORDER BY queued_at DESC LIMIT 20"
    ).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def _read_recon(conn: sqlite3.Connection) -> dict[str, Any] | None:
    if not _table_exists(conn, "reconciliations"):
        return None
    row = conn.execute(
        "SELECT kind, ran_at, status, diff FROM reconciliations ORDER BY ran_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return {
        "kind": row["kind"],
        "ran_at": row["ran_at"],
        "status": row["status"],
        "diff": json.loads(row["diff"]) if row["diff"] else {},
    }


# ---------------------------------------------------------------------------
# Per-date path helpers (Tab 1 + Tab 3)
# ---------------------------------------------------------------------------


def _list_available_dates() -> list[tuple[str, Path]]:
    """Return ``[(date_str, root_path), ...]`` sorted by date DESC.

    Live ``data/reports/<date>/`` directories take precedence; committed
    ``sample_runs/<date>/`` directories are appended only when the same date
    is not already present in the live set. A directory only counts if its
    name parses as ``YYYY-MM-DD`` and it actually exists as a directory.
    """
    seen: set[str] = set()
    out: list[tuple[str, Path]] = []

    for root in (REPORTS_ROOT, SAMPLE_RUNS_ROOT):
        if not root.exists():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            name = child.name
            if len(name) != 10 or name[4] != "-" or name[7] != "-":
                continue
            try:
                datetime.strptime(name, "%Y-%m-%d")
            except ValueError:
                continue
            if name in seen:
                continue
            seen.add(name)
            out.append((name, root))

    out.sort(key=lambda t: t[0], reverse=True)
    return out


def _resolve_bundle_path(
    date_str: str, source_root: Path, filename: str
) -> Path | None:
    """Return ``source_root/<date>/<filename>`` if it exists, else None."""
    candidate = source_root / date_str / filename
    return candidate if candidate.is_file() else None


def _resolve_trace_path(date_str: str, source_root: Path) -> Path | None:
    """Prefer ``data/traces/<date>/run-*.jsonl`` (live), else fall back to
    ``source_root/<date>/trace.jsonl`` (committed sample). Returns None if
    neither exists."""
    live_dir = TRACES_ROOT / date_str
    if live_dir.is_dir():
        matches = sorted(live_dir.glob("run-*.jsonl"))
        if matches:
            return matches[0]
    fallback = source_root / date_str / "trace.jsonl"
    return fallback if fallback.is_file() else None


# ---------------------------------------------------------------------------
# Tab 1 — Today's Report
# ---------------------------------------------------------------------------


def _render_today_tab(date_str: str, source_root: Path) -> None:
    """Embed the per-date HTML report and expose download buttons."""
    st.subheader(f"Daily report — {date_str}")

    html_path = _resolve_bundle_path(date_str, source_root, "daily_report.html")
    if html_path is not None:
        html_content = html_path.read_text(encoding="utf-8")
        st.components.v1.html(html_content, height=1800, scrolling=True)
    else:
        st.info(
            "daily_report.html not yet generated for this date. Run "
            f"`python -m firm.cli report --date {date_str}` to create it."
        )
        md_path = _resolve_bundle_path(date_str, source_root, "daily_report.md")
        if md_path is not None:
            st.markdown(md_path.read_text(encoding="utf-8"))
        else:
            st.warning(
                "No daily_report.md either — bundle has not been generated yet."
            )

    st.divider()
    st.subheader("Download bundle")
    any_button = False
    for filename in _BUNDLE_FILES:
        path = _resolve_bundle_path(date_str, source_root, filename)
        if path is None:
            continue
        size_kb = path.stat().st_size / 1024.0
        mime = _BUNDLE_MIME.get(filename, "application/octet-stream")
        with path.open("rb") as f:
            st.download_button(
                label=f"Download {filename} ({size_kb:.1f} KB)",
                data=f.read(),
                file_name=f"{date_str}_{filename}",
                mime=mime,
                key=f"dl-{date_str}-{filename}",
            )
        any_button = True
    if not any_button:
        st.info("No artifacts found in the selected bundle directory.")


# ---------------------------------------------------------------------------
# Tab 2 — Live Desk (the demoted legacy dashboard)
# ---------------------------------------------------------------------------


def _render_live_desk_tab() -> None:
    """The legacy live-observability view. Reads ``firm.db`` and self-refreshes.

    Refresh strategy: we always call ``st.rerun()`` at the end of this tab,
    even if the user is on Tab 1 or 3. Streamlit's ``st.tabs`` preserves the
    active selection across reruns, so the user sees no flicker on the static
    tabs while the live data behind this tab stays current.
    """
    st.caption(
        f"Live observability — updates every {REFRESH_SECONDS}s while "
        f"`firm run --loop` is running. Reading {DB_PATH}."
    )

    conn = _connect()
    if conn is None:
        st.warning(
            f"No firm.db at `{DB_PATH}`. Start the firm with "
            "`python -m firm.cli run --loop --interval-seconds 60`."
        )
        return

    cash = _read_cash(conn)
    positions = _read_positions(conn)
    recon = _read_recon(conn)

    # Top-row tiles: portfolio P&L instead of LLM ops debug metrics. Operators
    # care about "how is the firm doing", not cache hit %. LLM cost remains
    # available in the per-date daily_report.html bundle for cost audits.
    cash_f = float(cash) if cash is not None else 0.0
    gross = float(positions["gross_value"].sum()) if not positions.empty else 0.0
    unrealized = (
        float(positions["unrealized_pnl"].sum()) if not positions.empty else 0.0
    )
    cost_basis = (
        float((positions["avg_cost"] * positions["shares"].abs()).sum())
        if not positions.empty
        else 0.0
    )
    total_equity = cash_f + gross
    unrealized_pct = (unrealized / cost_basis * 100.0) if cost_basis else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total equity", f"${total_equity:,.0f}")
    col2.metric("Cash", f"${cash_f:,.0f}" if cash is not None else "—")
    col3.metric(
        "Positions value",
        f"${gross:,.0f}",
        f"{len(positions)} position{'s' if len(positions) != 1 else ''}",
    )
    col4.metric(
        "Unrealized P&L",
        f"${unrealized:+,.0f}",
        f"{unrealized_pct:+.1f}% of cost basis" if cost_basis else "—",
    )

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Positions")
        if positions.empty:
            st.info("No open positions yet.")
        else:
            st.dataframe(
                positions[
                    [
                        "ticker",
                        "shares",
                        "avg_cost",
                        "mark",
                        "gross_value",
                        "unrealized_pnl",
                        "updated_at",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("HITL queue")
        hitl = _read_hitl(conn)
        if hitl.empty:
            st.info("No HITL items.")
        else:
            st.dataframe(hitl, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Recent decisions")
        decisions = _read_decisions(conn)
        if decisions.empty:
            st.info("No decisions yet — waiting for heartbeat.")
        else:
            for _, row in decisions.iterrows():
                color = ACTION_COLORS.get(row["action"], "#444")
                ticker = f" · {row['ticker']}" if row["ticker"] else ""
                shares = f" × {row['shares']}" if row["shares"] else ""
                fm = f" · {row['failure_mode']}" if row["failure_mode"] else ""
                st.markdown(
                    f"<div style='border-left:4px solid {color}; padding:6px 12px;"
                    f" margin:4px 0; background:#f7f7f9; border-radius:4px;'>"
                    f"<b style='color:{color}'>{row['action']}</b>{ticker}{shares}"
                    f" <span style='color:#888; font-size:0.85em'>"
                    f"conf={row['confidence']:.2f} · {row['citations']} cite{fm}"
                    f" · {row['ts']}</span>"
                    f"<br><span style='color:#333; font-size:0.9em'>{row['rationale']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.subheader("Reconciliation")
    if recon is None:
        st.info("No reconciliation events yet.")
    else:
        status_emoji = {"ok": "✅", "mismatch": "⚠️", "acked": "🛠"}.get(
            recon["status"], "❓"
        )
        st.write(
            f"{status_emoji} **{recon['kind'].upper()}** @ {recon['ran_at']} — "
            f"`{recon['status']}`"
        )
        if recon["diff"]:
            st.json(recon["diff"], expanded=False)

    conn.close()


# ---------------------------------------------------------------------------
# Tab 3 — Trace
# ---------------------------------------------------------------------------


# Span fields we surface in the trace table (in display order).
_TRACE_COLUMNS = (
    "operation",
    "agent",
    "duration_ms",
    "model",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "citations",
    "status",
)


def _load_trace_spans(trace_path: Path, decision_id: str) -> list[dict[str, Any]]:
    """Parse JSONL file, returning spans matching ``decision_id`` in file order."""
    matched: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                span = json.loads(line)
            except json.JSONDecodeError:
                continue
            if span.get("decision_id") == decision_id:
                matched.append({col: span.get(col, "") for col in _TRACE_COLUMNS})
    return matched


def _render_trace_tab(date_str: str, source_root: Path) -> None:
    """Per-decision span explorer."""
    st.subheader(f"Trace — {date_str}")

    trace_path = _resolve_trace_path(date_str, source_root)
    if trace_path is None:
        st.warning(
            f"No trace file found under `{TRACES_ROOT / date_str}` or "
            f"`{source_root / date_str / 'trace.jsonl'}`."
        )
        return

    st.caption(f"Reading {trace_path}")

    decision_id = st.text_input("decision_id", placeholder="dec-buy-1")
    if not decision_id:
        st.info("Enter a decision_id above to inspect its span chain.")
        return

    spans = _load_trace_spans(trace_path, decision_id)
    if not spans:
        st.info(
            f"No spans found for decision_id={decision_id!r} in {trace_path.name}"
        )
        return

    df = pd.DataFrame(spans, columns=list(_TRACE_COLUMNS))
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**Copy the grep command:**")
    cmd = f"grep '\"decision_id\":\"{decision_id}\"' {trace_path} | jq ."
    st.code(cmd, language="bash")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render() -> None:
    st.set_page_config(
        page_title="AI Investment Firm",
        layout="wide",
        page_icon="📈",
    )
    st.title("AI Investment Firm — Daily Report")

    dates = _list_available_dates()
    if not dates:
        st.warning(
            f"No date directories found under `{REPORTS_ROOT}` or "
            f"`{SAMPLE_RUNS_ROOT}`. Generate a report with "
            "`python -m firm.cli report --date YYYY-MM-DD` or commit sample runs."
        )
        return

    labels = [d for d, _ in dates]
    selected_label = st.selectbox("Date", labels, index=0)
    selected_date, selected_source = next(
        (d, r) for d, r in dates if d == selected_label
    )
    st.caption(f"Source: {selected_source / selected_date}")

    tab1, tab2, tab3 = st.tabs(["Today's Report", "Live Desk", "Trace"])

    with tab1:
        _render_today_tab(selected_date, selected_source)
    with tab2:
        _render_live_desk_tab()
    with tab3:
        _render_trace_tab(selected_date, selected_source)

    # Auto-refresh — Streamlit reruns the script every REFRESH_SECONDS so the
    # Live Desk tab stays current. ``st.tabs`` preserves the active tab across
    # reruns, so users on Tabs 1/3 see no flicker.
    import time as _time

    _time.sleep(REFRESH_SECONDS)
    st.rerun()


if __name__ == "__main__":
    render()
