"""Streamlit dashboard — primary demoable report channel.

Reads `firm.db` directly (positions, decisions, cost_ledger, audit_log,
hitl_queue, reconciliations) and renders a live view that updates while
`firm run --loop` is producing heartbeats.

Run:
    pip install -e ".[dashboard]"
    streamlit run firm/dashboard.py

Env:
    FIRM_DB_PATH       sqlite path (default: data/firm.db)
    FIRM_REPORTS_ROOT  reports root for the xlsx link (default: data/reports)
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
REFRESH_SECONDS = 5

ACTION_COLORS = {
    "BUY": "#1b8a4d",
    "SELL": "#c0392b",
    "HOLD": "#7f8c8d",
    "REFUSE": "#8e44ad",
    "ESCALATE": "#e67e22",
}


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
    if not _table_exists(conn, "positions"):
        return pd.DataFrame()
    rows = conn.execute(
        "SELECT ticker, shares, avg_cost, updated_at FROM positions ORDER BY ticker"
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["shares"] = df["shares"].astype(float)
    df["avg_cost"] = df["avg_cost"].astype(float)
    df["gross_value"] = df["shares"] * df["avg_cost"]
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


def _read_cost_today(conn: sqlite3.Connection) -> dict[str, Any]:
    if not _table_exists(conn, "cost_ledger"):
        return {}
    today = datetime.utcnow().strftime("%Y-%m-%d")
    total = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) FROM cost_ledger WHERE date(created_at)=?",
        (today,),
    ).fetchone()[0]
    cached = conn.execute(
        "SELECT COUNT(*) FROM cost_ledger WHERE date(created_at)=? AND cached_tokens IS NOT NULL",
        (today,),
    ).fetchone()[0]
    live = conn.execute(
        "SELECT COUNT(*) FROM cost_ledger WHERE date(created_at)=? AND input_tokens IS NOT NULL",
        (today,),
    ).fetchone()[0]
    return {
        "total_usd": float(total),
        "cached_calls": cached,
        "live_calls": live,
        "cache_pct": (cached / (cached + live) * 100.0) if (cached + live) else 0.0,
    }


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


def _latest_xlsx() -> Path | None:
    if not REPORTS_ROOT.exists():
        return None
    candidates = sorted(REPORTS_ROOT.glob("*/positions.xlsx"), reverse=True)
    return candidates[0] if candidates else None


def render() -> None:
    st.set_page_config(page_title="AI Investment Firm", layout="wide", page_icon="📈")
    st.title("AI Investment Firm — Live Desk")
    st.caption(
        f"Reading {DB_PATH} · refresh every {REFRESH_SECONDS}s · "
        f"started {datetime.utcnow().strftime('%H:%M:%S UTC')}"
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
    cost = _read_cost_today(conn)
    recon = _read_recon(conn)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cash", f"${cash:,.0f}" if cash is not None else "—")
    gross = positions["gross_value"].sum() if not positions.empty else 0.0
    col2.metric("Gross exposure", f"${gross:,.0f}", f"{len(positions)} positions")
    col3.metric("LLM spend (today)", f"${cost.get('total_usd', 0):.3f}")
    col4.metric("Cache hit %", f"{cost.get('cache_pct', 0):.0f}%",
                f"{cost.get('cached_calls', 0)}c / {cost.get('live_calls', 0)}l")

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Positions")
        if positions.empty:
            st.info("No open positions yet.")
        else:
            st.dataframe(
                positions[["ticker", "shares", "avg_cost", "gross_value", "updated_at"]],
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
                    f"<div style='border-left:4px solid {color}; padding:6px 12px; margin:4px 0;"
                    f" background:#f7f7f9; border-radius:4px;'>"
                    f"<b style='color:{color}'>{row['action']}</b>{ticker}{shares}"
                    f" <span style='color:#888; font-size:0.85em'>"
                    f"conf={row['confidence']:.2f} · {row['citations']} cite{fm} · {row['ts']}</span>"
                    f"<br><span style='color:#333; font-size:0.9em'>{row['rationale']}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()
    foot_l, foot_r = st.columns([1, 1])
    with foot_l:
        st.subheader("Reconciliation")
        if recon is None:
            st.info("No reconciliation events yet.")
        else:
            status_emoji = {"ok": "✅", "mismatch": "⚠️", "acked": "🛠"}.get(recon["status"], "❓")
            st.write(f"{status_emoji} **{recon['kind'].upper()}** @ {recon['ran_at']} — `{recon['status']}`")
            if recon["diff"]:
                st.json(recon["diff"], expanded=False)
    with foot_r:
        st.subheader("Excel export (channel #2)")
        xlsx = _latest_xlsx()
        if xlsx is None:
            st.info(f"No positions.xlsx under {REPORTS_ROOT} yet.")
        else:
            with xlsx.open("rb") as f:
                st.download_button(
                    f"Download {xlsx.parent.name}/positions.xlsx",
                    f.read(),
                    file_name=f"positions_{xlsx.parent.name}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    conn.close()

    # Auto-refresh — Streamlit reruns the script every REFRESH_SECONDS.
    import time as _time
    _time.sleep(REFRESH_SECONDS)
    st.rerun()


if __name__ == "__main__":
    render()
