"""Tests for ``firm.eval.runner.run_regime`` (Plan 4 §T13).

Every test injects a stub heartbeat — the real CLI heartbeat (which needs
LLM cassettes + Qdrant) is wired by T15. The stub seeds the DB once on its
first call so the multi-day loop is observable without needing to seed
each day separately.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from firm.core.models import FailureMode
from firm.eval.regimes import R1_EARNINGS
from firm.eval.runner import HeartbeatFn, RegimeReport, run_regime


# ---------------------------------------------------------------------------
# Stub heartbeat factory: one BUY + one SELL + one HITL row + two decisions,
# seeded on the FIRST day only, so we can verify the per-day loop semantics
# without overcounting.
# ---------------------------------------------------------------------------


def _seed_db(db_path: Path) -> None:
    """Insert two decisions, two broker.fill audit events, and one hitl row.

    Decision shape mirrors the schema in firm/db/schema.sql. The BUY+SELL
    pair is constructed so the per-trade match produces a clean +10.0%
    return (100 shares @ $100 → 100 shares @ $110, zero commission).
    """
    citations_json = json.dumps(
        [
            {"source_id": "src-A", "chunk_id": "c1", "span": [0, 10]},
            {"source_id": "src-B", "chunk_id": "c2", "span": [0, 10]},
        ]
    )
    conn = sqlite3.connect(str(db_path))
    try:
        # Two decisions: one clean, one schema_validation_failed for the
        # rejection-count metric. The clean one carries two citations from
        # two distinct sources so decision_discipline + citation_diversity
        # both pass.
        conn.executemany(
            """
            INSERT INTO decisions
              (id, parent_chain, action, payload, rationale, confidence,
               citations, falsification, escalation, failure_mode, metadata,
               nonce, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    "dec-clean-1",
                    "[]",
                    "BUY",
                    json.dumps({"kind": "buy", "ticker": "AAPL", "shares": "100"}),
                    "research thesis is supported",
                    0.7,
                    citations_json,
                    "if EPS misses by 10% reverse",
                    None,
                    None,
                    "{}",
                    "nonce-1",
                    "2024-03-11T09:30:00+00:00",
                ),
                (
                    "dec-rejected-1",
                    "[]",
                    "HOLD",
                    json.dumps({"kind": "hold", "reason": "no signal"}),
                    "rejected by schema",
                    0.1,
                    citations_json,
                    "n/a",
                    None,
                    FailureMode.SCHEMA_VALIDATION_FAILED.value,
                    "{}",
                    "nonce-2",
                    "2024-03-12T09:30:00+00:00",
                ),
            ],
        )

        # Two broker.fill audit events: BUY 100 AAPL @ $100, SELL 100 AAPL @ $110.
        # Zero commission → per-trade return = +10.0% exactly.
        conn.executemany(
            "INSERT INTO audit_log (ts, event, detail) VALUES (?, ?, ?)",
            [
                (
                    "2024-03-11T09:30:00+00:00",
                    "broker.fill",
                    json.dumps(
                        {
                            "side": "buy",
                            "ticker": "AAPL",
                            "shares": "100",
                            "fill_price": "100",
                            "commission": "0",
                        }
                    ),
                ),
                (
                    "2024-03-13T15:55:00+00:00",
                    "broker.fill",
                    json.dumps(
                        {
                            "side": "sell",
                            "ticker": "AAPL",
                            "shares": "100",
                            "fill_price": "110",
                            "commission": "0",
                        }
                    ),
                ),
            ],
        )

        # One approved HITL row → hitl_correctness passes.
        conn.execute(
            """
            INSERT INTO hitl_queue
              (decision_id, queued_at, status, approver, approval_nonce, decided_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "dec-clean-1",
                "2024-03-11T09:31:00+00:00",
                "approved",
                "alice",
                "approval-nonce-1",
                "2024-03-11T09:32:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _stub_heartbeat_factory(call_log: list[date] | None = None) -> HeartbeatFn:
    """Return a stub HeartbeatFn that seeds once and records every call."""
    state = {"seeded": False}

    def _stub(day: date, db_path: Path) -> None:
        if call_log is not None:
            call_log.append(day)
        if state["seeded"]:
            return
        state["seeded"] = True
        _seed_db(db_path)

    return _stub


# ---------------------------------------------------------------------------
# Spec §9.7 section headers that the placeholder template MUST emit so the
# T14 swap doesn't regress section coverage.
# ---------------------------------------------------------------------------
_REQUIRED_SECTION_HEADERS = (
    "EVAL REPORT — Replay smoke test across 3 regimes",
    "REGIME",
    "Total return:",
    "vs SPY (primary):",
    "vs equal-weight basket:",
    "Per-trade returns:",
    "Hit rate:",
    "PROCESS METRICS (aggregated)",
    "NOT MEASURED",
)


def test_run_regime_writes_report_with_all_section_headers(tmp_path: Path) -> None:
    heartbeat = _stub_heartbeat_factory()
    report = run_regime(
        R1_EARNINGS,
        output_dir=tmp_path,
        heartbeat=heartbeat,
        spy_return=0.008,
        basket_return=-0.004,
        # No open positions after the BUY+SELL pair, so no marks needed.
        final_marks={},
    )
    content = report.report_path.read_text(encoding="utf-8")
    for header in _REQUIRED_SECTION_HEADERS:
        assert header in content, f"missing section header: {header!r}"


def test_run_regime_returns_regime_report_with_correct_counts(tmp_path: Path) -> None:
    heartbeat = _stub_heartbeat_factory()
    report = run_regime(
        R1_EARNINGS,
        output_dir=tmp_path,
        heartbeat=heartbeat,
        spy_return=0.008,
        basket_return=-0.004,
        final_marks={},
    )
    assert isinstance(report, RegimeReport)
    assert report.num_days == 5  # 2024-03-11..15 inclusive
    assert report.num_decisions == 2
    assert report.num_fills == 2
    assert report.regime_id == "r1_earnings"


def test_run_regime_calls_heartbeat_once_per_calendar_day(tmp_path: Path) -> None:
    call_log: list[date] = []
    heartbeat = _stub_heartbeat_factory(call_log=call_log)
    run_regime(
        R1_EARNINGS,
        output_dir=tmp_path,
        heartbeat=heartbeat,
        spy_return=0.008,
        basket_return=-0.004,
        final_marks={},
    )
    assert len(call_log) == 5
    # And in chronological order.
    assert call_log == [
        date(2024, 3, 11),
        date(2024, 3, 12),
        date(2024, 3, 13),
        date(2024, 3, 14),
        date(2024, 3, 15),
    ]


def test_run_regime_resets_db_on_re_run(tmp_path: Path) -> None:
    # First run.
    heartbeat_1 = _stub_heartbeat_factory()
    report_1 = run_regime(
        R1_EARNINGS,
        output_dir=tmp_path,
        heartbeat=heartbeat_1,
        spy_return=0.008,
        basket_return=-0.004,
        final_marks={},
    )
    # Second run on the same output_dir — DB MUST be wiped, not appended.
    heartbeat_2 = _stub_heartbeat_factory()
    report_2 = run_regime(
        R1_EARNINGS,
        output_dir=tmp_path,
        heartbeat=heartbeat_2,
        spy_return=0.008,
        basket_return=-0.004,
        final_marks={},
    )
    assert report_2.num_decisions == report_1.num_decisions == 2
    assert report_2.num_fills == report_1.num_fills == 2


def test_default_heartbeat_raises_not_implemented(tmp_path: Path) -> None:
    # Production heartbeat wiring is deferred to T15; the default sentinel
    # must fail loudly so a caller that forgets to wire it sees a clear error.
    with pytest.raises(NotImplementedError, match="T15"):
        run_regime(
            R1_EARNINGS,
            output_dir=tmp_path,
            spy_return=0.0,
            basket_return=0.0,
            final_marks={},
        )
