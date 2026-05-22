"""Tests for T13: --dev-ack CLI gate on ack and reject commands.

Ensures:
- ack/reject exit 1 without --dev-ack in non-test environments
- ack/reject succeed with --dev-ack in non-test environments
- ack/reject succeed without --dev-ack when PYTEST_CURRENT_TEST is set (test bypass)
"""
from __future__ import annotations

import json
from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from click.testing import CliRunner

from firm.cli import cli
from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision, EscalatePayload
from firm.db.connection import get_conn
from firm.db.migrations import init_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clock() -> ReplayClock:
    return ReplayClock(datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc))


def _seed_pending_decision(db: Path, decision_id: str) -> None:
    """Insert a decisions row + pending hitl_queue row for the given id."""
    clock = _clock()
    now = clock.now().isoformat()
    decision = Decision(
        id=decision_id,
        decision_id_chain=["pm-1"],
        action=ActionEnum.ESCALATE,
        payload=EscalatePayload(
            proposed=BuyPayload(ticker="AAPL", shares=Decimal("50")),
            reason="trade > HITL threshold",
        ),
        rationale="requires human review",
        confidence=0.9,
        citations=[],
        falsification_condition="if cap raised",
        escalation_reason="trade > HITL threshold",
        failure_mode=None,
        metadata={},
        nonce=f"nonce-{decision_id}",
    )
    with closing(get_conn(db)) as conn:
        conn.execute(
            "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                decision.id,
                json.dumps(decision.decision_id_chain),
                decision.action.value,
                decision.payload.model_dump_json(),
                decision.rationale,
                decision.confidence,
                json.dumps([c.model_dump(mode="json") for c in decision.citations]),
                decision.falsification_condition,
                decision.escalation_reason,
                decision.failure_mode.value if decision.failure_mode else None,
                json.dumps(decision.metadata),
                decision.nonce,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO hitl_queue (decision_id, queued_at, status) VALUES (?, ?, 'pending')",
            (decision.id, now),
        )


# ---------------------------------------------------------------------------
# Test 1 — ack without --dev-ack in prod exits 1 with reminder
# ---------------------------------------------------------------------------


def test_ack_without_dev_ack_in_prod_exits_1(tmp_path: Path, monkeypatch) -> None:
    """In non-test env, ack without --dev-ack must exit 1 with Slack reminder."""
    db = tmp_path / "firm.db"
    init_db(db)
    _seed_pending_decision(db, "dec-1")

    monkeypatch.setenv("FIRM_DB_PATH", str(db))
    monkeypatch.setenv("FIRM_REPLAY_AT", "2024-06-01T12:00:00+00:00")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli, ["ack", "dec-1"])

    assert result.exit_code == 1
    assert "Slack" in result.output or "--dev-ack" in result.output


# ---------------------------------------------------------------------------
# Test 2 — ack with --dev-ack in prod succeeds and approves DB row
# ---------------------------------------------------------------------------


def test_ack_with_dev_ack_in_prod_succeeds(tmp_path: Path, monkeypatch) -> None:
    """In non-test env, ack with --dev-ack must exit 0 and approve the DB row."""
    db = tmp_path / "firm.db"
    init_db(db)
    _seed_pending_decision(db, "dec-2")

    monkeypatch.setenv("FIRM_DB_PATH", str(db))
    monkeypatch.setenv("FIRM_REPLAY_AT", "2024-06-01T12:00:00+00:00")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli, ["ack", "--dev-ack", "dec-2"])

    assert result.exit_code == 0, result.output
    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", ("dec-2",)
        ).fetchone()
    assert row is not None
    assert row["status"] == "approved"


# ---------------------------------------------------------------------------
# Test 3 — ack without --dev-ack in test env succeeds (PYTEST_CURRENT_TEST set)
# ---------------------------------------------------------------------------


def test_ack_without_dev_ack_in_test_env_succeeds(tmp_path: Path, monkeypatch) -> None:
    """When PYTEST_CURRENT_TEST is set (pytest env), ack bypasses the gate."""
    db = tmp_path / "firm.db"
    init_db(db)
    _seed_pending_decision(db, "dec-3")

    monkeypatch.setenv("FIRM_DB_PATH", str(db))
    monkeypatch.setenv("FIRM_REPLAY_AT", "2024-06-01T12:00:00+00:00")
    # PYTEST_CURRENT_TEST is naturally set by pytest — we rely on it being present
    # (but ensure it's set in case the runner strips env)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/unit/test_cli_dev_ack.py::test_ack_without_dev_ack_in_test_env_succeeds")

    runner = CliRunner()
    result = runner.invoke(cli, ["ack", "dec-3"])

    assert result.exit_code == 0, result.output
    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", ("dec-3",)
        ).fetchone()
    assert row is not None
    assert row["status"] == "approved"


# ---------------------------------------------------------------------------
# Test 4 — reject without --dev-ack in prod exits 1 with reminder
# ---------------------------------------------------------------------------


def test_reject_without_dev_ack_in_prod_exits_1(tmp_path: Path, monkeypatch) -> None:
    """In non-test env, reject without --dev-ack must exit 1 with Slack reminder."""
    db = tmp_path / "firm.db"
    init_db(db)
    _seed_pending_decision(db, "dec-4")

    monkeypatch.setenv("FIRM_DB_PATH", str(db))
    monkeypatch.setenv("FIRM_REPLAY_AT", "2024-06-01T12:00:00+00:00")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli, ["reject", "dec-4"])

    assert result.exit_code == 1
    assert "Slack" in result.output or "--dev-ack" in result.output


# ---------------------------------------------------------------------------
# Test 5 — reject with --dev-ack in prod succeeds and rejects DB row
# ---------------------------------------------------------------------------


def test_reject_with_dev_ack_in_prod_succeeds(tmp_path: Path, monkeypatch) -> None:
    """In non-test env, reject with --dev-ack must exit 0 and reject the DB row."""
    db = tmp_path / "firm.db"
    init_db(db)
    _seed_pending_decision(db, "dec-5")

    monkeypatch.setenv("FIRM_DB_PATH", str(db))
    monkeypatch.setenv("FIRM_REPLAY_AT", "2024-06-01T12:00:00+00:00")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    runner = CliRunner()
    result = runner.invoke(cli, ["reject", "--dev-ack", "dec-5"])

    assert result.exit_code == 0, result.output
    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", ("dec-5",)
        ).fetchone()
    assert row is not None
    assert row["status"] == "rejected"
