import sqlite3
from pathlib import Path

import pytest

from firm.db.connection import get_conn
from firm.db.migrations import init_db


def test_init_db_creates_all_tables(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    conn = get_conn(db)
    tables = {
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert tables == {
        "decisions",
        "outbox",
        "positions",
        "cash",
        "hitl_queue",
        "reconciliations",
        "audit_log",
        "llm_cache",
        "ingest_runs",
        "cost_ledger",
        "news_cache",
    }


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    init_db(db)  # second call must not raise
    conn = get_conn(db)
    count = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    assert count >= 7


def test_fk_constraints_enforced(tmp_path: Path) -> None:
    """Inserting an outbox row with a non-existent decision_id must raise IntegrityError.
    This locks the invariant that get_conn enables PRAGMA foreign_keys=ON."""
    db = tmp_path / "test.db"
    init_db(db)
    conn = get_conn(db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO outbox (key, decision_id, payload, status, created_at, updated_at) "
            "VALUES ('k1', 'nonexistent', '{}', 'pending', 'ts', 'ts')"
        )


def test_llm_cache_unique_on_prompt_hash_and_model(tmp_path: Path) -> None:
    """Inserting two llm_cache rows with the same (prompt_hash, model) must raise IntegrityError.
    The PRIMARY KEY (prompt_hash, model) enforces this uniqueness."""
    db = tmp_path / "test.db"
    init_db(db)
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO llm_cache (prompt_hash, model, response_json, created_at) "
        "VALUES ('hash1', 'sonnet', '{\"text\": \"ok\"}', '2026-05-19T00:00:00Z')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO llm_cache (prompt_hash, model, response_json, created_at) "
            "VALUES ('hash1', 'sonnet', '{\"text\": \"duplicate\"}', '2026-05-19T00:00:01Z')"
        )


def test_llm_cache_allows_same_hash_different_model(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_db(db)
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO llm_cache (prompt_hash, model, response_json, created_at) "
        "VALUES ('hash1', 'sonnet', '{}', '2026-05-19T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO llm_cache (prompt_hash, model, response_json, created_at) "
        "VALUES ('hash1', 'haiku', '{}', '2026-05-19T00:00:01Z')"
    )
    rows = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
    assert rows == 2


# ---------------------------------------------------------------------------
# T14: hitl_queue.approver CHECK constraint
# approver must be supplied for decided rows (approved/rejected);
# pending and timed_out rows may have NULL approver — Slack/CLI always
# supplies it on approval/rejection.
# ---------------------------------------------------------------------------

def _seed_decision_row(conn: sqlite3.Connection, decision_id: str) -> None:
    """Insert a minimal decisions row so FK constraints on hitl_queue are satisfied."""
    conn.execute(
        "INSERT INTO decisions "
        "(id, parent_chain, action, payload, rationale, confidence, citations, "
        "falsification, metadata, nonce, created_at) "
        "VALUES (?, '[]', 'BUY', '{}', 'r', 0.9, '[]', 'f', '{}', 'n', 'ts')",
        (decision_id,),
    )


def test_hitl_queue_rejects_approved_without_approver(tmp_path: Path) -> None:
    """status='approved' with approver=NULL must raise IntegrityError (CHECK constraint)."""
    db = tmp_path / "firm.db"
    init_db(db)
    conn = get_conn(db)
    _seed_decision_row(conn, "dec-001")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO hitl_queue (decision_id, queued_at, status, approver) "
            "VALUES (?, 'ts', 'approved', NULL)",
            ("dec-001",),
        )


def test_hitl_queue_rejects_rejected_without_approver(tmp_path: Path) -> None:
    """status='rejected' with approver=NULL must raise IntegrityError (CHECK constraint)."""
    db = tmp_path / "firm.db"
    init_db(db)
    conn = get_conn(db)
    _seed_decision_row(conn, "dec-002")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO hitl_queue (decision_id, queued_at, status, approver) "
            "VALUES (?, 'ts', 'rejected', NULL)",
            ("dec-002",),
        )


def test_hitl_queue_allows_pending_without_approver(tmp_path: Path) -> None:
    """status='pending' with approver=NULL must succeed (current ESCALATE flow preserved)."""
    db = tmp_path / "firm.db"
    init_db(db)
    conn = get_conn(db)
    _seed_decision_row(conn, "dec-003")
    conn.execute(
        "INSERT INTO hitl_queue (decision_id, queued_at, status, approver) "
        "VALUES (?, 'ts', 'pending', NULL)",
        ("dec-003",),
    )
    row = conn.execute(
        "SELECT approver FROM hitl_queue WHERE decision_id = ?", ("dec-003",)
    ).fetchone()
    assert row["approver"] is None


def test_hitl_queue_allows_timed_out_without_approver(tmp_path: Path) -> None:
    """status='timed_out' with approver=NULL must succeed (timeouts have no approver)."""
    db = tmp_path / "firm.db"
    init_db(db)
    conn = get_conn(db)
    _seed_decision_row(conn, "dec-004")
    conn.execute(
        "INSERT INTO hitl_queue (decision_id, queued_at, status, approver) "
        "VALUES (?, 'ts', 'timed_out', NULL)",
        ("dec-004",),
    )
    row = conn.execute(
        "SELECT approver FROM hitl_queue WHERE decision_id = ?", ("dec-004",)
    ).fetchone()
    assert row["approver"] is None


def test_hitl_queue_allows_approved_with_approver(tmp_path: Path) -> None:
    """status='approved' with approver='alice' must succeed (positive control)."""
    db = tmp_path / "firm.db"
    init_db(db)
    conn = get_conn(db)
    _seed_decision_row(conn, "dec-005")
    conn.execute(
        "INSERT INTO hitl_queue (decision_id, queued_at, status, approver) "
        "VALUES (?, 'ts', 'approved', 'alice')",
        ("dec-005",),
    )
    row = conn.execute(
        "SELECT approver FROM hitl_queue WHERE decision_id = ?", ("dec-005",)
    ).fetchone()
    assert row["approver"] == "alice"
