import sqlite3
from pathlib import Path

import pytest

from firm.db.connection import get_conn
from firm.db.migrations import init_db


def test_init_db_creates_all_tables(tmp_path: Path):
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
    }


def test_init_db_is_idempotent(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    init_db(db)  # second call must not raise
    conn = get_conn(db)
    count = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    assert count >= 7


def test_fk_constraints_enforced(tmp_path: Path):
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
