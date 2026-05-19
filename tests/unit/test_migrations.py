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
        "llm_cache",
        "ingest_runs",
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


def test_llm_cache_unique_on_prompt_hash_and_model(tmp_path: Path):
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


def test_llm_cache_allows_same_hash_different_model(tmp_path: Path):
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
