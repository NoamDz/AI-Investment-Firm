from pathlib import Path
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
