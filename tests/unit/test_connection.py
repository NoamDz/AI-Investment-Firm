from pathlib import Path
from firm.db.connection import get_conn


def test_connection_applies_pragmas(tmp_path: Path):
    conn = get_conn(tmp_path / "test.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_connection_row_factory_returns_dicts(tmp_path: Path):
    conn = get_conn(tmp_path / "test.db")
    conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'hello')")
    row = conn.execute("SELECT * FROM t").fetchone()
    assert row["a"] == 1
    assert row["b"] == "hello"
