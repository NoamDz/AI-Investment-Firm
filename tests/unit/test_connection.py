from pathlib import Path
import sqlite3 as _sqlite3  # alias to avoid colliding with any other local import
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


def test_connection_creates_missing_parent_dirs(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "c" / "test.db"
    conn = get_conn(nested)
    assert nested.exists()
    conn.execute("CREATE TABLE t (x INTEGER)")


def test_foreign_keys_are_per_connection(tmp_path: Path):
    """foreign_keys=ON does NOT persist across close/reopen.
    Documents the invariant that every caller must use get_conn."""
    conn = get_conn(tmp_path / "test.db")
    conn.close()
    raw = _sqlite3.connect(str(tmp_path / "test.db"))
    assert raw.execute("PRAGMA foreign_keys").fetchone()[0] == 0
    raw.close()
