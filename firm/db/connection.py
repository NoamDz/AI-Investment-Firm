"""SQLite connection with WAL + synchronous=FULL + foreign keys. See spec §5.1."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def get_conn(db_path: Path) -> sqlite3.Connection:
    """Open (or create) a SQLite connection with durability-grade pragmas.

    journal_mode=WAL allows concurrent readers + single writer.
    synchronous=FULL flushes WAL to disk on every commit (no lost commits on crash).
    foreign_keys=ON enforces FK constraints (off by default in SQLite).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
