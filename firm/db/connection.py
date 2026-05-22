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
    mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
    if mode != "wal":
        raise RuntimeError(f"SQLite failed to switch to WAL mode (got {mode!r})")
    conn.execute("PRAGMA synchronous = FULL")
    # wal_autocheckpoint pages (default 1000): trigger an automatic checkpoint
    # whenever the WAL reaches ~4MB (1000 pages × 4KB default page size). Paired
    # with litestream's max-wal-size: 16MB ceiling in config/litestream.yml,
    # this keeps a stuck checkpointer visible as a replication error rather
    # than unbounded WAL growth.
    conn.execute("PRAGMA wal_autocheckpoint = 1000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
