"""Database initialization. Reads schema.sql and applies idempotently."""
from __future__ import annotations

from pathlib import Path

from firm.db.connection import get_conn


def init_db(db_path: Path) -> None:
    schema_sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    conn = get_conn(db_path)
    conn.executescript(schema_sql)
