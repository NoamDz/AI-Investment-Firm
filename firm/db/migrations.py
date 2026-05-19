"""Database initialization. Reads schema.sql and applies idempotently."""
from __future__ import annotations

from contextlib import closing
from pathlib import Path

from firm.db.connection import get_conn


def init_db(db_path: Path) -> None:
    """Apply schema.sql idempotently.

    Uses executescript, which commits any open transaction before running.
    Do not call inside an open transaction.
    """
    schema_sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    with closing(get_conn(db_path)) as conn:
        conn.executescript(schema_sql)
