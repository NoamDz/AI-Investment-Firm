"""Append-only audit log. See spec §1, §3.4, §10.1."""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
from typing import Any

from firm.core.clock import Clock
from firm.db.connection import get_conn


class AuditLog:
    def __init__(self, db_path: Path, clock: Clock) -> None:
        self._db_path = db_path
        self._clock = clock

    def append(self, event: str, detail: dict[str, Any]) -> None:
        with closing(get_conn(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO audit_log (ts, event, detail) VALUES (?, ?, ?)",
                (self._clock.now().isoformat(), event, json.dumps(detail, default=str)),
            )

    def read_all(self) -> list[dict[str, Any]]:
        with closing(get_conn(self._db_path)) as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM audit_log ORDER BY id")]
