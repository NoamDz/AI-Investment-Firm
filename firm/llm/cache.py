"""SQLite-backed LLM response cache keyed by (prompt_hash, model). See Plan 2 §T15.

The cache table (``llm_cache``) is declared in ``firm/db/schema.sql``. This module
exposes:

* :func:`hash_prompt` - deterministic sha256 over canonical JSON of the model
  inputs (``system``, ``messages``, ``tools``). Canonicalisation uses
  ``sort_keys=True`` so that dict-ordering noise does not invalidate the cache.
* :class:`LlmCache` - get/put accessor backed by the SQLite table. ``put`` is
  idempotent via ``INSERT OR REPLACE``, so retrying after a partial failure does
  not raise on the primary key.
* :class:`CachedResponse` - Pydantic v2 model returned by ``get``.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

from firm.core.clock import Clock
from firm.db.connection import get_conn

# Serialises concurrent put() calls so the SQLite WAL doesn't surface
# "database is locked" errors when multiple threads write simultaneously.
_WRITE_LOCK = threading.Lock()


class CachedResponse(BaseModel):
    response_json: str
    input_tokens: int | None
    output_tokens: int | None
    created_at: datetime  # tz-aware UTC


def hash_prompt(
    *,
    system: str | None,
    messages: Sequence[dict[str, object]],
    tools: Sequence[dict[str, object]] | None,
) -> str:
    """sha256 over canonical JSON of all model-relevant inputs.

    ``sort_keys=True`` ensures dict-ordering noise does not invalidate the cache,
    while ``separators=(",", ":")`` removes whitespace variation. ``default=str``
    is a defensive fallback for incidental non-JSON-native values (e.g. enums);
    callers should still pass plain JSON types.
    """
    canonical = json.dumps(
        {
            "system": system,
            "messages": list(messages),
            "tools": list(tools) if tools is not None else None,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_created_at(raw: str) -> datetime:
    """Parse the stored ISO-8601 timestamp back into a tz-aware UTC datetime."""
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        # Defensive: legacy rows (if any) without offset are treated as UTC.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class LlmCache:
    """SQLite-backed deterministic LLM response cache.

    Concurrency model
    -----------------
    Each :class:`LlmCache` instance maintains one SQLite connection **per
    thread**, opened lazily on first use via :meth:`_thread_conn` and stored in
    a :class:`threading.local` slot.  This avoids the per-call overhead of
    opening and closing a fresh connection while remaining safe to share across
    threads.

    * **Reads** (``get``) are lock-free: WAL mode allows any number of
      concurrent readers without blocking each other or an in-progress write.
    * **Writes** (``put``) acquire the module-level :data:`_WRITE_LOCK` before
      executing ``INSERT OR REPLACE``.  This serialises writes across threads so
      the SQLite WAL never surfaces ``"database is locked"`` errors.

    Long-lived processes do not need to call :meth:`close`; each thread's
    connection lives for the thread's lifetime and is released when the thread
    exits.  Tests that want deterministic cleanup should call
    ``cache.close()`` explicitly from the thread that opened the connection.

    The underlying connection is WAL-mode with ``synchronous=FULL``
    (see :func:`firm.db.connection.get_conn`).
    """

    def __init__(self, db_path: Path, clock: Clock) -> None:
        self._db_path = db_path
        self._clock = clock
        self._local: threading.local = threading.local()

    def _thread_conn(self) -> sqlite3.Connection:
        """Return (or lazily create) this thread's SQLite connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = get_conn(self._db_path)
        return self._local.conn  # type: ignore[no-any-return]

    def close(self) -> None:
        """Close the calling thread's connection, if one was opened.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            del self._local.conn

    def get(self, *, prompt_hash: str, model: str) -> CachedResponse | None:
        conn = self._thread_conn()
        row = conn.execute(
            "SELECT response_json, input_tokens, output_tokens, created_at "
            "FROM llm_cache WHERE prompt_hash = ? AND model = ?",
            (prompt_hash, model),
        ).fetchone()
        if row is None:
            return None
        return CachedResponse(
            response_json=row["response_json"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            created_at=_parse_created_at(row["created_at"]),
        )

    def put(
        self,
        *,
        prompt_hash: str,
        model: str,
        response_json: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        created_at = self._clock.now().isoformat()
        conn = self._thread_conn()
        with _WRITE_LOCK:
            conn.execute(
                "INSERT OR REPLACE INTO llm_cache "
                "(prompt_hash, model, response_json, input_tokens, output_tokens, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (prompt_hash, model, response_json, input_tokens, output_tokens, created_at),
            )
