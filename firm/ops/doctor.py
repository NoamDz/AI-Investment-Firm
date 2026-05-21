"""Health-check logic for `firm doctor`. See Plan 3 §T23a.

Each check function accepts injected dependencies (paths, clock, qdrant client)
and returns a :class:`CheckResult`.  The CLI wires real deps; tests use stubs.
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    pass  # keep import-time cost low; QdrantClient is always passed in by caller


# ---------------------------------------------------------------------------
# Constants — mirrors config/litestream.yml and connection.py
# ---------------------------------------------------------------------------

WAL_WARN_BYTES = 12 * 1024 * 1024   # 12 MB
WAL_CEIL_BYTES = 16 * 1024 * 1024   # 16 MB (matches litestream max-wal-size)

CHECKPOINT_WARN_SECS = 5 * 60       # 5 minutes
CHECKPOINT_FAIL_SECS = 30 * 60      # 30 minutes

REPLICATION_WARN_SECS = 60          # 1 minute
REPLICATION_FAIL_SECS = 5 * 60      # 5 minutes


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------

Status = Literal["OK", "WARN", "FAIL"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str


class Clock(Protocol):
    def now(self) -> Any: ...


class QdrantCountable(Protocol):
    """Minimal interface for the qdrant client used by :func:`check_qdrant_points`."""

    def count(self, collection_name: str) -> Any: ...


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_wal_size(db_path: Path) -> CheckResult:
    """Check 1: WAL file size against 12 MB WARN / 16 MB FAIL thresholds."""
    wal = Path(str(db_path) + "-wal")
    size = os.path.getsize(wal) if wal.exists() else 0
    mb = round(size / (1024 * 1024), 2)
    detail = f"{mb} MB"

    if size >= WAL_CEIL_BYTES:
        return CheckResult(name="wal_size", status="FAIL", detail=detail)
    if size >= WAL_WARN_BYTES:
        return CheckResult(name="wal_size", status="WARN", detail=detail)
    return CheckResult(name="wal_size", status="OK", detail=detail)


def check_last_checkpoint_age(db_path: Path) -> CheckResult:
    """Check 2: Approximate age of last WAL checkpoint.

    WHY mtime-as-proxy: SQLite exposes no "last checkpoint at" timestamp.
    We run PRAGMA wal_checkpoint(PASSIVE) to flush pending pages; if the
    WAL log count is zero afterwards the database is fully checkpointed and
    we report age=0.  Otherwise we use the WAL file mtime as a conservative
    proxy for "time since last write" — a stale mtime strongly implies a
    stale checkpoint.
    """
    wal = Path(str(db_path) + "-wal")
    if not wal.exists():
        return CheckResult(name="last_checkpoint_age", status="OK", detail="no WAL")

    # Run a PASSIVE checkpoint (no-op if another writer holds the lock).
    with closing(sqlite3.connect(str(db_path))) as conn:
        row = conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
    # row = (busy, log, checkpointed)
    log_pages: int = row[1] if row else 0

    if log_pages == 0:
        age_s = 0
    else:
        # WHY: mtime is the proxy for "last WAL write" per the spec note above.
        age_s = int(time.time() - os.path.getmtime(str(wal)))

    detail = f"{age_s}s"
    if age_s >= CHECKPOINT_FAIL_SECS:
        return CheckResult(name="last_checkpoint_age", status="FAIL", detail=detail)
    if age_s >= CHECKPOINT_WARN_SECS:
        return CheckResult(name="last_checkpoint_age", status="WARN", detail=detail)
    return CheckResult(name="last_checkpoint_age", status="OK", detail=detail)


def check_last_replication(litestream_dir: Path) -> CheckResult:
    """Check 3: Age of most-recent file under the litestream replica directory."""
    if not litestream_dir.exists():
        return CheckResult(
            name="last_replication", status="FAIL", detail="litestream dir missing"
        )

    mtimes = [
        os.path.getmtime(str(p))
        for p in litestream_dir.rglob("*")
        if p.is_file()
    ]
    if not mtimes:
        # Dir exists but is empty — treat as missing replica content.
        return CheckResult(
            name="last_replication", status="FAIL", detail="litestream dir missing"
        )

    age_s = int(time.time() - max(mtimes))
    detail = f"{age_s}s ago"

    if age_s >= REPLICATION_FAIL_SECS:
        return CheckResult(name="last_replication", status="FAIL", detail=detail)
    if age_s >= REPLICATION_WARN_SECS:
        return CheckResult(name="last_replication", status="WARN", detail=detail)
    return CheckResult(name="last_replication", status="OK", detail=detail)


def check_qdrant_points(
    qdrant_client: QdrantCountable,
    collection_name: str,
) -> CheckResult:
    """Check 4: Total point count in the named Qdrant collection."""
    try:
        count = qdrant_client.count(collection_name=collection_name).count
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            name="qdrant_points",
            status="FAIL",
            detail=f"unreachable: {type(exc).__name__}",
        )

    detail = f"{count} points in {collection_name}"
    if count == 0:
        return CheckResult(name="qdrant_points", status="WARN", detail=detail)
    return CheckResult(name="qdrant_points", status="OK", detail=detail)


def check_cost_ledger_today(db_path: Path, clock: Clock) -> CheckResult:
    """Check 5: Number of cost_ledger rows written today (UTC).

    Uses ``clock.now()`` for determinism in tests — no wall-clock dependency.
    """
    today_utc = clock.now().strftime("%Y-%m-%d")
    midnight_iso = f"{today_utc}T00:00:00+00:00"

    with closing(sqlite3.connect(str(db_path))) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM cost_ledger WHERE created_at >= ?",
            (midnight_iso,),
        ).fetchone()
    n: int = row[0] if row else 0
    detail = f"{n} rows"

    if n == 0:
        return CheckResult(name="cost_ledger_today", status="WARN", detail=detail)
    return CheckResult(name="cost_ledger_today", status="OK", detail=detail)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_doctor(
    *,
    db_path: Path,
    litestream_dir: Path,
    qdrant_client: QdrantCountable,
    collection_name: str,
    clock: Clock,
) -> list[CheckResult]:
    """Run all five health checks in the canonical order and return results."""
    return [
        check_wal_size(db_path),
        check_last_checkpoint_age(db_path),
        check_last_replication(litestream_dir),
        check_qdrant_points(qdrant_client, collection_name),
        check_cost_ledger_today(db_path, clock),
    ]


def format_results(results: list[CheckResult]) -> str:
    """Render check results as one line each: ``<STATUS> <name>: <detail>``."""
    return "\n".join(f"{r.status} {r.name}: {r.detail}" for r in results)
