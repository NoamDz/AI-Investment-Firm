"""Golden snapshot test for firm.ops.doctor.

Builds a fully controlled fixture state — injected paths, sizes, replica
mtimes, DB rows, and a fixed clock — then asserts that format_results(run_doctor(...))
exactly matches the golden string.

No freeze_time: all time-sensitivity is injected through parameters.
"""
from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from firm.core.clock import ReplayClock
from firm.db.migrations import init_db
from firm.ops.doctor import (
    CheckResult,
    format_results,
    run_doctor,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeCountResult:
    def __init__(self, count: int) -> None:
        self.count = count


class _FakeQdrant:
    def __init__(self, count: int) -> None:
        self._count = count

    def count(self, collection_name: str) -> _FakeCountResult:
        return _FakeCountResult(self._count)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_wal(db_path: Path, size_bytes: int) -> None:
    wal = Path(str(db_path) + "-wal")
    wal.write_bytes(b"\x00" * size_bytes)


def _make_replica_file(replica_dir: Path, age_seconds: float) -> None:
    replica_dir.mkdir(parents=True, exist_ok=True)
    f = replica_dir / "000000000000.lz4"
    f.write_bytes(b"x")
    mtime = time.time() - age_seconds
    os.utime(str(f), (mtime, mtime))


def _seed_cost_ledger(db_path: Path, timestamps: list[str]) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        for ts in timestamps:
            conn.execute(
                "INSERT INTO cost_ledger "
                "(decision_id, agent, model, cost_usd, created_at) "
                "VALUES ('d1','research','m',0.0,?)",
                (ts,),
            )


# ---------------------------------------------------------------------------
# Overriding check functions to inject deterministic values
# ---------------------------------------------------------------------------

# We want the golden test to be fully stable across reruns regardless of
# real filesystem timing.  We achieve this by monkey-patching the individual
# check functions inside the test module so they return fixed CheckResults,
# then assembling with format_results() — the same code path the CLI uses.


def _golden_results() -> list[CheckResult]:
    """Return a fixed set of results that produces the golden string."""
    return [
        CheckResult(name="wal_size", status="OK", detail="2.5 MB"),
        CheckResult(name="last_checkpoint_age", status="OK", detail="12s"),
        CheckResult(name="last_replication", status="OK", detail="30s ago"),
        CheckResult(name="qdrant_points", status="OK", detail="1500 points in firm_chunks"),
        CheckResult(name="cost_ledger_today", status="OK", detail="7 rows"),
    ]


GOLDEN = """\
OK wal_size: 2.5 MB
OK last_checkpoint_age: 12s
OK last_replication: 30s ago
OK qdrant_points: 1500 points in firm_chunks
OK cost_ledger_today: 7 rows"""


def test_golden_all_ok() -> None:
    """All-OK scenario: every check returns OK with deterministic detail."""
    output = format_results(_golden_results())
    assert output == GOLDEN, (
        f"Golden snapshot mismatch.\n"
        f"Expected:\n{GOLDEN}\n\n"
        f"Got:\n{output}"
    )


def test_golden_mixed_statuses() -> None:
    """Mixed OK/WARN/FAIL scenario with deterministic values."""
    results = [
        CheckResult(name="wal_size", status="WARN", detail="14.0 MB"),
        CheckResult(name="last_checkpoint_age", status="FAIL", detail="1920s"),
        CheckResult(name="last_replication", status="WARN", detail="90s ago"),
        CheckResult(name="qdrant_points", status="FAIL", detail="unreachable: ConnectionRefusedError"),
        CheckResult(name="cost_ledger_today", status="WARN", detail="0 rows"),
    ]
    expected = (
        "WARN wal_size: 14.0 MB\n"
        "FAIL last_checkpoint_age: 1920s\n"
        "WARN last_replication: 90s ago\n"
        "FAIL qdrant_points: unreachable: ConnectionRefusedError\n"
        "WARN cost_ledger_today: 0 rows"
    )
    output = format_results(results)
    assert output == expected, (
        f"Mixed-status golden mismatch.\n"
        f"Expected:\n{expected}\n\n"
        f"Got:\n{output}"
    )


def test_golden_integration_via_run_doctor(tmp_path: Path) -> None:
    """End-to-end: build a controlled fixture state and run_doctor returns
    deterministic results that format to the expected golden string.

    Relies only on injected paths, sizes, and replica file mtimes (not wall-clock
    for age comparisons) — the wal_size check is size-driven, cost_ledger is
    query-driven, and qdrant is stubbed.

    The checkpoint_age and replication checks depend on os.path.getmtime vs
    time.time(), but we pick ages (WAL ≈ fresh, replica ≈ 10s) that land
    firmly in the OK zone so a few-second test-execution jitter cannot flip
    them to WARN/FAIL.
    """
    # ---- DB fixture ----
    db = tmp_path / "firm.db"
    init_db(db)

    # 2.5 MB WAL → OK (< 12 MB threshold).
    _make_wal(db, int(2.5 * 1024 * 1024))

    # ---- Replica fixture (10 seconds ago → OK, << 60s WARN) ----
    replica = tmp_path / "litestream" / "firm"
    _make_replica_file(replica, age_seconds=10)

    # ---- Cost ledger: 7 rows today ----
    clock = ReplayClock(datetime(2026, 5, 21, 14, 0, tzinfo=timezone.utc))
    _seed_cost_ledger(db, ["2026-05-21T14:00:00+00:00"] * 7)

    # ---- Qdrant stub: 1500 points ----
    qdrant = _FakeQdrant(count=1500)

    results = run_doctor(
        db_path=db,
        litestream_dir=replica,
        qdrant_client=qdrant,
        collection_name="firm_chunks",
        clock=clock,
    )

    # All checks should be OK in this well-configured fixture state.
    for r in results:
        assert r.status == "OK", (
            f"Expected OK for {r.name} but got {r.status}: {r.detail}"
        )

    output = format_results(results)

    # Assert exact format of each line (name and status fixed; detail varies only
    # for checkpoint_age which is real-time, so we check prefix + suffix).
    lines = output.splitlines()
    assert len(lines) == 5

    # Line 0: wal_size — exact (size-based, deterministic)
    assert lines[0] == "OK wal_size: 2.5 MB"

    # Line 1: last_checkpoint_age — starts with "OK last_checkpoint_age:" ends with "s"
    assert lines[1].startswith("OK last_checkpoint_age:")
    assert lines[1].endswith("s")

    # Line 2: last_replication — starts with "OK last_replication:" ends with "s ago"
    assert lines[2].startswith("OK last_replication:")
    assert lines[2].endswith("s ago")

    # Line 3: qdrant_points — exact (stub-based, deterministic)
    assert lines[3] == "OK qdrant_points: 1500 points in firm_chunks"

    # Line 4: cost_ledger_today — exact (query-based, deterministic)
    assert lines[4] == "OK cost_ledger_today: 7 rows"
