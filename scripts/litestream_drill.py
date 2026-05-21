#!/usr/bin/env python
"""T23: PIT restore drill.

Generates a small SQLite DB, replicates with litestream for a few seconds,
restores into a temp DB, and asserts row counts match.  Also asserts that
data/firm.db-wal (if present) is under the 16 MB ceiling from
config/litestream.yml — catches a paused-but-undetected replicator.

Exits 0 on success or graceful skip.  Non-zero only on a real failure
(restore-mismatch, oversized WAL).

Usage:
  python scripts/litestream_drill.py [--force] [--db PATH]

Environment:
  FIRM_LITESTREAM_DRILL_SKIP_REPLICATE=1  Skip the replicate-restore cycle
      (still runs the WAL size check).  Used by unit tests to isolate the
      WAL guard without needing Docker.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WAL_CEILING_BYTES = 16 * 1024 * 1024  # 16 MB — mirrors config/litestream.yml
LITESTREAM_IMAGE = "litestream/litestream:0.3.13"
REPLICATE_SLEEP_SECS = 5  # seconds to let litestream flush before killing it

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _warn(msg: str) -> None:
    print(msg, file=sys.stderr)


def _litestream_runner() -> tuple[str, list[str]] | None:
    """Return (kind, argv_prefix) for a working litestream runner, or None.

    ``kind`` is "native" or "docker".
    ``argv_prefix`` is only populated for the native case; docker invocations
    build their full argv per-call because they need a volume mount.
    """
    if shutil.which("litestream"):
        return ("native", ["litestream"])
    if shutil.which("docker"):
        return ("docker", [])
    return None


def _check_wal_size(cwd: Path) -> bool:
    """Assert data/firm.db-wal is under the ceiling.

    Returns True (pass/vacuous) or prints an error and returns False.
    """
    wal = cwd / "data" / "firm.db-wal"
    if not wal.exists():
        return True  # vacuously true — no WAL file present
    size = os.path.getsize(wal)
    if size < WAL_CEILING_BYTES:
        return True
    _warn(
        f"ERROR: WAL size check failed. "
        f"data/firm.db-wal is {size:,} bytes "
        f"(>= {WAL_CEILING_BYTES:,} byte ceiling). "
        "The litestream replicator may be paused or crashed."
    )
    return False


def _seed_db(db_path: Path) -> dict[str, int]:
    """Create a tiny SQLite DB with a few tables and known row counts.

    Returns {table_name: row_count}.
    """
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        "CREATE TABLE IF NOT EXISTS drill_alpha (id INTEGER PRIMARY KEY, val TEXT)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS drill_beta  (id INTEGER PRIMARY KEY, num REAL)"
    )
    con.executemany(
        "INSERT INTO drill_alpha (val) VALUES (?)",
        [("foo",), ("bar",), ("baz",)],
    )
    con.executemany(
        "INSERT INTO drill_beta  (num) VALUES (?)",
        [(1.1,), (2.2,), (3.3,), (4.4,)],
    )
    con.commit()
    counts = {
        "drill_alpha": con.execute("SELECT COUNT(*) FROM drill_alpha").fetchone()[0],
        "drill_beta": con.execute("SELECT COUNT(*) FROM drill_beta").fetchone()[0],
    }
    con.close()
    return counts


def _replicate_native(
    litestream_bin: str,
    db_path: Path,
    replica_dir: Path,
) -> None:
    """Run ``litestream replicate`` natively for a short window then stop it."""
    cfg_content = (
        f"dbs:\n"
        f"  - path: {db_path}\n"
        f"    replicas:\n"
        f"      - type: file\n"
        f"        path: {replica_dir}\n"
        f"        sync-interval: 1s\n"
    )
    cfg_path = replica_dir.parent / "litestream_drill.yml"
    cfg_path.write_text(cfg_content, encoding="utf-8")

    proc = subprocess.Popen(
        [litestream_bin, "replicate", "-config", str(cfg_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(REPLICATE_SLEEP_SECS)
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _restore_native(
    litestream_bin: str,
    replica_dir: Path,
    source_db: Path,
    out_db: Path,
) -> None:
    """Restore from a file replica using native litestream."""
    cfg_content = (
        f"dbs:\n"
        f"  - path: {source_db}\n"
        f"    replicas:\n"
        f"      - type: file\n"
        f"        path: {replica_dir}\n"
    )
    cfg_path = replica_dir.parent / "litestream_restore.yml"
    cfg_path.write_text(cfg_content, encoding="utf-8")

    subprocess.run(
        [litestream_bin, "restore", "-config", str(cfg_path), "-o", str(out_db), str(source_db)],
        check=True,
    )


def _replicate_docker(
    db_path: Path,
    replica_dir: Path,
    tmp_dir: Path,
) -> None:
    """Run ``litestream replicate`` via Docker for a short window then stop."""
    cfg_content = (
        "dbs:\n"
        "  - path: /data/drill.db\n"
        "    replicas:\n"
        "      - type: file\n"
        "        path: /data/replica\n"
        "        sync-interval: 1s\n"
    )
    cfg_path = tmp_dir / "litestream_drill.yml"
    cfg_path.write_text(cfg_content, encoding="utf-8")

    # db_path is already inside tmp_dir (tmp_dir/drill.db), so no copy needed.
    # The entire tmp_dir is mounted as /data inside the container.

    proc = subprocess.Popen(
        [
            "docker", "run", "--rm",
            "-v", f"{tmp_dir}:/data",
            LITESTREAM_IMAGE,
            "replicate",
            "-config", "/data/litestream_drill.yml",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(REPLICATE_SLEEP_SECS)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()

    # replica_dir is tmp_dir/replica — Docker wrote into it directly via the mount.


def _restore_docker(
    tmp_dir: Path,
    out_db: Path,
) -> None:
    """Restore from /data/replica inside a Docker container.

    litestream v0.3.13 requires a config file even for ``restore``.
    We reuse the same config that was written by ``_replicate_docker``.
    """
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{tmp_dir}:/data",
            LITESTREAM_IMAGE,
            "restore",
            "-config", "/data/litestream_drill.yml",
            "-o", "/data/firm.restored.db",
            "/data/drill.db",
        ],
        check=True,
    )
    restored_in_tmp = tmp_dir / "firm.restored.db"
    if restored_in_tmp.exists():
        shutil.copy2(str(restored_in_tmp), str(out_db))


def _count_rows(db_path: Path, tables: list[str]) -> dict[str, int]:
    """Return {table: row_count} for the given tables in db_path."""
    con = sqlite3.connect(str(db_path))
    counts: dict[str, int] = {}
    for tbl in tables:
        counts[tbl] = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]  # noqa: S608
    con.close()
    return counts


# ---------------------------------------------------------------------------
# Main drill
# ---------------------------------------------------------------------------


def run_drill(force: bool, project_root: Path) -> int:
    """Execute the full PIT restore drill.  Returns exit code (0=pass).

    The WAL size check runs relative to ``Path.cwd()`` so that running the
    script from the project root (the normal operator case) inspects
    ``./data/firm.db-wal``.  Unit tests set cwd to a tmp_path and plant a
    synthetic WAL there, which is how the test exercises the guard without
    touching the real repo's data/ directory.
    """

    # -- Step 1: WAL size check (always runs, no docker required) ------------
    wal_ok = _check_wal_size(Path.cwd())
    if not wal_ok:
        return 1

    # -- Skip switch for unit tests ------------------------------------------
    if os.environ.get("FIRM_LITESTREAM_DRILL_SKIP_REPLICATE") == "1":
        _warn("SKIPPED: replicate-restore cycle (FIRM_LITESTREAM_DRILL_SKIP_REPLICATE=1)")
        return 0

    # -- Check runner availability -------------------------------------------
    runner = _litestream_runner()
    if runner is None:
        _warn(
            "SKIPPED: litestream drill - neither a 'litestream' binary nor 'docker'"
            " is available on PATH. The WAL size check above still ran."
        )
        return 0

    kind, argv_prefix = runner

    # -- Guard: refuse to touch production data/firm.db ----------------------
    prod_db = project_root / "data" / "firm.db"
    if prod_db.exists() and not force:
        _warn(
            f"ERROR: {prod_db} exists. Pass --force to acknowledge that this"
            " drill uses a synthetic DB and will NOT touch production data."
        )
        return 1

    # -- Run replicate->restore cycle in a temp dir --------------------------
    with tempfile.TemporaryDirectory(prefix="litestream_drill_") as tmp_str:
        tmp_dir = Path(tmp_str)
        drill_db = tmp_dir / "drill.db"
        replica_dir = tmp_dir / "replica"
        replica_dir.mkdir()
        restored_db = tmp_dir / "restored.db"

        print(f"Seeding drill DB at {drill_db} ...")
        expected_counts = _seed_db(drill_db)
        print(f"  Rows seeded: {expected_counts}")

        print(f"Replicating via {kind} for {REPLICATE_SLEEP_SECS}s ...")
        if kind == "native":
            bin_path = argv_prefix[0]
            _replicate_native(bin_path, drill_db, replica_dir)
            _restore_native(bin_path, replica_dir, drill_db, restored_db)
        else:
            # Docker: tmp_dir is the shared volume root
            _replicate_docker(drill_db, replica_dir, tmp_dir)
            _restore_docker(tmp_dir, restored_db)

        if not restored_db.exists():
            _warn("ERROR: Restore failed - restored DB not found after the drill.")
            return 1

        print(f"Restored DB at {restored_db}; comparing row counts ...")
        actual_counts = _count_rows(restored_db, list(expected_counts.keys()))
        print(f"  Expected: {expected_counts}")
        print(f"  Actual:   {actual_counts}")

        if actual_counts != expected_counts:
            _warn(
                f"ERROR: Row count mismatch after restore. "
                f"Expected {expected_counts}, got {actual_counts}."
            )
            return 1

    print("Drill passed. WAL guard OK; replicate->restore round-trip matches.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Acknowledge that the drill uses a synthetic DB (required if data/firm.db exists).",
    )
    p.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="Unused; reserved for future injection of a custom source DB path.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Project root is two levels up from this script.
    root = Path(__file__).resolve().parent.parent
    sys.exit(run_drill(force=args.force, project_root=root))
