"""Tests for the ``cost_ledger`` table + writer (Plan 3 T09).

Pins the schema (table + two indices on ``decision_id`` and ``created_at``)
and the append-only writer convention from
:func:`firm.db.cost_ledger.write_cost_ledger_row`:

* live row -> ``input_tokens`` / ``output_tokens`` populated,
  ``cached_tokens`` NULL, ``cost_usd > 0`` (per rate card).
* cached row -> ``input_tokens`` / ``output_tokens`` NULL,
  ``cached_tokens`` populated, ``cost_usd == 0.0``.
* writer is append-only -> two writes with the same business key produce
  two distinct rows (no UPSERT semantics).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from firm.core.clock import ReplayClock
from firm.db.connection import get_conn
from firm.db.cost_ledger import write_cost_ledger_row
from firm.db.migrations import init_db


def _init(tmp_path: Path) -> Path:
    db = tmp_path / "t09.db"
    init_db(db)
    return db


# ---------------------------------------------------------------------------
# 1. Schema — cost_ledger table created with all 9 columns
# ---------------------------------------------------------------------------


def test_schema_creates_cost_ledger_table(tmp_path: Path) -> None:
    db = _init(tmp_path)
    conn = get_conn(db)
    row = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='cost_ledger'"
    ).fetchone()
    assert row is not None, "cost_ledger table not created"

    # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk).
    cols = {
        r["name"]: r["type"]
        for r in conn.execute("PRAGMA table_info(cost_ledger)")
    }
    # All nine spec columns must be present.
    expected = {
        "id": "INTEGER",
        "decision_id": "TEXT",
        "agent": "TEXT",
        "model": "TEXT",
        "input_tokens": "INTEGER",
        "output_tokens": "INTEGER",
        "cached_tokens": "INTEGER",
        "cost_usd": "REAL",
        "created_at": "TEXT",
    }
    assert cols == expected, f"column mismatch: got {cols}"


# ---------------------------------------------------------------------------
# 2. Schema — required indices on decision_id and created_at exist
# ---------------------------------------------------------------------------


def test_schema_creates_required_indices(tmp_path: Path) -> None:
    """T09 spec literally: "Test asserts schema indices on decision_id and created_at"."""
    db = _init(tmp_path)
    conn = get_conn(db)
    names = {
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='cost_ledger'"
        )
    }
    assert "idx_cost_ledger_decision_id" in names, (
        f"missing decision_id index; saw: {sorted(names)}"
    )
    assert "idx_cost_ledger_created_at" in names, (
        f"missing created_at index; saw: {sorted(names)}"
    )


# ---------------------------------------------------------------------------
# 3. Writer — live row stores tokens + cost, NULL cached_tokens
# ---------------------------------------------------------------------------


def test_writer_inserts_live_row(tmp_path: Path) -> None:
    db = _init(tmp_path)
    clock = ReplayClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))

    write_cost_ledger_row(
        db_path=db,
        decision_id="dec-001",
        agent="research",
        model="claude-sonnet-4-6",
        input_tokens=1200,
        output_tokens=800,
        cached_tokens=None,
        cost_usd=0.015,
        clock=clock,
    )

    conn = get_conn(db)
    rows = [dict(r) for r in conn.execute("SELECT * FROM cost_ledger")]
    assert len(rows) == 1
    row = rows[0]
    assert row["decision_id"] == "dec-001"
    assert row["agent"] == "research"
    assert row["model"] == "claude-sonnet-4-6"
    assert row["input_tokens"] == 1200
    assert row["output_tokens"] == 800
    assert row["cached_tokens"] is None
    assert row["cost_usd"] == 0.015
    # Timestamp is the clock's ISO-8601 string (tz-aware UTC).
    assert row["created_at"] == "2026-05-21T12:00:00+00:00"


# ---------------------------------------------------------------------------
# 4. Writer — cached row stores cached_tokens + cost=0, NULL input/output
# ---------------------------------------------------------------------------


def test_writer_inserts_cached_row(tmp_path: Path) -> None:
    db = _init(tmp_path)
    clock = ReplayClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))

    write_cost_ledger_row(
        db_path=db,
        decision_id="dec-002",
        agent="pm",
        model="claude-haiku-4-5",
        input_tokens=None,
        output_tokens=None,
        cached_tokens=150,
        cost_usd=0.0,
        clock=clock,
    )

    conn = get_conn(db)
    rows = [dict(r) for r in conn.execute("SELECT * FROM cost_ledger")]
    assert len(rows) == 1
    row = rows[0]
    assert row["input_tokens"] is None
    assert row["output_tokens"] is None
    assert row["cached_tokens"] == 150
    assert row["cost_usd"] == 0.0
    assert row["agent"] == "pm"
    assert row["model"] == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# 5. Writer — append-only (two identical writes -> two rows, no upsert)
# ---------------------------------------------------------------------------


def test_writer_appends_not_upserts(tmp_path: Path) -> None:
    """Append-only semantics: re-calling the writer with the same business key
    produces a NEW row (autoincrement id) rather than updating the existing
    one. Required by the spec's "Append-only" wording.
    """
    db = _init(tmp_path)
    clock = ReplayClock(datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc))

    def _write() -> None:
        write_cost_ledger_row(
            db_path=db,
            decision_id="dec-dup",
            agent="research",
            model="claude-sonnet-4-6",
            input_tokens=10,
            output_tokens=5,
            cached_tokens=None,
            cost_usd=0.001,
            clock=clock,
        )

    _write()
    _write()

    conn = get_conn(db)
    rows = [dict(r) for r in conn.execute("SELECT * FROM cost_ledger ORDER BY id")]
    assert len(rows) == 2, f"expected 2 rows (append-only), got {rows}"
    # Distinct autoincrement ids — confirms INSERT (not REPLACE).
    assert rows[0]["id"] != rows[1]["id"]
    # Both carry the same business payload.
    assert rows[0]["decision_id"] == rows[1]["decision_id"] == "dec-dup"
