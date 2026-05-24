"""Tests for ``scripts/hydrate_sample_db.py``.

Covers the two cases enumerated in PLAN_reports_overhaul.md T6:
    1. Hydrates 2024-03-13 fixture into a tmp DB with the right row counts.
    2. Hydrates all three committed dates (counts match the JSONL line counts).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "hydrate_sample_db.py"
SAMPLE_RUNS = REPO_ROOT / "sample_runs"


def _load_module():
    """Import ``scripts/hydrate_sample_db.py`` as a module."""
    spec = importlib.util.spec_from_file_location(
        "hydrate_sample_db", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hydrate_sample_db"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load_module()


def _seed_date_dir(tmp_root: Path, date: str, copy_from: Path) -> Path:
    """Mirror sample_runs/<date>/* into tmp_root/<date>/."""
    d = tmp_root / date
    d.mkdir(parents=True, exist_ok=True)
    for fname in ("decisions.jsonl", "trace.jsonl"):
        src = copy_from / fname
        if src.exists():
            shutil.copy2(src, d / fname)
    return d


def _count_rows(db_path: Path, table: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
        return int(cur.fetchone()[0])
    finally:
        conn.close()


def _count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def test_hydrates_2024_03_13(tmp_path: Path, mod) -> None:
    src = SAMPLE_RUNS / "2024-03-13"
    assert src.is_dir(), "committed fixture missing"
    _seed_date_dir(tmp_path, "2024-03-13", copy_from=src)
    out_db = tmp_path / "firm.db"

    rc = mod.main(
        [
            "--sample-runs-root",
            str(tmp_path),
            "--date",
            "2024-03-13",
            "--out",
            str(out_db),
        ]
    )
    assert rc == 0
    assert out_db.exists()

    expected_decisions = _count_jsonl_lines(src / "decisions.jsonl")
    assert _count_rows(out_db, "decisions") == expected_decisions
    # The 2024-03-13 fixture has 2 decisions.
    assert expected_decisions == 2

    # The 2024-03-13 fixture has 6 llm.call spans (sonnet × 5 + haiku × 1).
    assert _count_rows(out_db, "cost_ledger") == 6

    # Positions seeded from _POSITIONS_BY_DATE: AAPL only.
    assert _count_rows(out_db, "positions") == 1


def test_citations_count_matches_trace_spans(tmp_path: Path, mod) -> None:
    """``decisions.citations`` JSON length must equal the sum of
    ``citations`` counters across matching ``llm.call`` spans.

    This is the only contract the HTML report relies on; the placeholder
    objects in the JSON have no other meaning.
    """
    date = "2024-03-13"
    src = SAMPLE_RUNS / date
    assert src.is_dir(), "committed fixture missing"
    _seed_date_dir(tmp_path, date, copy_from=src)
    out_db = tmp_path / "firm.db"

    rc = mod.main(
        [
            "--sample-runs-root",
            str(tmp_path),
            "--date",
            date,
            "--out",
            str(out_db),
        ]
    )
    assert rc == 0

    # Re-derive the expected counts directly from the trace JSONL.
    expected: dict[str, int] = {}
    with (src / "trace.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            span = json.loads(line)
            if span.get("operation") != "llm.call":
                continue
            did = span.get("decision_id") or ""
            if not did:
                continue
            expected[did] = expected.get(did, 0) + int(span.get("citations") or 0)

    # Compare against what we actually wrote into the DB.
    conn = sqlite3.connect(str(out_db))
    try:
        rows = conn.execute("SELECT id, citations FROM decisions").fetchall()
    finally:
        conn.close()

    assert rows, "decisions table should be non-empty"
    for decision_id, citations_json in rows:
        actual = len(json.loads(citations_json))
        assert actual == expected.get(decision_id, 0), (
            f"citation count mismatch for {decision_id}: "
            f"DB length={actual} expected={expected.get(decision_id, 0)}"
        )


def test_hydrates_all_three_dates(tmp_path: Path, mod) -> None:
    dates = ("2024-03-13", "2024-08-07", "2023-11-08")
    for date in dates:
        src = SAMPLE_RUNS / date
        assert src.is_dir(), f"committed fixture missing for {date}"
        date_root = tmp_path / date
        date_root.mkdir(exist_ok=True)
        _seed_date_dir(date_root, date, copy_from=src)
        out_db = tmp_path / f"{date}.db"

        rc = mod.main(
            [
                "--sample-runs-root",
                str(date_root),
                "--date",
                date,
                "--out",
                str(out_db),
            ]
        )
        assert rc == 0
        assert out_db.exists()

        expected_decisions = _count_jsonl_lines(src / "decisions.jsonl")
        assert _count_rows(out_db, "decisions") == expected_decisions, date
