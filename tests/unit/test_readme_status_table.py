"""Plan 4 T45 — README status table regex check.

Asserts every "Plan N: …" line in README.md is marked complete (`- [x]`).
Fails if any of the four Plan checkboxes regresses to `- [ ]`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_README = Path(__file__).resolve().parents[2] / "README.md"
_PLAN_ROW = re.compile(r"^- \[(?P<mark>[ x])\] Plan (?P<n>\d+):", re.MULTILINE)


def test_readme_status_table_marks_all_four_plans_done() -> None:
    text = _README.read_text(encoding="utf-8")
    rows = _PLAN_ROW.findall(text)
    assert len(rows) == 4, (
        f"README status table must contain exactly 4 'Plan N:' rows; "
        f"found {len(rows)}: {rows}"
    )
    unfinished = [n for mark, n in rows if mark != "x"]
    assert not unfinished, (
        f"Plan rows {unfinished} are not marked '- [x]' in README.md status table."
    )
