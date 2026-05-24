"""Tests for ``scripts/build_sample_run_readme.py``.

Covers the five cases enumerated in PLAN_reports_overhaul.md T5:
    1. Runs successfully against the committed 2024-03-13 fixture.
    2. Idempotent: running twice produces byte-identical output.
    3. Handles a missing trace.jsonl gracefully.
    4. Falls back to ``unknown`` regime for an unrecognised date.
    5. Escapes pipe characters in rationale text.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_sample_run_readme.py"
SAMPLE_RUNS = REPO_ROOT / "sample_runs"


def _load_main():
    """Import ``scripts/build_sample_run_readme.py`` as a module."""
    spec = importlib.util.spec_from_file_location(
        "build_sample_run_readme", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_sample_run_readme"] = mod
    spec.loader.exec_module(mod)
    return mod.main


@pytest.fixture
def main():
    return _load_main()


def _seed_date_dir(
    tmp_root: Path,
    date: str,
    decisions: list[dict] | None = None,
    spans: list[dict] | None = None,
    copy_from: Path | None = None,
) -> Path:
    """Create ``tmp_root/<date>/`` and seed decisions/trace jsonl files."""
    d = tmp_root / date
    d.mkdir(parents=True, exist_ok=True)
    if copy_from is not None:
        for fname in ("decisions.jsonl", "trace.jsonl"):
            src = copy_from / fname
            if src.exists():
                shutil.copy2(src, d / fname)
        return d
    if decisions is not None:
        with (d / "decisions.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for row in decisions:
                f.write(json.dumps(row) + "\n")
    if spans is not None:
        with (d / "trace.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for row in spans:
                f.write(json.dumps(row) + "\n")
    return d


def test_runs_for_2024_03_13(tmp_path: Path, main) -> None:
    src = SAMPLE_RUNS / "2024-03-13"
    assert src.is_dir(), "committed fixture missing"
    _seed_date_dir(tmp_path, "2024-03-13", copy_from=src)

    rc = main(["--date", "2024-03-13", "--sample-runs-root", str(tmp_path)])
    assert rc == 0

    out = tmp_path / "2024-03-13" / "README.md"
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    # Sanity-check key sections.
    assert text.startswith("# Sample run — 2024-03-13\n")
    assert "## What this day shows" in text
    assert "earnings_heavy" in text
    assert "## Decisions" in text
    assert "| ts | action | ticker | shares | conf |" in text
    assert "## Walking one trade" in text
    assert "## Bundle" in text
    # Highest-cited decision in the fixture is dec-buy-1 (4 citations on its
    # llm.call span; other spans are not counted to avoid double-counting).
    assert "`dec-buy-1`" in text
    # HOLD row from decisions.jsonl is present.
    assert "| HOLD |" in text
    # BUY row for AAPL is present.
    assert "| BUY | AAPL | 100 |" in text
    # Trailing newline.
    assert text.endswith("\n")


def test_idempotent(tmp_path: Path, main) -> None:
    src = SAMPLE_RUNS / "2024-03-13"
    _seed_date_dir(tmp_path, "2024-03-13", copy_from=src)

    assert main(["--date", "2024-03-13", "--sample-runs-root", str(tmp_path)]) == 0
    first = (tmp_path / "2024-03-13" / "README.md").read_bytes()

    assert main(["--date", "2024-03-13", "--sample-runs-root", str(tmp_path)]) == 0
    second = (tmp_path / "2024-03-13" / "README.md").read_bytes()

    assert first == second, "output must be byte-identical on a second run"


def test_handles_missing_trace(tmp_path: Path, main) -> None:
    # Seed decisions.jsonl only.
    decisions = [
        {
            "ts": "2024-03-13T14:30:00+00:00",
            "research_decision": {
                "id": "dec-buy-1",
                "action": "BUY",
                "payload": {"kind": "buy", "ticker": "AAPL", "shares": "100"},
                "rationale": "Strong earnings momentum",
                "confidence": 0.85,
                "failure_mode": None,
            },
            "trace_id": "00000000000000000000000000000001",
        }
    ]
    _seed_date_dir(tmp_path, "2024-03-13", decisions=decisions)

    rc = main(["--date", "2024-03-13", "--sample-runs-root", str(tmp_path)])
    assert rc == 0

    text = (tmp_path / "2024-03-13" / "README.md").read_text(encoding="utf-8")
    assert "(no trace data available for this run)" in text


def test_regime_unknown_for_new_date(tmp_path: Path, main) -> None:
    decisions = [
        {
            "ts": "2099-01-01T10:00:00+00:00",
            "research_decision": {
                "id": "dec-hold-1",
                "action": "HOLD",
                "payload": {"kind": "hold", "reason": "speculative"},
                "rationale": "Nothing to do",
                "confidence": 0.5,
                "failure_mode": None,
            },
            "trace_id": "20990101000000000000000000000001",
        }
    ]
    _seed_date_dir(tmp_path, "2099-01-01", decisions=decisions)

    rc = main(["--date", "2099-01-01", "--sample-runs-root", str(tmp_path)])
    assert rc == 0

    text = (tmp_path / "2099-01-01" / "README.md").read_text(encoding="utf-8")
    assert "**Regime tag:** `unknown`" in text


def test_pipe_in_rationale_escaped(tmp_path: Path, main) -> None:
    decisions = [
        {
            "ts": "2024-03-13T14:30:00+00:00",
            "research_decision": {
                "id": "dec-buy-1",
                "action": "BUY",
                "payload": {"kind": "buy", "ticker": "AAPL", "shares": "100"},
                "rationale": "foo|bar",
                "confidence": 0.85,
                "failure_mode": None,
            },
            "trace_id": "00000000000000000000000000000001",
        }
    ]
    _seed_date_dir(tmp_path, "2024-03-13", decisions=decisions)

    rc = main(["--date", "2024-03-13", "--sample-runs-root", str(tmp_path)])
    assert rc == 0

    text = (tmp_path / "2024-03-13" / "README.md").read_text(encoding="utf-8")
    assert r"foo\|bar" in text
    # The literal unescaped form must not appear in the rationale cell.
    # (It could legitimately appear in table-divider rows, so check for the
    # specific token surrounded by spaces typical of the rationale column.)
    assert " foo|bar " not in text
