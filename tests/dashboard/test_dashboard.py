"""Unit tests for ``firm.dashboard`` pure-Python path helpers.

Streamlit rendering is intentionally *not* exercised here (its testing story
is heavy). We cover only the path-resolution helpers that decide which
``data/reports/`` vs ``sample_runs/`` directory the dashboard reads from.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("pandas")

import firm.dashboard as dashboard  # noqa: E402


# ---------------------------------------------------------------------------
# _list_available_dates
# ---------------------------------------------------------------------------


def test_list_available_dates_prefers_live_over_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live ``data/reports/<date>`` shadows ``sample_runs/<date>``; sort DESC."""
    reports_root = tmp_path / "reports"
    samples_root = tmp_path / "samples"
    (reports_root / "2024-03-13").mkdir(parents=True)
    (samples_root / "2024-03-13").mkdir(parents=True)  # shadowed
    (samples_root / "2023-11-08").mkdir(parents=True)

    monkeypatch.setattr(dashboard, "REPORTS_ROOT", reports_root)
    monkeypatch.setattr(dashboard, "SAMPLE_RUNS_ROOT", samples_root)

    result = dashboard._list_available_dates()

    assert result == [
        ("2024-03-13", reports_root),
        ("2023-11-08", samples_root),
    ]


def test_list_available_dates_ignores_non_date_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Directories whose name is not YYYY-MM-DD are excluded."""
    samples_root = tmp_path / "samples"
    (samples_root / "2024-03-13").mkdir(parents=True)
    (samples_root / "notes").mkdir(parents=True)
    (samples_root / "2024-13-99").mkdir(parents=True)  # invalid date

    monkeypatch.setattr(dashboard, "REPORTS_ROOT", tmp_path / "absent")
    monkeypatch.setattr(dashboard, "SAMPLE_RUNS_ROOT", samples_root)

    result = dashboard._list_available_dates()
    assert result == [("2024-03-13", samples_root)]


# ---------------------------------------------------------------------------
# _resolve_bundle_path
# ---------------------------------------------------------------------------


def test_resolve_bundle_path_missing_returns_none(tmp_path: Path) -> None:
    """No file present -> None."""
    assert (
        dashboard._resolve_bundle_path("2024-03-13", tmp_path, "daily_report.html")
        is None
    )


def test_resolve_bundle_path_returns_existing(tmp_path: Path) -> None:
    """Existing file -> the resolved Path."""
    date_dir = tmp_path / "2024-03-13"
    date_dir.mkdir()
    target = date_dir / "daily_report.html"
    target.write_text("<html></html>", encoding="utf-8")

    result = dashboard._resolve_bundle_path(
        "2024-03-13", tmp_path, "daily_report.html"
    )
    assert result == target


# ---------------------------------------------------------------------------
# _resolve_trace_path
# ---------------------------------------------------------------------------


def test_resolve_trace_path_prefers_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live ``data/traces/<date>/run-*.jsonl`` shadows the sample fallback."""
    traces_root = tmp_path / "traces"
    samples_root = tmp_path / "samples"
    live_dir = traces_root / "2024-03-13"
    live_dir.mkdir(parents=True)
    live_run = live_dir / "run-001.jsonl"
    live_run.write_text("{}\n", encoding="utf-8")

    sample_dir = samples_root / "2024-03-13"
    sample_dir.mkdir(parents=True)
    (sample_dir / "trace.jsonl").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(dashboard, "TRACES_ROOT", traces_root)

    result = dashboard._resolve_trace_path("2024-03-13", samples_root)
    assert result == live_run


def test_resolve_trace_path_falls_back_to_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No live traces -> sample ``trace.jsonl``."""
    samples_root = tmp_path / "samples"
    sample_dir = samples_root / "2024-03-13"
    sample_dir.mkdir(parents=True)
    sample_trace = sample_dir / "trace.jsonl"
    sample_trace.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(dashboard, "TRACES_ROOT", tmp_path / "no-traces")

    result = dashboard._resolve_trace_path("2024-03-13", samples_root)
    assert result == sample_trace


def test_resolve_trace_path_returns_none_when_nothing_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No live, no sample -> None (not a crash)."""
    monkeypatch.setattr(dashboard, "TRACES_ROOT", tmp_path / "absent")
    assert dashboard._resolve_trace_path("2024-03-13", tmp_path) is None


# ---------------------------------------------------------------------------
# _load_trace_spans (lightweight JSONL filtering)
# ---------------------------------------------------------------------------


def test_load_trace_spans_filters_by_decision_id(tmp_path: Path) -> None:
    """Only spans with matching decision_id are kept, in file order."""
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"decision_id":"dec-1","operation":"a","agent":"x","duration_ms":1,'
        '"model":"","input_tokens":0,"output_tokens":0,"cost_usd":0,'
        '"citations":0,"status":"ok"}\n'
        '{"decision_id":"dec-2","operation":"b","agent":"y"}\n'
        '{"decision_id":"dec-1","operation":"c","agent":"x"}\n',
        encoding="utf-8",
    )
    spans = dashboard._load_trace_spans(trace, "dec-1")
    assert [s["operation"] for s in spans] == ["a", "c"]
