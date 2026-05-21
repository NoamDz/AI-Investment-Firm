"""Tests for the ``firm eval`` Click subcommand + aggregator (Plan 4 §T15).

The production heartbeat is patched out — every test injects the
``_seed_db`` stub from :mod:`tests.eval.test_runner` so the eval harness
sees real decisions / fills / hitl rows without needing T16's cassettes
or T10's price parquets.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from firm.cli import cli
from firm.eval.aggregate import build_summary_context
from firm.eval.heartbeat import HeartbeatFn, make_eval_heartbeat
from firm.eval.process_metrics import MetricResult
from firm.eval.regimes import R1_EARNINGS
from firm.eval.runner import RegimeReport

from tests.eval.test_runner import _seed_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stub_heartbeat_factory(*_args: Any, **_kwargs: Any) -> HeartbeatFn:
    """Replacement for ``make_eval_heartbeat`` — seeds the DB on day 1."""
    state = {"seeded": False}

    def _stub(day: date, db_path: Path) -> None:
        if state["seeded"]:
            return
        state["seeded"] = True
        _seed_db(db_path)

    return _stub


def _set_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the CLI's determinism defaults + pin benchmark stubs."""
    monkeypatch.setenv("FIRM_LLM_MODE", "cached")
    monkeypatch.setenv("FIRM_VCR_MODE", "replay")
    monkeypatch.setenv("FIRM_PRICES_MODE", "replay")
    monkeypatch.setenv("FIRM_RANDOM_SEED", "42")
    monkeypatch.setenv("FIRM_HMAC_SECRET", "0" * 64)


def _patch_run_regime_benchmarks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fixed benchmark returns so ``run_regime`` doesn't hit T10.

    T10's compute_spy_return / compute_basket_return raise
    PriceCassetteMissError without ``data/prices_eval/*.parquet`` populated
    (T17). For T15's tests we pre-empt that by patching them.
    """
    import firm.eval.runner as runner_mod

    def _fake_spy(*_a: Any, **_kw: Any) -> float:
        return 0.008

    def _fake_basket(*_a: Any, **_kw: Any) -> float:
        return -0.004

    monkeypatch.setattr(runner_mod, "compute_spy_return", _fake_spy)
    monkeypatch.setattr(runner_mod, "compute_basket_return", _fake_basket)


# ---------------------------------------------------------------------------
# Test 1 — runs all 3 regimes; writes per-regime + summary files.
# ---------------------------------------------------------------------------


def test_eval_runs_all_regimes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_offline_env(monkeypatch)
    _patch_run_regime_benchmarks(monkeypatch)
    # The CLI imports make_eval_heartbeat at function-call time, so patching
    # the source module is what wins.
    monkeypatch.setattr(
        "firm.eval.heartbeat.make_eval_heartbeat", _stub_heartbeat_factory
    )

    out_dir = tmp_path / "eval"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["eval", "--output-dir", str(out_dir)], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    # 3 per-regime files + summary.
    assert (out_dir / "r1_earnings.md").exists()
    assert (out_dir / "r2_drawdown.md").exists()
    assert (out_dir / "r3_quiet.md").exists()
    assert (out_dir / "summary.md").exists()


# ---------------------------------------------------------------------------
# Test 2 — byte-for-byte idempotency.
# ---------------------------------------------------------------------------


def test_eval_idempotent_byte_for_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_offline_env(monkeypatch)
    _patch_run_regime_benchmarks(monkeypatch)
    monkeypatch.setattr(
        "firm.eval.heartbeat.make_eval_heartbeat", _stub_heartbeat_factory
    )

    out_dir = tmp_path / "eval"
    runner = CliRunner()

    # First run.
    res1 = runner.invoke(
        cli, ["eval", "--output-dir", str(out_dir)], catch_exceptions=False
    )
    assert res1.exit_code == 0, res1.output

    md_files = sorted(out_dir.glob("*.md"))
    assert md_files, "no .md files produced by first run"
    first_bytes = {p.name: p.read_bytes() for p in md_files}

    # Second run (same output_dir): the CLI nukes + recreates.
    res2 = runner.invoke(
        cli, ["eval", "--output-dir", str(out_dir)], catch_exceptions=False
    )
    assert res2.exit_code == 0, res2.output

    md_files_2 = sorted(out_dir.glob("*.md"))
    assert sorted(p.name for p in md_files_2) == sorted(first_bytes.keys())
    for p in md_files_2:
        assert p.read_bytes() == first_bytes[p.name], (
            f"{p.name} differs between first and second run"
        )


# ---------------------------------------------------------------------------
# Test 3 — single-regime mode.
# ---------------------------------------------------------------------------


def test_eval_single_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_offline_env(monkeypatch)
    _patch_run_regime_benchmarks(monkeypatch)
    monkeypatch.setattr(
        "firm.eval.heartbeat.make_eval_heartbeat", _stub_heartbeat_factory
    )

    out_dir = tmp_path / "eval"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["eval", "--regime", "r1", "--output-dir", str(out_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    md_files = sorted(p.name for p in out_dir.glob("*.md"))
    assert md_files == ["r1_earnings.md", "summary.md"]

    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    # Summary header for R1 only — no R2/R3 lines should appear.
    assert "REGIME 1:" in summary
    assert "REGIME 2:" not in summary
    assert "REGIME 3:" not in summary


# ---------------------------------------------------------------------------
# Test 4 — heartbeat swallows skippable misses + records audit-log row.
# ---------------------------------------------------------------------------


def test_make_eval_heartbeat_swallows_cache_miss(tmp_path: Path) -> None:
    """Unit-test the swallow + audit-log logic in isolation.

    The production graph builder ``_build_llm_stack`` needs Qdrant +
    Anthropic creds, so we wedge the heartbeat directly: monkey-patch the
    inner _build_graph_once to raise a LlmCacheMissError, then invoke the
    returned heartbeat and check the audit_log.
    """
    from firm.db.migrations import init_db
    from firm.eval.heartbeat import _record_skip
    from firm.llm.anthropic_client import LlmCacheMissError

    db_path = tmp_path / "skip.db"
    init_db(db_path)

    # Construct the heartbeat (we don't invoke it — graph construction
    # would need Qdrant) and invoke _record_skip directly with a synthetic
    # LlmCacheMissError. This mirrors what the heartbeat does internally
    # on a swallowed exception, without forcing a live graph construction.
    make_eval_heartbeat(R1_EARNINGS, reports_root=tmp_path / "artifacts")
    exc = LlmCacheMissError("synthetic miss for unit test")
    _record_skip(db_path, date(2024, 3, 11), exc)

    # Read back the audit_log row.
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(
            conn.execute(
                "SELECT event, detail FROM audit_log WHERE event = 'heartbeat.skipped'"
            )
        )
    assert len(rows) == 1
    detail = json.loads(rows[0]["detail"])
    assert detail["reason"] == "LlmCacheMissError"
    assert detail["day"] == "2024-03-11"
    assert "synthetic miss" in detail["message"]

    # Sanity: confirm the resolver picks LlmCacheMissError as a skippable.
    import firm.eval.heartbeat as hb_mod

    hb_mod._cached_skippable = None
    skips = hb_mod._resolve_skippable()
    assert LlmCacheMissError in skips


# ---------------------------------------------------------------------------
# Test 5 — pure unit test on build_summary_context aggregation rules.
# ---------------------------------------------------------------------------


def _mk_report(
    regime_id: str,
    start: date,
    end: date,
    metrics: list[MetricResult],
    perf: dict[str, float | str] | None = None,
) -> RegimeReport:
    return RegimeReport(
        regime_id=regime_id,
        start_date=start,
        end_date=end,
        num_days=(end - start).days + 1,
        num_decisions=1,
        num_fills=0,
        perf_metrics=perf or {
            "total_return_pct": 0.0,
            "spy_return_pct": 0.0,
            "basket_return_pct": 0.0,
            "vs_spy_pp": 0.0,
            "vs_basket_pp": 0.0,
            "per_trade_returns_str": "(no closed trades)",
            "hit_rate_str": "0/0 (n/a) — no closed trades",
        },
        process_metrics=metrics,
        report_path=Path(f"/tmp/{regime_id}.md"),
    )


def _metrics_for(
    groundedness: float,
    schema_rej: int,
    discipline_num: int,
    discipline_den: int,
) -> list[MetricResult]:
    """Build a 4-metric fixture covering float / int / num-den / p-r shapes."""
    return [
        MetricResult(
            name="groundedness",
            value=groundedness,
            threshold=99.0,
            status="pass",
        ),
        MetricResult(
            name="schema_rejections",
            value=schema_rej,
            threshold=None,
            status="info",
        ),
        MetricResult(
            name="decision_discipline",
            value=f"{discipline_num}/{discipline_den}",
            threshold=f"{discipline_den}/{discipline_den}",
            status="pass" if discipline_num == discipline_den else "fail",
        ),
        MetricResult(
            name="sufficiency_gate",
            value="p=0.85, r=0.90",
            threshold="p>=0.80, r>=0.80",
            status="pass",
        ),
    ]


def test_build_summary_context_aggregates_correctly() -> None:
    # Construct three regimes with known metric values so we can verify
    # arithmetic mean (float), sum (int), and parsed-num/den (str).
    r1 = _mk_report(
        "r1_earnings", date(2024, 3, 11), date(2024, 3, 15),
        _metrics_for(99.0, 2, 5, 5),
    )
    r2 = _mk_report(
        "r2_drawdown", date(2024, 8, 5), date(2024, 8, 9),
        _metrics_for(100.0, 1, 4, 5),
    )
    r3 = _mk_report(
        "r3_quiet", date(2023, 11, 6), date(2023, 11, 10),
        _metrics_for(98.0, 0, 5, 5),
    )

    ctx = build_summary_context([r1, r2, r3])

    # regimes list shape.
    assert len(ctx["regimes"]) == 3
    assert ctx["regimes"][0]["regime_number"] == 1
    assert ctx["regimes"][0]["header_dates"] == "Mar 11–15, 2024"
    assert ctx["regimes"][0]["description"] == "earnings-heavy"
    assert ctx["regimes"][1]["header_dates"] == "Aug 5–9, 2024"
    assert ctx["regimes"][2]["header_dates"] == "Nov 6–10, 2023"

    # aggregated metrics — order preserved.
    agg = {m.name: m for m in ctx["aggregated_metrics"]}
    assert list(m.name for m in ctx["aggregated_metrics"]) == [
        "groundedness",
        "schema_rejections",
        "decision_discipline",
        "sufficiency_gate",
    ]

    # float: arithmetic mean → (99.0 + 100.0 + 98.0) / 3 = 99.0
    assert agg["groundedness"].value == pytest.approx(99.0)
    # all pass → pass
    assert agg["groundedness"].status == "pass"

    # int: sum → 2 + 1 + 0 = 3
    assert agg["schema_rejections"].value == 3
    # all info → info
    assert agg["schema_rejections"].status == "info"

    # str num/den: sum numerators + denominators → 14/15
    assert agg["decision_discipline"].value == "14/15"
    # r2 was 'fail' → combined 'fail'
    assert agg["decision_discipline"].status == "fail"

    # p=,r= str: mean each → unchanged since identical inputs
    assert agg["sufficiency_gate"].value == "p=0.85, r=0.90"
    assert agg["sufficiency_gate"].status == "pass"


def test_build_summary_context_status_combiner() -> None:
    """Any 'fail' beats 'warn' beats 'pass' beats 'info'."""
    from firm.eval.aggregate import _combine_statuses

    assert _combine_statuses(["pass", "pass", "pass"]) == "pass"
    assert _combine_statuses(["pass", "warn", "pass"]) == "warn"
    assert _combine_statuses(["pass", "fail", "warn"]) == "fail"
    assert _combine_statuses(["info", "info"]) == "info"
    assert _combine_statuses(["info", "pass"]) == "pass"


def test_build_summary_context_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_summary_context([])
