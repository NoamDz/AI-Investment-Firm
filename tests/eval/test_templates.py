"""Tests for the eval report Jinja templates (Plan 4 §T14).

Covers ``firm/reports/templates/regime.md.j2`` (the per-regime detail doc
rendered by ``firm.eval.runner.run_regime``) and ``summary.md.j2`` (the
multi-regime cross-cut consumed by T15's ``firm eval --regime all``).

Three checks:

1. ``test_regime_template_matches_golden`` — byte-for-byte comparison
   against ``tests/eval/fixtures/golden_regime_report.md``. Pins the
   full §9.7 styled output. Any drift (whitespace, dash codepoints,
   metric labels, format specifiers, NOT MEASURED bullets) trips this.

2. ``test_regime_template_has_required_section_headers`` — keeps the
   T13 ``_REQUIRED_SECTION_HEADERS`` contract alive. Belt-and-braces
   alongside the golden file: the golden could be regenerated and lose
   a header silently; this list pins the headers explicitly.

3. ``test_summary_template_renders`` — loads ``summary.md.j2`` directly
   via a Jinja FileSystemLoader (no runner involvement) and asserts the
   section headers + per-regime headers + NOT MEASURED block render.
   T15 owns the production wiring, so the summary shape is allowed to
   evolve — no golden file here.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from firm.eval.regimes import R1_EARNINGS
from firm.eval.runner import run_regime
from tests.eval.test_runner import _stub_heartbeat_factory

_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "firm" / "reports" / "templates"
_GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "golden_regime_report.md"

# Mirror of ``tests.eval.test_runner._REQUIRED_SECTION_HEADERS``. Duplicated
# (rather than imported) so the contract is asserted independently — if
# someone weakens the runner-side list this test still fails.
# Keep in sync with tests/eval/test_runner.py:_REQUIRED_SECTION_HEADERS
_REQUIRED_SECTION_HEADERS = (
    "EVAL REPORT — Replay smoke test across 3 regimes",
    "REGIME",
    "Total return:",
    "vs SPY (primary):",
    "vs equal-weight basket:",
    "Per-trade returns:",
    "Hit rate:",
    "PROCESS METRICS (aggregated)",
    "NOT MEASURED",
)


def _run_with_stub(tmp_path: Path) -> bytes:
    """Render R1_EARNINGS with the T13 stub heartbeat and return raw bytes."""
    report = run_regime(
        R1_EARNINGS,
        output_dir=tmp_path,
        heartbeat=_stub_heartbeat_factory(),
        spy_return=0.008,
        basket_return=-0.004,
        final_marks={},
        # T12 wired real sufficiency measurement; pin the (1.0, 1.0) values
        # the golden fixture was generated with so this template test stays
        # focused on rendering rather than on the new measurement logic
        # (which has its own dedicated tests in tests/unit/test_sufficiency.py).
        sufficiency_precision=1.0,
        sufficiency_recall=1.0,
    )
    # ``read_bytes`` defeats any Windows CRLF translation that might
    # otherwise hide a line-ending drift in the golden comparison.
    return report.report_path.read_bytes()


def test_regime_template_matches_golden(tmp_path: Path) -> None:
    actual = _run_with_stub(tmp_path)
    expected = _GOLDEN_PATH.read_bytes()
    assert actual == expected, (
        "regime.md output drifted from golden fixture.\n"
        f"  actual length:   {len(actual)}\n"
        f"  expected length: {len(expected)}\n"
        "If the template intentionally changed, regenerate the fixture and "
        "eyeball the diff before committing."
    )


def test_regime_template_has_required_section_headers(tmp_path: Path) -> None:
    content = _run_with_stub(tmp_path).decode("utf-8")
    for header in _REQUIRED_SECTION_HEADERS:
        assert header in content, f"missing section header: {header!r}"


def test_summary_template_renders() -> None:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    template = env.get_template("summary.md.j2")

    # Minimal hand-built context — T15 owns the production wiring; this is
    # just the shape documented at the top of summary.md.j2.
    perf = {
        "total_return_pct": -1.2,
        "spy_return_pct": 0.8,
        "basket_return_pct": -0.4,
        "vs_spy_pp": -2.0,
        "vs_basket_pp": -0.8,
        "per_trade_returns_str": "+2.8%, -1.4%, +0.6%, -0.9%, -2.1%",
        "hit_rate_str": "2/5 (40%) — n=5, not statistically significant",
    }
    regimes = [
        {
            "regime_number": 1,
            "header_dates": "Mar 11–15, 2024",
            "description": "earnings-heavy",
            "perf": perf,
        },
        {
            "regime_number": 2,
            "header_dates": "Aug 5–9, 2024",
            "description": "post-Aug-5 sell-off",
            "perf": perf,
        },
        {
            "regime_number": 3,
            "header_dates": "Nov 6–10, 2023",
            "description": "low-volatility quiet",
            "perf": perf,
        },
    ]

    # Synthesise a minimal MetricResult-compatible list. The template only
    # touches ``.name`` and ``.value`` so a small duck-typed namespace is
    # sufficient and avoids tying this test to ``MetricResult``'s shape.
    class _M:
        def __init__(self, name: str, value: object) -> None:
            self.name = name
            self.value = value

    aggregated_metrics = [
        _M("groundedness", 99.5),
        _M("decision_discipline", "15/15"),
        _M("red_team_pass", "50/50"),
        _M("risk_policy_compliance", 0),
        _M("hitl_correctness", "12/12"),
        _M("failure_mode_coverage", "14/14"),
    ]

    rendered = template.render(
        regimes=regimes, aggregated_metrics=aggregated_metrics
    )

    # Banner + global sections.
    assert "EVAL REPORT — Replay smoke test across 3 regimes" in rendered
    assert "PROCESS METRICS (aggregated)" in rendered
    assert "NOT MEASURED" in rendered

    # Each regime gets its own numbered header line.
    assert "REGIME 1: Mar 11–15, 2024 (earnings-heavy)" in rendered
    assert "REGIME 2: Aug 5–9, 2024 (post-Aug-5 sell-off)" in rendered
    assert "REGIME 3: Nov 6–10, 2023 (low-volatility quiet)" in rendered

    # Metric labels survive the §9.5→§9.7 display-label remapping.
    assert "Groundedness:" in rendered
    assert "Decision discipline:" in rendered
    assert "Red-team pass:" in rendered
    assert "Privileged-action attempts:" in rendered
    assert "HITL correctness:" in rendered
    assert "FailureMode coverage:" in rendered

    # NOT MEASURED bullets — all 5 from spec §9.6.
    for bullet in (
        "Investment quality / alpha",
        "Generalization beyond 3 declared regimes",
        "Real-world fill quality",
        "Forward references inside chunks",
        "Long-horizon learning effects",
    ):
        assert bullet in rendered, f"missing NOT MEASURED bullet: {bullet!r}"
