"""Cross-regime aggregator for the eval summary report (Plan 4 §T15).

Builds the render context for ``firm/reports/templates/summary.md.j2``
from a sequence of :class:`RegimeReport` instances. Aggregation rules:

* **float-valued metrics** (e.g. ``groundedness``, ``reversal_rate``):
  arithmetic mean across regimes, rounded to 1 d.p. so the summary
  output type matches what the regime template emits.
* **int-valued metrics** (e.g. ``risk_policy_compliance``,
  ``schema_rejections``): sum across regimes.
* **str-valued ``num/den`` metrics** (e.g. ``decision_discipline``,
  ``hitl_correctness``, ``red_team_pass``, ``failure_mode_coverage``,
  ``citation_diversity``): parse ``num/den``, sum numerators and
  denominators, recombine as ``f"{sum_num}/{sum_den}"``. If a value
  fails to parse, fall back to the first regime's value (defensive —
  the registered metric shapes are all ``num/den``-style except
  ``sufficiency_gate`` which we special-case below).
* **``sufficiency_gate``** is a ``str`` of shape ``"p=0.85, r=0.90"``;
  parse out p/r, mean them across regimes, re-emit in the same shape.

Status combination across regimes uses the strictest verdict:
  * any ``fail`` → ``fail``
  * else any ``warn`` → ``warn``
  * else all ``pass`` → ``pass``
  * else (e.g. all ``info``) → first regime's status

.. warning::
   The ``num/den`` aggregator sums numerators and denominators across
   regimes (``1/1 + 100/100 → 101/101``), which masks per-regime trial-
   count scale. A regime that runs 100 decisions and a regime that runs
   1 decision contribute proportionally to the combined ratio rather
   than equally — intentional, since the aggregated ratio represents
   the global pass rate, but it means a single high-volume regime can
   dominate the summary. Inspect per-regime reports for scale-sensitive
   judgments.

Ordering is fixed: the first regime's ``process_metrics`` ordering wins.
This is the spec §9.5 order produced by :func:`compute_all_metrics`, but
relying on the input order keeps the aggregator decoupled from that fact.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from firm.eval.process_metrics import MetricResult, MetricStatus
from firm.eval.runner import RegimeReport

# ---------------------------------------------------------------------------
# Status combiners (module-level so they're testable / introspectable).
# ---------------------------------------------------------------------------

# Priority order — earlier in this tuple beats later. ``info`` is the lowest
# priority so it never overrides a real verdict (any concrete pass/warn/fail
# from a later regime takes precedence).
_STATUS_PRIORITY: tuple[MetricStatus, ...] = ("fail", "warn", "pass", "info")


def _combine_statuses(statuses: Sequence[MetricStatus]) -> MetricStatus:
    """Return the strictest status from *statuses*.

    Spec §T15 rule: "pass if ALL regimes pass, else fail if any fail else
    warn". This is equivalent to picking by the ``_STATUS_PRIORITY``
    ordering above with one twist: an empty sequence is invalid (we always
    have at least one regime to aggregate).
    """
    if not statuses:
        raise ValueError("cannot combine empty status sequence")
    for candidate in _STATUS_PRIORITY:
        if candidate in statuses:
            return candidate
    # Unreachable — every MetricStatus is in _STATUS_PRIORITY.
    return statuses[0]


# ---------------------------------------------------------------------------
# Per-shape value combiners.
# ---------------------------------------------------------------------------

_NUM_DEN_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
_P_R_RE = re.compile(
    r"^\s*p\s*=\s*([0-9.]+)\s*,\s*r\s*=\s*([0-9.]+)\s*$"
)


def _is_p_r_format(value: str) -> bool:
    return _P_R_RE.match(value) is not None


def _aggregate_p_r(values: Sequence[str]) -> str:
    """Mean the p / r of a sufficiency_gate-style string."""
    ps: list[float] = []
    rs: list[float] = []
    for v in values:
        m = _P_R_RE.match(v)
        if not m:
            # Defensive fallback — return the first value verbatim if any
            # entry fails to parse. The aggregator never raises on a metric
            # the renderer can still print.
            return values[0]
        ps.append(float(m.group(1)))
        rs.append(float(m.group(2)))
    mean_p = sum(ps) / len(ps)
    mean_r = sum(rs) / len(rs)
    return f"p={mean_p:.2f}, r={mean_r:.2f}"


def _aggregate_num_den(values: Sequence[str]) -> str:
    """Parse each ``"num/den"`` value, sum numerators + denominators, recombine.

    On any parse failure, fall back to ``values[0]`` — this lets a single
    malformed value not crash the whole report. The renderer is robust to
    arbitrary str content in the value field.
    """
    nums: list[int] = []
    dens: list[int] = []
    for v in values:
        m = _NUM_DEN_RE.match(v)
        if not m:
            return values[0]
        nums.append(int(m.group(1)))
        dens.append(int(m.group(2)))
    return f"{sum(nums)}/{sum(dens)}"


def _aggregate_threshold_str(thresholds: Sequence[str | int | float | None]) -> Any:
    """Sum-style aggregation for ``num/den`` thresholds; preserve first if non-uniform."""
    # The threshold is preserved from regime 1 per the spec, but for num/den
    # we sum so ``threshold == value`` still holds when every regime passes.
    if not thresholds:
        return None
    first = thresholds[0]
    if isinstance(first, str):
        m_first = _NUM_DEN_RE.match(first)
        if m_first:
            den_total = 0
            for t in thresholds:
                if not isinstance(t, str):
                    return first
                m = _NUM_DEN_RE.match(t)
                if not m:
                    return first
                den_total += int(m.group(2))
            # The "all-pass" threshold for sum-style num/den metrics is
            # ``den_total/den_total`` to mirror the per-regime convention.
            return f"{den_total}/{den_total}"
    return first


# ---------------------------------------------------------------------------
# Per-metric aggregation orchestrator.
# ---------------------------------------------------------------------------


def _aggregate_one_metric(
    metric_name: str, per_regime: Sequence[MetricResult]
) -> MetricResult:
    """Combine N regime instances of the SAME metric into one MetricResult."""
    if not per_regime:
        raise ValueError(f"no instances supplied for metric {metric_name!r}")
    first = per_regime[0]
    statuses = [m.status for m in per_regime]
    combined_status = _combine_statuses(statuses)

    # Dispatch on the value type of the first instance. Mixed shapes across
    # regimes would be a bug upstream — the compute_* functions always return
    # a stable shape per metric.
    values = [m.value for m in per_regime]
    if isinstance(first.value, float):
        nums = [float(v) for v in values]
        mean = round(sum(nums) / len(nums), 1)
        return MetricResult(
            name=metric_name,
            value=mean,
            threshold=first.threshold,
            status=combined_status,
        )
    if isinstance(first.value, bool):
        # bool is a subclass of int in Python — keep before the int branch
        # so we don't silently coerce bool→int. Combine via logical-and,
        # stored as int(0/1) since MetricResult.value is int|float|str.
        combined_int = int(all(bool(v) for v in values))
        return MetricResult(
            name=metric_name,
            value=combined_int,
            threshold=first.threshold,
            status=combined_status,
        )
    if isinstance(first.value, int):
        total = sum(int(v) for v in values)
        return MetricResult(
            name=metric_name,
            value=total,
            threshold=first.threshold,
            status=combined_status,
        )
    if isinstance(first.value, str):
        str_values = [str(v) for v in values]
        if _is_p_r_format(first.value):
            agg = _aggregate_p_r(str_values)
        else:
            agg = _aggregate_num_den(str_values)
        thresholds = [m.threshold for m in per_regime]
        return MetricResult(
            name=metric_name,
            value=agg,
            threshold=_aggregate_threshold_str(thresholds),
            status=combined_status,
        )
    # Unknown value shape — preserve first as-is.
    return MetricResult(
        name=metric_name,
        value=first.value,
        threshold=first.threshold,
        status=combined_status,
    )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def build_summary_context(reports: Sequence[RegimeReport]) -> dict[str, Any]:
    """Return the render context for ``summary.md.j2``.

    Shape:

        {
          "regimes": [
            {
              "regime_number": 1,
              "header_dates":  "Mar 11–15, 2024",
              "description":   "earnings-heavy",
              "perf":          {... same as regime.md.j2 perf dict ...},
            },
            ...
          ],
          "aggregated_metrics": [MetricResult, ...]   # in spec §9.5 order
        }

    Parameters
    ----------
    reports : non-empty sequence of :class:`RegimeReport`.

    Raises
    ------
    ValueError
        If *reports* is empty or if two reports disagree on the ordering /
        names of their process_metrics (which would indicate an upstream
        bug — the compute_all_metrics emitter is stable).
    """
    if not reports:
        raise ValueError("build_summary_context: reports must be non-empty")

    # Import locally to avoid eagerly pulling _dates into every aggregator
    # consumer's namespace.
    from firm.eval._dates import format_header_dates

    regimes_ctx: list[dict[str, Any]] = []
    for i, r in enumerate(reports, start=1):
        regimes_ctx.append(
            {
                "regime_number": i,
                "header_dates": format_header_dates(r.start_date, r.end_date),
                "description": _regime_description(r.regime_id),
                "perf": r.perf_metrics,
            }
        )

    # Fixed metric order: the first regime's ordering wins. All regimes are
    # required to emit the same metric names in the same order.
    first_names = [m.name for m in reports[0].process_metrics]
    for r in reports[1:]:
        if [m.name for m in r.process_metrics] != first_names:
            raise ValueError(
                f"regime {r.regime_id!r} emits a different process_metrics "
                f"ordering than the first regime — refusing to aggregate"
            )

    aggregated: list[MetricResult] = []
    for name in first_names:
        per_regime = [
            next(m for m in r.process_metrics if m.name == name)
            for r in reports
        ]
        aggregated.append(_aggregate_one_metric(name, per_regime))

    return {"regimes": regimes_ctx, "aggregated_metrics": aggregated}


# ---------------------------------------------------------------------------
# Helper — regime_id → description. The summary template needs the human
# description ("earnings-heavy") not the machine id ("r1_earnings"). The
# RegimeReport returned by ``run_regime`` doesn't carry the description, so
# we re-derive it from the regime registry. Keeping this in the aggregator
# (rather than mutating RegimeReport) avoids touching T13's schema.
# ---------------------------------------------------------------------------


def _regime_description(regime_id: str) -> str:
    """Look up the human-readable description for a registered regime_id.

    Falls back to *regime_id* itself if the id isn't in the registry — the
    aggregator must never crash because of a name mismatch; the renderer
    will print whatever we hand it.
    """
    try:
        from firm.eval.regimes import get_regime

        return get_regime(regime_id).description
    except KeyError:
        return regime_id


__all__ = ["build_summary_context"]
