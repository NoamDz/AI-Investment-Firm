"""Per-regime eval runner (Plan 4 §T13).

Orchestrates one :class:`RegimeConfig` end-to-end:

  1. Wipe + re-init a sqlite DB so the run starts from a clean schema.
  2. Loop one heartbeat callable per calendar day in ``[start, end]``.
  3. Query decisions / audit log / hitl_queue out of the DB the heartbeat
     wrote into.
  4. Reconstruct ``Fill`` records from the ``broker.fill`` audit-log events
     (there is no ``fills`` table — fills live as audit-log details, by
     convention; the heartbeat is responsible for emitting them).
  5. Compute T11 perf metrics + T12 process metrics.
  6. Render ``reports/eval/<start_date>.md`` via the placeholder Jinja
     template at ``firm/reports/templates/regime.md.j2``. T14 will swap that
     template for the full spec §9.7 styled version; this T13 template
     already includes every section header the spec calls out so the report
     never regresses below the section-coverage assertion.

Heartbeat contract
------------------
The ``heartbeat`` callable is what runs the agent graph for a single day.
T15 owns the production wiring; T13 owns the orchestration contract:

* Called once per calendar day in ``[start_date, end_date]`` inclusive.
* MUST write any executed orders as ``audit_log`` rows with
  ``event = 'broker.fill'`` and a JSON detail dict containing
  ``side``, ``ticker``, ``shares``, ``fill_price``, ``commission``.
* MUST write each emitted Decision into the ``decisions`` table.
* The default heartbeat (``heartbeat=None``) raises ``NotImplementedError``;
  tests inject a stub that seeds the DB directly. T15 supplies the live one.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, ConfigDict

from firm.audit.log import AuditLog
from firm.core.clock import ReplayClock
from firm.core.models import Citation, Claim, FailureMode
from firm.db.migrations import init_db
from firm.eval.benchmarks import compute_basket_return, compute_spy_return
from firm.eval.perf_metrics import Fill, compute_perf_metrics
from firm.eval.process_metrics import (
    ClosedTrade,
    HitlPair,
    MetricResult,
    ProcessMetricsInput,
    compute_all_metrics,
)
from firm.eval.regimes import RegimeConfig

HeartbeatFn = Callable[[date, Path], None]


class RegimeReport(BaseModel):
    """Immutable summary of one regime run returned from :func:`run_regime`."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    regime_id: str
    start_date: date
    end_date: date
    num_days: int
    num_decisions: int
    num_fills: int
    perf_metrics: dict[str, float | str]
    process_metrics: list[MetricResult]
    report_path: Path


# ---------------------------------------------------------------------------
# Lightweight "decision-like" shim used for the T12 discipline + citation
# metrics. The real ``firm.core.models.Decision`` enforces a strict
# discriminated-union payload schema; hydrating archived DB rows back through
# that validator would couple the runner to every payload schema change.
# T12's metric functions only read ``rationale``, ``citations``,
# ``falsification_condition`` — so a duck-typed shim is sufficient and keeps
# the orchestration boundary narrow.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _DecisionShim:
    rationale: str
    citations: list[Citation]
    falsification_condition: str


def _default_heartbeat(day: date, db_path: Path) -> None:
    raise NotImplementedError(
        "Production heartbeat wiring is task T15; for T13 pass a heartbeat "
        "callable explicitly or use the test stub."
    )


def _start_of_window_dt(start: date) -> datetime:
    return datetime.combine(start, time(0, 0), tzinfo=timezone.utc)


def _load_template() -> Any:
    templates_dir = Path(__file__).resolve().parents[1] / "reports" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env.get_template("regime.md.j2")


def _parse_date_from_iso(ts: str) -> date:
    # Audit-log ts is a tz-aware ISO 8601 string emitted by AuditLog.append;
    # ``datetime.fromisoformat`` accepts that shape on Python 3.11+.
    return datetime.fromisoformat(ts).date()


def _query_decision_rows(db_path: Path) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, action, payload, rationale, confidence, citations, "
            "falsification, escalation, failure_mode, metadata, nonce, created_at "
            "FROM decisions ORDER BY created_at"
        )
        return [dict(r) for r in cur.fetchall()]


def _query_broker_fills(audit_rows: Sequence[Mapping[str, Any]]) -> list[tuple[date, Fill]]:
    fills: list[tuple[date, Fill]] = []
    for row in audit_rows:
        if row.get("event") != "broker.fill":
            continue
        detail_raw = row.get("detail")
        if not isinstance(detail_raw, str):
            continue
        detail = json.loads(detail_raw)
        side = detail["side"]
        if side not in ("buy", "sell"):
            raise ValueError(f"broker.fill detail has invalid side: {side!r}")
        fill = Fill(
            side=side,
            ticker=str(detail["ticker"]),
            shares=Decimal(str(detail["shares"])),
            fill_price=Decimal(str(detail["fill_price"])),
            commission=Decimal(str(detail.get("commission", "0"))),
        )
        ts = row.get("ts")
        if not isinstance(ts, str):
            raise ValueError("broker.fill row missing ts")
        fills.append((_parse_date_from_iso(ts), fill))
    return fills


def _query_hitl_rows(db_path: Path) -> list[HitlPair]:
    with closing(sqlite3.connect(str(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT decision_id, status, approval_nonce FROM hitl_queue"
        )
        rows = cur.fetchall()
    pairs: list[HitlPair] = []
    for r in rows:
        # TODO(plan4): verify HMAC via firm.hitl.signing.verify_with_rotation;
        # for T13 scope we accept presence-of-approval as the validity proxy.
        approval_valid = r["status"] == "approved"
        pairs.append(
            HitlPair(
                decision_id=str(r["decision_id"]),
                above_threshold=True,
                approval_valid=approval_valid,
            )
        )
    return pairs


def _query_rejection_count(db_path: Path) -> int:
    with closing(sqlite3.connect(str(db_path))) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE failure_mode = ?",
            (FailureMode.SCHEMA_VALIDATION_FAILED.value,),
        ).fetchone()
    return int(row[0])


def _query_cost_ledger_count(db_path: Path) -> int:
    with closing(sqlite3.connect(str(db_path))) as conn:
        row = conn.execute("SELECT COUNT(*) FROM cost_ledger").fetchone()
    return int(row[0])


def _query_triggered_failure_modes(db_path: Path) -> list[FailureMode]:
    with closing(sqlite3.connect(str(db_path))) as conn:
        cur = conn.execute(
            "SELECT DISTINCT failure_mode FROM decisions "
            "WHERE failure_mode IS NOT NULL"
        )
        raw = [r[0] for r in cur.fetchall()]
    modes: list[FailureMode] = []
    for raw_value in raw:
        try:
            modes.append(FailureMode(raw_value))
        except ValueError:
            # Unknown strings are ignored — the coverage metric is robustness
            # shaped, and stale FailureMode values in fixtures shouldn't crash
            # the runner.
            continue
    return modes


def _claims_from_decisions(rows: Sequence[Mapping[str, Any]]) -> list[Claim]:
    """Build the claims sequence consumed by ``compute_groundedness``.

    Each Citation embedded in a decision counts as one grounded Claim (its
    ``source_chunk_id`` is the citation's ``chunk_id``). For any decision
    whose ``failure_mode == 'uncited_claim'`` we additionally emit one
    Claim with no provenance, so the groundedness metric registers the
    failure mode rather than silently passing on a count of zero claims.
    """
    claims: list[Claim] = []
    for row in rows:
        cit_raw = row.get("citations")
        if isinstance(cit_raw, str) and cit_raw:
            try:
                cit_list = json.loads(cit_raw)
            except json.JSONDecodeError:
                cit_list = []
            if isinstance(cit_list, list):
                for c in cit_list:
                    chunk_id = c.get("chunk_id") if isinstance(c, dict) else None
                    claims.append(
                        Claim(
                            text=(c.get("cited_text") or "") if isinstance(c, dict) else "",
                            source_chunk_id=chunk_id,
                        )
                    )
        if row.get("failure_mode") == FailureMode.UNCITED_CLAIM.value:
            claims.append(Claim(text="(uncited)"))
    return claims


def _decision_shims(rows: Sequence[Mapping[str, Any]]) -> list[_DecisionShim]:
    shims: list[_DecisionShim] = []
    for row in rows:
        cit_raw = row.get("citations") or "[]"
        try:
            cit_list = json.loads(cit_raw) if isinstance(cit_raw, str) else []
        except json.JSONDecodeError:
            cit_list = []
        citations: list[Citation] = []
        if isinstance(cit_list, list):
            for c in cit_list:
                if not isinstance(c, dict):
                    continue
                span = c.get("span", (0, 1))
                if isinstance(span, list):
                    span_tuple = (int(span[0]), int(span[1])) if len(span) == 2 else (0, 1)
                else:
                    span_tuple = tuple(span) if isinstance(span, tuple) else (0, 1)
                citations.append(
                    Citation(
                        source_id=str(c.get("source_id", "")),
                        chunk_id=str(c.get("chunk_id", "")),
                        span=span_tuple,
                    )
                )
        shims.append(
            _DecisionShim(
                rationale=str(row.get("rationale") or ""),
                citations=citations,
                falsification_condition=str(row.get("falsification") or ""),
            )
        )
    return shims


def _fifo_closed_trades(dated_fills: Sequence[tuple[date, Fill]]) -> list[ClosedTrade]:
    """FIFO-match buys against sells per ticker, return closed round-trips.

    Mirrors the per-trade matcher in ``perf_metrics`` but additionally
    threads through ``(entry_date, exit_date)`` for ``reversal_rate``. Open
    lots at end-of-window are excluded — T12's reversal_rate only sees
    closed trades.
    """
    @dataclass
    class _Lot:
        shares: Decimal
        buy_price: Decimal
        buy_comm_per_share: Decimal
        entry_date: date

    lots: dict[str, list[_Lot]] = {}
    closed: list[ClosedTrade] = []
    for d, f in dated_fills:
        if f.side == "buy":
            buy_comm_per_share = (
                f.commission / f.shares if f.shares > 0 else Decimal("0")
            )
            lots.setdefault(f.ticker, []).append(
                _Lot(
                    shares=f.shares,
                    buy_price=f.fill_price,
                    buy_comm_per_share=buy_comm_per_share,
                    entry_date=d,
                )
            )
            continue
        # sell
        queue = lots.get(f.ticker, [])
        remaining = f.shares
        sell_comm_per_share = (
            f.commission / f.shares if f.shares > 0 else Decimal("0")
        )
        while remaining > 0 and queue:
            lot = queue[0]
            matched = min(lot.shares, remaining)
            pnl = (
                (f.fill_price - sell_comm_per_share)
                - (lot.buy_price + lot.buy_comm_per_share)
            ) * matched
            closed.append(
                ClosedTrade(
                    ticker=f.ticker,
                    entry_date=lot.entry_date,
                    exit_date=d,
                    pnl=pnl,
                )
            )
            lot.shares -= matched
            remaining -= matched
            if lot.shares == 0:
                queue.pop(0)
    return closed


def _final_cash_and_positions(
    initial_cash: Decimal, fills: Sequence[Fill]
) -> tuple[Decimal, dict[str, Decimal]]:
    cash = initial_cash
    positions: dict[str, Decimal] = {}
    for f in fills:
        gross = f.shares * f.fill_price
        if f.side == "buy":
            cash -= gross + f.commission
            positions[f.ticker] = positions.get(f.ticker, Decimal("0")) + f.shares
        else:
            cash += gross - f.commission
            positions[f.ticker] = positions.get(f.ticker, Decimal("0")) - f.shares
    return cash, positions


def run_regime(
    config: RegimeConfig,
    *,
    output_dir: Path,
    db_path: Path | None = None,
    heartbeat: HeartbeatFn | None = None,
    spy_return: float | None = None,
    basket_return: float | None = None,
    prices_dir: Path | None = None,
    initial_cash: Decimal = Decimal("100000"),
    final_marks: dict[str, Decimal] | None = None,
    red_team_passed: int = 0,
    red_team_total: int = 50,
    sufficiency_precision: float = 1.0,
    sufficiency_recall: float = 1.0,
) -> RegimeReport:
    """Run one regime end-to-end and write its Markdown report.

    Algorithm:

    1. Resolve + wipe ``db_path``; ``init_db`` installs the fresh schema.
    2. Call ``heartbeat(day, db_path)`` for each calendar day in
       ``[config.start_date, config.end_date]`` inclusive.
    3. Query decisions / audit_log / hitl_queue / cost_ledger from the DB.
    4. Reconstruct ``Fill`` records from ``broker.fill`` audit events.
    5. Compute T11 perf + T12 process metrics.
    6. Render ``output_dir / f"{start_date}.md"`` via
       ``firm/reports/templates/regime.md.j2``.

    Parameters
    ----------
    config            : the regime to run.
    output_dir        : directory to write the ``.md`` into (created if missing).
    db_path           : sqlite path; if ``None`` derived from output_dir +
                        regime_id. Deleted before init so multiple calls to
                        the same path do NOT leak state.
    heartbeat         : per-day callable; if ``None`` raises (production
                        wiring is T15).
    spy_return,
    basket_return     : pre-computed benchmark returns; if either is ``None``
                        the runner calls T10. ``PriceCassetteMissError``
                        propagates rather than being swallowed.
    prices_dir        : forwarded to T10 calls when benchmarks aren't pre-
                        computed.
    initial_cash      : starting balance for cash-walk reconstruction.
    final_marks       : ticker→mark for open positions at end-of-window. If
                        ``None`` AND open positions exist, ``compute_perf_metrics``
                        will raise — the caller must pass marks for any open
                        positions to get a clean report.
    red_team_passed,
    red_team_total    : red-team rollup. The suite runs in CI, not here, so
                        defaults to ``0 / 50``; callers that already ran the
                        suite can pipe values in.
    sufficiency_*     : sufficiency-gate precision/recall, not measured by
                        T13 scope; defaults to ``1.0, 1.0``.

    Returns
    -------
    RegimeReport      : frozen pydantic model with metric values + the .md
                        path that was written.
    """
    if heartbeat is None:
        heartbeat = _default_heartbeat

    # ------ Resolve + wipe the DB so each run starts from a fresh schema.
    output_dir.mkdir(parents=True, exist_ok=True)
    if db_path is None:
        db_path = output_dir / f"{config.regime_id}.db"
    if db_path.exists():
        db_path.unlink()
    # SQLite WAL leaves -wal / -shm sidecars; clear them too so a stale
    # WAL from a prior run can't shadow the new schema.
    for sidecar in (
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    ):
        if sidecar.exists():
            sidecar.unlink()
    init_db(db_path)

    # ------ Per-calendar-day loop (inclusive on both ends).
    num_days = (config.end_date - config.start_date).days + 1
    for i in range(num_days):
        heartbeat(config.start_date + timedelta(days=i), db_path)

    # ------ Pull what every downstream metric needs out of the DB.
    decision_rows = _query_decision_rows(db_path)
    audit_rows = AuditLog(db_path, ReplayClock(_start_of_window_dt(config.start_date))).read_all()
    dated_fills = _query_broker_fills(audit_rows)
    fills_only: list[Fill] = [f for _, f in dated_fills]
    hitl_required = _query_hitl_rows(db_path)
    rejection_count = _query_rejection_count(db_path)
    _ = _query_cost_ledger_count(db_path)  # T13 surfaces count only via DB; not in report
    triggered_failure_modes = _query_triggered_failure_modes(db_path)

    claims = _claims_from_decisions(decision_rows)
    decision_shims = _decision_shims(decision_rows)
    closed_trades = _fifo_closed_trades(dated_fills)

    # ------ Resolve benchmark returns (kwargs win; otherwise T10).
    if spy_return is None:
        spy_return = compute_spy_return(
            config.start_date, config.end_date, prices_dir=prices_dir
        )
    if basket_return is None:
        basket_return = compute_basket_return(
            list(config.universe),
            config.start_date,
            config.end_date,
            prices_dir=prices_dir,
        )

    # ------ Cash + positions walk → perf metrics.
    final_cash, final_positions = _final_cash_and_positions(initial_cash, fills_only)
    marks = dict(final_marks) if final_marks is not None else {}
    perf = compute_perf_metrics(
        initial_cash=initial_cash,
        fills=fills_only,
        final_cash=final_cash,
        final_positions=final_positions,
        final_marks=marks,
        spy_return=spy_return,
        basket_return=basket_return,
    )

    # ------ Process metrics.
    process_metrics = compute_all_metrics(
        ProcessMetricsInput(
            claims=claims,
            decisions=decision_shims,  # type: ignore[arg-type]  # duck-typed shim, see _DecisionShim
            closed_trades=closed_trades,
            audit_log=audit_rows,
            hitl_required=hitl_required,
            rejection_count=rejection_count,
            red_team_passed=red_team_passed,
            red_team_total=red_team_total,
            sufficiency_precision=sufficiency_precision,
            sufficiency_recall=sufficiency_recall,
            triggered_failure_modes=triggered_failure_modes,
        )
    )

    # ------ Render report.
    template = _load_template()
    rendered = template.render(
        regime_id=config.regime_id,
        description=config.description,
        start_date=config.start_date.isoformat(),
        end_date=config.end_date.isoformat(),
        perf=perf,
        process_metrics=process_metrics,
        num_days=num_days,
        num_decisions=len(decision_rows),
        num_fills=len(fills_only),
    )
    report_path = output_dir / f"{config.start_date.isoformat()}.md"
    report_path.write_text(rendered, encoding="utf-8", newline="\n")

    return RegimeReport(
        regime_id=config.regime_id,
        start_date=config.start_date,
        end_date=config.end_date,
        num_days=num_days,
        num_decisions=len(decision_rows),
        num_fills=len(fills_only),
        perf_metrics=perf,
        process_metrics=process_metrics,
        report_path=report_path.resolve(),
    )


__all__ = ["HeartbeatFn", "RegimeReport", "run_regime"]
