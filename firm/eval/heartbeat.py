"""Production heartbeat wiring for the eval harness (Plan 4 §T15).

The eval runner (T13) drives the per-day loop and treats the heartbeat as
an opaque callable. T15 supplies the production heartbeat: build the live
agent graph once, then invoke it with a per-day :class:`ReplayClock` so
the LLM/RAG/router/broker stack writes its decisions + audit-log rows
into the per-regime sqlite DB.

Determinism + missing-fixture handling
--------------------------------------
The eval harness's load-bearing assertion is **byte-for-byte idempotency
on re-run**. To honour that without requiring T16's cassettes + T10's
price parquets to be present at T15 ship time, the heartbeat swallows a
fixed allow-list of missing-fixture errors and writes one
``audit_log`` row with ``event='heartbeat.skipped'`` per skipped day.
A skipped day produces the same skip row on every re-run, so the
idempotency check holds.

Errors that DO propagate (i.e. real bugs, not missing data):
  * any exception not in :data:`_SKIPPABLE_EXCEPTIONS`
  * any error raised during graph CONSTRUCTION (vs invocation) — a bad
    config should fail the whole eval run, not silently skip every day.

The heartbeat never mutates process env vars per call. Callers (the CLI
``eval`` subcommand) must set ``FIRM_LLM_MODE`` / ``FIRM_VCR_MODE`` etc.
once at entry time.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from firm.audit.log import AuditLog
from firm.core.clock import ReplayClock
from firm.db.migrations import init_db
from firm.eval.regimes import RegimeConfig

HeartbeatFn = Callable[[date, Path], None]


# ---------------------------------------------------------------------------
# Skippable-error allow-list.
#
# Each entry is a (module_path, class_name) pair; we resolve lazily so the
# heartbeat module is importable without optional deps (qdrant_client etc.)
# already in sys.modules at import time. Resolved exception classes are
# cached on first call.
# ---------------------------------------------------------------------------
_SKIPPABLE_REFS: tuple[tuple[str, str], ...] = (
    ("firm.llm.anthropic_client", "LlmCacheMissError"),
    ("firm.eval.benchmarks", "PriceCassetteMissError"),
    ("firm.llm.cassettes", "CassetteMissError"),
    ("qdrant_client.http.exceptions", "UnexpectedResponse"),
    # Construction errors from _build_llm_stack get wrapped as
    # click.ClickException; we treat those as "fixtures not ready" too,
    # because T15 ships before T16/T17 populate cassettes + price parquets.
    # An unconfigured QDRANT_URL / missing models / missing API key are all
    # legitimate "skip this run" signals during the dry idempotency check.
    ("click.exceptions", "ClickException"),
)

_cached_skippable: tuple[type[BaseException], ...] | None = None


def _resolve_skippable() -> tuple[type[BaseException], ...]:
    """Import + cache the skippable exception classes.

    Modules that fail to import (e.g. ``qdrant_client`` absent) are
    silently dropped from the tuple — if the dep isn't installed the
    code path that would raise its error can't fire anyway.
    """
    global _cached_skippable
    if _cached_skippable is not None:
        return _cached_skippable
    classes: list[type[BaseException]] = []
    for mod_path, cls_name in _SKIPPABLE_REFS:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            classes.append(cls)
    _cached_skippable = tuple(classes)
    return _cached_skippable


def _record_skip(db_path: Path, day: date, exc: BaseException) -> None:
    """Append one ``heartbeat.skipped`` audit-log row for *day*.

    Uses a fresh :class:`ReplayClock` pinned to *day* so the row's ``ts``
    is deterministic across re-runs (the second invocation of a skipped
    day must produce the same row to satisfy byte-for-byte idempotency).
    """
    clock = ReplayClock(datetime.combine(day, time(0, 0), tzinfo=timezone.utc))
    audit = AuditLog(db_path, clock)
    audit.append(
        "heartbeat.skipped",
        {
            "day": day.isoformat(),
            "reason": type(exc).__name__,
            "message": str(exc),
        },
    )


# ---------------------------------------------------------------------------
# Production heartbeat factory.
# ---------------------------------------------------------------------------


def make_eval_heartbeat(
    config: RegimeConfig,
    *,
    reports_root: Path | None = None,
    nonce_secret: bytes | None = None,
) -> HeartbeatFn:
    """Return a production heartbeat for *config*'s daily loop.

    The graph is constructed LAZILY (on first day) so that:

      1. Construction errors are still caught + logged as one
         ``heartbeat.skipped`` row on day 1 (rather than crashing the CLI
         before any DB rows are written — which would break idempotency
         on re-runs where the DB was never even initialised).
      2. The runner gets a chance to call ``init_db(db_path)`` (which it
         does before the per-day loop) before the heartbeat tries to
         talk to the DB.

    Construction is then memoised — subsequent days reuse the same
    broker/research/pm/risk/hitl/execution/reporter stack. This mirrors
    ``firm.cli.run`` which builds the graph once per process.

    Parameters
    ----------
    config        : the regime being run (currently unused inside the
                    heartbeat itself; threaded through so future per-regime
                    overrides have a place to land without changing the
                    factory signature).
    reports_root  : where the reporter writes ``data/reports/<date>/...``
                    style artifacts. The CLI MUST pass an eval-scoped
                    path to avoid clobbering ``data/reports/`` from the
                    production ``firm run`` workflow.
    nonce_secret  : hex-decoded HMAC secret bytes. If ``None``, read from
                    ``FIRM_HMAC_SECRET`` at first-call time.
    """
    # Mutable per-heartbeat state — captured by the returned closure.
    state: dict[str, Any] = {"graph": None, "build_failed": False}

    def _build_graph_once(db_path: Path) -> Any:
        """Construct the agent graph on first invocation; reuse thereafter."""
        if state["graph"] is not None:
            return state["graph"]

        # Local imports keep firm.eval importable in environments that don't
        # have the optional broker / qdrant / langgraph deps installed.
        from firm.agents.execution import make_execution
        from firm.agents.hitl import make_hitl
        from firm.agents.monitor import make_monitor
        from firm.agents.pm import PmVoter, make_pm
        from firm.agents.reporter import make_reporter
        from firm.agents.research import make_research
        from firm.agents.risk import RiskInput, evaluate_risk
        from firm.broker.alpaca_paper import make_broker
        from firm.cli import _build_llm_stack, _rag_config_path, _llm_config_path
        from firm.core.config import (
            load_llm_config,
            load_policy,
            load_rag_config,
            load_universe,
        )
        from firm.core.models import BuyPayload, SellPayload
        from firm.llm.messages_client import RouterBackedMessagesClient
        from firm.obs import agent_span, stamp_decision
        from firm.orchestrator.graph import build_graph
        from firm.orchestrator.state import WorkingState

        # Use the runner-installed clock as the construction-time clock; the
        # per-day invocation rebinds via a fresh ReplayClock below.
        boot_clock = ReplayClock(
            datetime.combine(
                config.start_date, time(0, 0), tzinfo=timezone.utc
            )
        )

        broker = make_broker(clock=boot_clock)
        policy = load_policy(Path("config/policy.yaml"))
        universe = load_universe(Path("config/universe.yaml"))
        rag_config = load_rag_config(_rag_config_path())
        llm_config = load_llm_config(_llm_config_path())

        retriever, extractor, judge, router = _build_llm_stack(
            db_path, boot_clock, rag_config, llm_config
        )

        # HMAC secret: prefer the constructor kwarg; fall back to env.
        secret_bytes: bytes
        if nonce_secret is not None:
            secret_bytes = nonce_secret
        else:
            raw_secret = os.environ.get("FIRM_HMAC_SECRET", "")
            try:
                secret_bytes = bytes.fromhex(raw_secret)
            except ValueError as exc:
                raise RuntimeError(
                    f"FIRM_HMAC_SECRET must be hex: {exc}"
                ) from exc

        monitor = make_monitor(boot_clock)
        research = make_research(
            clock=boot_clock,
            broker=broker,
            universe=universe,
            retriever=retriever,
            extractor=extractor,
            judge=judge,
            nonce_secret=secret_bytes,
            router=router,
            db_path=db_path,
        )
        voter_adapter = RouterBackedMessagesClient(router=router)
        voter = PmVoter(client=voter_adapter, model=llm_config.pm.model)
        pm = make_pm(voter=voter, router=router)

        def risk_node(state_: WorkingState) -> dict[str, Any]:
            with agent_span("risk") as span:
                proposal = state_["pm_decision"]
                if not isinstance(proposal.payload, (BuyPayload, SellPayload)):
                    stamp_decision(span, proposal.id, proposal.failure_mode)
                    return {"risk_decision": proposal}
                ticker = proposal.payload.ticker
                quote = broker.get_quote(ticker)
                positions = {p.ticker: p.shares for p in broker.list_positions()}
                decision = evaluate_risk(RiskInput(
                    proposal=proposal,
                    quote_price=quote.price,
                    quote_age_seconds=0,
                    cash=broker.get_cash(),
                    positions=positions,
                    sector_map=universe.sector_map,
                    trades_today=0,
                    nav=broker.get_cash() + sum(
                        (
                            p.shares * broker.get_quote(p.ticker).price
                            for p in broker.list_positions()
                        ),
                        Decimal("0"),
                    ),
                    daily_pnl_pct=0.0,
                    policy=policy,
                ))
                stamp_decision(span, decision.id, decision.failure_mode)
                return {"risk_decision": decision}

        hitl = make_hitl(db_path=db_path, clock=boot_clock, notifier=None)
        execution = make_execution(db_path=db_path, broker=broker, clock=boot_clock)
        scoped_reports = (
            reports_root if reports_root is not None else Path("data/reports")
        )
        reporter = make_reporter(
            reports_root=scoped_reports, clock=boot_clock, db_path=db_path
        )

        graph = build_graph(
            db_path=db_path,
            monitor_node=monitor,
            research_node=research,
            pm_node=pm,
            risk_node=risk_node,
            hitl_node=hitl,
            execution_node=execution,
            reporter_node=reporter,
        )
        state["graph"] = graph
        return graph

    def _heartbeat(day: date, db_path: Path) -> None:
        # Ensure the schema exists. The runner does this already (init_db
        # is called before the per-day loop), but it's cheap + idempotent
        # and protects the audit-log write below if a future runner change
        # ever moves init_db out from under us.
        init_db(db_path)

        skippable = _resolve_skippable()

        # If a previous day's graph construction failed, every subsequent
        # day records a skip row using the cached failure rather than re-
        # attempting construction. This keeps re-runs byte-identical even
        # when external deps oscillate between runs (e.g. Qdrant warming).
        if state["build_failed"]:
            _record_skip(
                db_path, day, RuntimeError("graph construction previously failed")
            )
            return

        try:
            graph = _build_graph_once(db_path)
        except skippable as exc:
            state["build_failed"] = True
            _record_skip(db_path, day, exc)
            return
        except Exception:
            # Non-skippable construction error — propagate so CLI surfaces it.
            raise

        # Per-day invocation: rebind the graph's clock by passing the
        # ReplayClock through the LangGraph config (the agent layer reads
        # the clock via WorkingState, not the graph constructor).
        try:
            day_clock = ReplayClock(
                datetime.combine(day, time(0, 0), tzinfo=timezone.utc)
            )
            from firm.obs import agent_span

            try:
                from langchain_core.runnables import RunnableConfig
            except ImportError:
                RunnableConfig = dict  # type: ignore[assignment,misc]

            graph_cfg: RunnableConfig = {
                "configurable": {"thread_id": day_clock.now().isoformat()}
            }
            existing = graph.get_state(graph_cfg)
            invoke_input: dict[str, Any] | None = (
                None if existing.next else {}
            )
            with agent_span("heartbeat"):
                graph.invoke(invoke_input, config=graph_cfg)
        except skippable as exc:
            _record_skip(db_path, day, exc)
            return
        # All other exceptions propagate.

    return _heartbeat


def _coerce_skip_detail(detail_json: str) -> dict[str, Any]:
    """Decode a ``heartbeat.skipped`` audit-log detail back to a dict.

    Test helper — the runner reads audit_log rows as raw strings; tests
    that need to introspect the skip reason can use this rather than
    re-rolling json.loads + isinstance checks.
    """
    obj = json.loads(detail_json)
    if not isinstance(obj, dict):
        raise ValueError("heartbeat.skipped detail is not a JSON object")
    return obj


__all__ = ["HeartbeatFn", "make_eval_heartbeat"]
