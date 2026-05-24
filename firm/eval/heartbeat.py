"""Production heartbeat wiring for the eval harness (Plan 4 §T15).

The eval runner (T13) drives the per-day loop and treats the heartbeat as
an opaque callable. T15 supplies the production heartbeat: build the live
agent graph once, then invoke it with a per-day :class:`ReplayClock` so
the LLM/RAG/router/broker stack writes its decisions + audit-log rows
into the per-regime sqlite DB.

Per-day clock injection
-----------------------
The graph is constructed ONCE on first call and the SAME ``ReplayClock``
instance is threaded into broker / monitor / research / pm / risk / hitl /
execution / reporter at construction time. Each subsequent call mutates
the clock via :meth:`ReplayClock.set` so every downstream component (all
holding the same reference) sees the new day. This avoids per-day graph
rebuild (cheaper) and keeps audit-log ``ts``, ``broker.fill`` rows and
reporter ``date_dir`` correctly bound to each day in the regime window.

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

Misconfig skip gate
-------------------
``_build_llm_stack`` raises :class:`click.ClickException` for missing
``QDRANT_URL`` / missing API key / missing model config. By default these
propagate (loud failure). For dev-mode runs where T16/T17 fixtures aren't
in place yet, set ``FIRM_EVAL_SKIP_MISCONFIG=1`` to additionally treat
``ClickException`` as skippable. The Makefile ``eval`` target sets this
env var so ``make eval`` keeps working pre-T16/T17; the bare ``firm eval``
invocation fails loudly on misconfig.

The heartbeat never mutates process env vars per call. Callers (the CLI
``eval`` subcommand) must set ``FIRM_LLM_MODE`` / ``FIRM_VCR_MODE`` etc.
once at entry time.
"""
from __future__ import annotations

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
    # Wrapped form of Qdrant's 404 — keep co-listed with UnexpectedResponse so
    # missing-collection skips survive the qdrant_store wrapping.
    ("firm.rag.qdrant_store", "MissingCollectionError"),
)

# Misconfig (missing QDRANT_URL / API key / model config) surfaces from
# _build_llm_stack as click.ClickException. By default we let it propagate
# so operators see the loud failure. The dev-mode env var below opts into
# treating it as skippable so ``make eval`` keeps working before T16/T17
# populate cassettes + price parquets.
_MISCONFIG_SKIP_REFS: tuple[tuple[str, str], ...] = (
    ("click.exceptions", "ClickException"),
)
_MISCONFIG_SKIP_ENV = "FIRM_EVAL_SKIP_MISCONFIG"

_cached_skippable: tuple[type[BaseException], ...] | None = None
_cached_skippable_with_misconfig: tuple[type[BaseException], ...] | None = None


def _resolve_refs(
    refs: tuple[tuple[str, str], ...],
) -> tuple[type[BaseException], ...]:
    """Import + return the exception classes named by *refs*.

    Modules that fail to import (e.g. ``qdrant_client`` absent) are
    silently dropped from the tuple — if the dep isn't installed the
    code path that would raise its error can't fire anyway.
    """
    classes: list[type[BaseException]] = []
    for mod_path, cls_name in refs:
        try:
            mod = __import__(mod_path, fromlist=[cls_name])
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            classes.append(cls)
    return tuple(classes)


def _resolve_skippable() -> tuple[type[BaseException], ...]:
    """Resolve the skip allow-list. Caches both env-on / env-off variants."""
    global _cached_skippable, _cached_skippable_with_misconfig
    if os.environ.get(_MISCONFIG_SKIP_ENV) == "1":
        if _cached_skippable_with_misconfig is None:
            _cached_skippable_with_misconfig = (
                _resolve_refs(_SKIPPABLE_REFS)
                + _resolve_refs(_MISCONFIG_SKIP_REFS)
            )
        return _cached_skippable_with_misconfig
    if _cached_skippable is None:
        _cached_skippable = _resolve_refs(_SKIPPABLE_REFS)
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

    Construction is memoised; the SAME ``ReplayClock`` instance is threaded
    into every downstream component (broker / monitor / research / pm /
    risk / hitl / execution / reporter). Each per-day invocation calls
    :meth:`ReplayClock.set` to advance that single instance — every holder
    immediately sees the new day's timestamp. This is materially cheaper
    than per-day graph rebuild while still binding each day's audit-log
    rows, ``broker.fill`` events and reporter ``date_dir`` to the correct
    calendar date.

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
    # ``boot_clock`` holds the SINGLE shared ReplayClock that downstream
    # components close over; each day's _heartbeat call mutates it via
    # .set() so every holder advances in lockstep.
    state: dict[str, Any] = {
        "graph": None,
        "build_failed": False,
        "boot_clock": None,
    }

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

        # The SINGLE shared clock — every downstream component closes over
        # this instance. _heartbeat mutates .set() per day so they all
        # observe the new day's now() without needing to be rebuilt.
        boot_clock = ReplayClock(
            datetime.combine(
                config.start_date, time(0, 0), tzinfo=timezone.utc
            )
        )
        state["boot_clock"] = boot_clock

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
        execution = make_execution(
            db_path=db_path,
            broker=broker,
            clock=boot_clock,
            nonce_secret=secret_bytes,
        )
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

        # Per-day invocation: mutate the shared boot_clock so every
        # downstream component (broker, audit log, reporter, ...) observes
        # the new day. ReplayClock instances are referenced — never copied
        # — by the downstream make_* factories, so a single .set() advances
        # them all at once.
        day_dt = datetime.combine(day, time(0, 0), tzinfo=timezone.utc)
        boot_clock = state["boot_clock"]
        assert boot_clock is not None  # _build_graph_once sets this
        boot_clock.set(day_dt)

        try:
            from firm.obs import agent_span

            try:
                from langchain_core.runnables import RunnableConfig
            except ImportError:
                RunnableConfig = dict  # type: ignore[assignment,misc]

            graph_cfg: RunnableConfig = {
                "configurable": {"thread_id": day_dt.isoformat()}
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


__all__ = ["HeartbeatFn", "make_eval_heartbeat"]
