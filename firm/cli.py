"""CLI entry points. See spec §3.1, §3.8."""
from __future__ import annotations

import functools
import itertools
import json
import os
import shutil
import sys
import time as _time_module
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from datetime import datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import click

from jinja2 import Environment, FileSystemLoader

from firm.agents.execution import make_execution
from firm.agents.hitl import (
    make_hitl,
    mark_approved,
    mark_rejected,
    reap_expired_hitl_entries,
)
from firm.agents.monitor import make_monitor
from firm.agents.pm import PmVoter, make_pm
from firm.agents.reporter import make_reporter
from firm.agents.research import make_research
from firm.agents.risk import RiskInput, evaluate_risk
from firm.broker.alpaca_paper import make_broker
from firm.broker.protocol import Broker
from firm.core.clock import Clock, ReplayClock, WallClock
from firm.core.config import (
    load_llm_config,
    load_policy,
    load_rag_config,
    load_router_config,
    load_universe,
)
from firm.core.models import BuyPayload, SellPayload
from firm.db.connection import get_conn
from firm.db.cost_ledger import write_cost_ledger_row
from firm.db.migrations import init_db
from firm.llm.messages_client import RouterBackedMessagesClient
from firm.llm.router import CostRouter
from firm.obs import agent_span, stamp_decision
from firm.orchestrator.graph import build_graph
from firm.orchestrator.state import WorkingState
from firm.reconcile.boot import reconcile_on_boot, resolve_from_broker
from firm.reports.daily import render_daily_report
from firm.reports.html import render_daily_html
from firm.reports.reconcile_block import render_reconcile_block
from firm.reports.xlsx import write_positions_xlsx

# Consecutive-identical-failure circuit breaker for ``firm run --loop``: a
# heartbeat that fails the same way this many times in a row is almost
# certainly a config/infra problem (qdrant DNS, missing API key, schema drift)
# rather than a transient hiccup, so re-raise instead of swallowing forever.
_LOOP_FAILURE_THRESHOLD = 3

try:
    from langchain_core.runnables import RunnableConfig
except ImportError:
    RunnableConfig = dict  # type: ignore[assignment,misc]


def _seed_db_from_broker(db_path: Path, broker: Broker, clock: Clock) -> None:
    """On first boot (empty cash table), sync local DB state from broker."""
    with closing(get_conn(db_path)) as conn:
        row = conn.execute("SELECT amount FROM cash WHERE id=1").fetchone()
        if row is not None:
            return  # already seeded
        # First run: initialise local state from broker (broker is source of truth)
        now = clock.now().isoformat()
        conn.execute(
            "INSERT INTO cash (id, amount, updated_at) VALUES (1, ?, ?)",
            (str(broker.get_cash()), now),
        )
        for pos in broker.list_positions():
            conn.execute(
                "INSERT INTO positions (ticker, shares, avg_cost, updated_at) VALUES (?, ?, ?, ?)",
                (pos.ticker, str(pos.shares), str(pos.avg_cost), now),
            )


def _resolve_clock() -> Clock:
    replay = os.environ.get("FIRM_REPLAY_AT")
    if replay:
        return ReplayClock(datetime.fromisoformat(replay))
    return WallClock()


def _compute_quote_age_seconds(quote: Any, clock: Clock) -> int:
    """Compute age of a broker Quote in whole seconds against ``clock.now()``.

    Returns ``max(0, int((now - ts).total_seconds()))``.  Falls back to 0
    on malformed timestamps — STALE_DATA is risk-side disposition we
    surface deterministically when the parse succeeds, and we deliberately
    do NOT crash the heartbeat on a single bad timestamp (the risk
    evaluator's deterministic kernel still gets exercised).
    """
    try:
        ts = datetime.fromisoformat(quote.timestamp)
    except (ValueError, TypeError):
        return 0
    delta = clock.now() - ts
    return max(0, int(delta.total_seconds()))


def _count_trades_today(db_path: Path, clock: Clock) -> int:
    """Count BUY/SELL decisions persisted today (UTC) for the trades_today gate.

    "Today" = today's UTC midnight onwards.  The risk evaluator uses this
    count against ``policy.max_trades_per_day``; over-counting (e.g., counting
    REFUSE/HOLD rows) would falsely trip the limit.  Inline SQL — there is
    no existing helper for this read pattern.
    """
    now_utc = clock.now().astimezone(timezone.utc)
    sod = datetime.combine(now_utc.date(), time.min, tzinfo=timezone.utc)
    with closing(get_conn(db_path)) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM decisions "
            "WHERE action IN ('BUY','SELL') AND created_at >= ?",
            (sod.isoformat(),),
        ).fetchone()
    return int(row["n"]) if row is not None else 0


def _compute_daily_pnl_pct(
    db_path: Path, clock: Clock, current_nav: Decimal
) -> float:
    """Compute daily P&L percentage vs. start-of-day NAV.

    No SOD-NAV snapshotter exists in the codebase today (positions/cash
    tables are mutable with no time-series).  Return 0.0 to keep the
    daily_pnl_pct gate inert until the snapshotter ships, rather than
    shipping a half-working measurement that could trip the limit on
    spurious values.
    """
    # TODO(plan5): wire SOD nav snapshotter (time-series of EOD/SOD NAV) and
    # compute (current_nav - sod_nav) / sod_nav here.  Until then the gate
    # is intentionally inert (0.0) — production policy.max_daily_loss_pct
    # is still wired into the deterministic kernel; only the input is stubbed.
    _ = db_path, clock, current_nav  # parameters reserved for the wired path
    return 0.0


def _db_path() -> Path:
    return Path(os.environ.get("FIRM_DB_PATH", "data/firm.db"))


def _reports_root() -> Path:
    return Path(os.environ.get("FIRM_REPORTS_ROOT", "data/reports"))


def _llm_config_path() -> Path:
    return Path(os.environ.get("FIRM_LLM_CONFIG", "config/llm.yaml"))


def _rag_config_path() -> Path:
    return Path(os.environ.get("FIRM_RAG_CONFIG", "config/rag.yaml"))


def _fit_bm25_sparse(rag_config: Any, sparse: Any) -> None:
    """Pre-pass: fit BM25 vocabulary from the FinanceBench corpus.

    Mirrors the pre-pass in the ``ingest`` command. Both commands share the
    same corpus source; extracting a shared helper is a Plan 3 follow-up.
    The BM25 fit is CPU-only and model-free — the only I/O is reading the
    source documents.
    """
    from firm.rag.chunk import chunk_document
    from firm.rag.financebench import FinanceBenchSource
    from firm.rag.preprocess import tables_to_prose

    fixture_env = os.environ.get("FIRM_FINANCEBENCH_FIXTURE")
    if fixture_env:
        source: Any = FinanceBenchSource(
            dataset_loader=_make_fixture_loader(fixture_env)
        )
    else:
        source = FinanceBenchSource()

    effective_max_docs: int | None = rag_config.corpus.financebench.max_docs

    import itertools as _itertools

    def _iter_source() -> Any:
        if effective_max_docs is not None:
            return _itertools.islice(source.iter_docs(), effective_max_docs)
        return source.iter_docs()

    all_texts: list[str] = []
    for pre_doc in _iter_source():
        processed_html = tables_to_prose(pre_doc.html)
        pre_doc_processed = pre_doc.model_copy(update={"html": processed_html})
        chunks = chunk_document(
            pre_doc_processed,
            target_tokens=rag_config.chunk.target_tokens,
            overlap_tokens=rag_config.chunk.overlap_tokens,
            source="financebench",
        )
        all_texts.extend(c.text for c in chunks)

    if all_texts:
        sparse.fit(all_texts)
    else:
        sparse.fit(["placeholder"])


def _build_llm_stack(
    db: Path, clock: Clock, rag_config: Any, llm_config: Any
) -> tuple[Any, Any, Any, CostRouter]:
    """Construct RAG + LLM components for the grounded research path.

    Returns ``(retriever, extractor, judge, router)`` on success.  Any
    construction failure (Qdrant unreachable, sentence-transformers model
    missing, ``ANTHROPIC_API_KEY`` unset in LIVE/RECORD mode, etc.) is
    converted to a :class:`click.ClickException` pointing the operator to
    ``make ingest`` and ``make record``.  Silent fallback to the Plan 1
    stub is intentionally disallowed — a misconfigured production
    deployment must fail loudly.

    T08: extractor + judge are constructed against router-backed adapters
    (one each, never shared — see T08 spec) so the agent layer can rebind
    them per heartbeat.  The router itself is wired with a partial of
    :func:`write_cost_ledger_row` so each successful underlying call
    appends a row to ``cost_ledger`` (T09).

    The PM voter is constructed unconditionally by the caller (it is cheap and
    has no external model dependencies at construction time).
    """
    try:
        from firm.llm.anthropic_client import CachedAnthropicClient
        from firm.llm.cache import LlmCache
        from firm.rag.embed import BM25Sparse, NomicEmbedder
        from firm.rag.qdrant_store import VectorStore
        from firm.rag.retrieve import GroundedRetriever, HybridRetriever
        from firm.rag.rerank import BgeReranker
        from firm.tools.fundamentals import FundamentalsTool
        from firm.tools.risk_metrics import RiskMetricsTool

        llm_cache = LlmCache(db, clock)
        client = CachedAnthropicClient.from_env(cache=llm_cache, clock=clock)

        # T08: build the cost router with a ledger writer bound to (db, clock).
        # functools.partial keeps the LedgerWriterFn Protocol intact — the
        # router invokes the partial with the per-call kwargs (decision_id,
        # agent, model, tokens, cost) and the partial supplies the rest.
        router_cfg = load_router_config(Path("config/router.yaml"))
        ledger_writer = functools.partial(
            write_cost_ledger_row, db_path=db, clock=clock
        )
        router = CostRouter(
            router_cfg=router_cfg,
            anthropic_client=client,
            ledger_writer=ledger_writer,
        )

        embedder = NomicEmbedder()
        sparse = BM25Sparse()

        click.echo("[firm run] Fitting BM25 vocabulary (pre-pass)...", err=True)
        _fit_bm25_sparse(rag_config, sparse)

        qdrant_client = _make_qdrant_client()
        store = VectorStore(qdrant_client)

        hybrid = HybridRetriever(
            store=store,
            embedder=embedder,
            sparse=sparse,
            collection=rag_config.qdrant.collection,
        )
        reranker = BgeReranker(
            model_id=rag_config.rerank.model,
            score_floor=rag_config.rerank.score_floor,
        )
        retriever = GroundedRetriever(
            hybrid=hybrid,
            reranker=reranker,
            k_final=rag_config.retrieval.top_k_rerank,
        )

        from firm.tools import Tool

        tools: list[Tool] = [
            FundamentalsTool(Path("data/precomputed/fundamentals.parquet")),
            RiskMetricsTool(Path("data/precomputed/risk_metrics.parquet")),
        ]

        from firm.llm.citations import AnthropicCitationsExtractor
        from firm.grounding.judge import SufficiencyJudge

        # T08: each collaborator gets its OWN RouterBackedMessagesClient so a
        # bind() on the extractor's adapter cannot accidentally affect the
        # judge (or vice-versa) within a single heartbeat. Cheap to allocate.
        extractor_adapter = RouterBackedMessagesClient(router=router)
        judge_adapter = RouterBackedMessagesClient(router=router)

        extractor = AnthropicCitationsExtractor(
            client=extractor_adapter,
            model=llm_config.research.model,
            max_tokens=llm_config.research.max_tokens,
            tools=tools,
        )
        judge = SufficiencyJudge(
            client=judge_adapter,
            model=llm_config.judge.model,
        )

        return retriever, extractor, judge, router

    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(
            f"LLM/RAG stack unavailable ({type(exc).__name__}: {exc}).\n"
            "Run 'make ingest' to populate the corpus and 'make record' to "
            "warm the LLM cache before invoking 'firm run'."
        ) from exc


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--once/--loop",
    default=True,
    help="Single heartbeat (default) or loop until SIGINT/SIGTERM.",
)
@click.option(
    "--interval-seconds",
    default=60,
    type=int,
    show_default=True,
    help="Seconds between heartbeats in --loop mode.",
)
def run(once: bool, interval_seconds: int) -> None:
    """Run heartbeats of the firm end-to-end.

    Default is a single heartbeat. ``--loop`` runs continuously, sleeping
    ``--interval-seconds`` between heartbeats; one bad heartbeat is logged
    and the loop continues. SIGINT/SIGTERM stops cleanly after the current
    heartbeat (a second signal exits immediately).
    """
    db = _db_path()
    init_db(db)
    clock = _resolve_clock()
    broker = make_broker(clock=clock)
    policy = load_policy(Path("config/policy.yaml"))
    universe = load_universe(Path("config/universe.yaml"))
    rag_config = load_rag_config(_rag_config_path())
    llm_config = load_llm_config(_llm_config_path())

    _seed_db_from_broker(db, broker, clock)
    recon = reconcile_on_boot(db, broker, clock)
    if recon.status == "mismatch":
        # Boot reconcile = implicit ack (spec §5.7): broker is source of truth,
        # local DB is rewritten. The diff is already audit-logged by
        # reconcile_on_boot() above, and resolve_from_broker() logs the rewrite.
        click.echo(
            f"BOOT RECONCILED: broker is source of truth; local DB synced. "
            f"diff={recon.diff}"
        )
        resolve_from_broker(db, broker, clock, recon.diff)

    monitor = make_monitor(clock)

    # Build the grounded LLM stack — hard-fails if any external dep is missing
    # (Qdrant unreachable, models absent, etc.) so misconfigured deployments
    # surface clearly instead of silently dropping to the Plan 1 stub.
    retriever, extractor, judge, router = _build_llm_stack(
        db, clock, rag_config, llm_config
    )

    # nonce_secret is required for the grounded path; sourced from env.
    raw_secret = os.environ.get("FIRM_HMAC_SECRET")
    if not raw_secret:
        raise click.ClickException(
            "FIRM_HMAC_SECRET is required for the grounded research path."
        )
    try:
        nonce_secret = bytes.fromhex(raw_secret)
    except ValueError as e:
        raise click.ClickException(
            f"FIRM_HMAC_SECRET must be a hex string: {e}"
        ) from e

    research = make_research(
        clock=clock,
        broker=broker,
        universe=universe,
        retriever=retriever,
        extractor=extractor,
        judge=judge,
        nonce_secret=nonce_secret,
        router=router,
        db_path=db,
    )

    # PM voter: cheap to construct (no external deps), and shares the
    # T08 cost router with research so the cost-ledger writer is wired
    # uniformly across both agents.
    voter_adapter = RouterBackedMessagesClient(router=router)
    voter = PmVoter(client=voter_adapter, model=llm_config.pm.model)
    pm = make_pm(voter=voter, router=router)

    def risk_node(state: WorkingState) -> dict[str, Any]:
        # Wrap at the LangGraph-node layer (not inside ``evaluate_risk``, which
        # is the pure-function deterministic kernel) so each heartbeat emits
        # exactly one ``agent.risk`` span carrying ``failure_mode`` propagated
        # from the produced Decision.
        with agent_span("risk") as span:
            proposal = state["pm_decision"]
            if not isinstance(proposal.payload, (BuyPayload, SellPayload)):
                # No trade to risk-check — pass the proposal through unchanged.
                # Still stamp ``decision_id`` (and ``failure_mode`` when the
                # upstream PM Decision carries one, e.g. REFUSE/ESCALATE) so
                # dashboards filtering "spans with decision_id" do not silently
                # drop heartbeats whose PM produced REFUSE/ESCALATE/HOLD.
                stamp_decision(span, proposal.id, proposal.failure_mode)
                return {"risk_decision": proposal}
            ticker = proposal.payload.ticker
            quote = broker.get_quote(ticker)
            positions = {p.ticker: p.shares for p in broker.list_positions()}
            current_nav = broker.get_cash() + sum(
                (p.shares * broker.get_quote(p.ticker).price for p in broker.list_positions()),
                Decimal("0"),
            )
            # T21 (Plan 4): live inputs for the previously-stubbed risk gates.
            # quote_age_seconds + trades_today are computed against real
            # inputs; daily_pnl_pct stays at 0.0 (see _compute_daily_pnl_pct
            # docstring — no SOD-NAV snapshotter exists yet).
            decision = evaluate_risk(RiskInput(
                proposal=proposal, quote_price=quote.price,
                quote_age_seconds=_compute_quote_age_seconds(quote, clock),
                cash=broker.get_cash(), positions=positions, sector_map=universe.sector_map,
                trades_today=_count_trades_today(db, clock),
                nav=current_nav,
                daily_pnl_pct=_compute_daily_pnl_pct(db, clock, current_nav),
                policy=policy,
            ))
            # T03: propagate the deterministic failure_mode onto the span so
            # dashboards can split RISK_LIMIT_BREACHED vs STALE_DATA vs the
            # null/None happy path without joining back to the decisions row.
            stamp_decision(span, decision.id, decision.failure_mode)
            return {"risk_decision": decision}

    # Build Slack notifier if bot token is present; silent skip otherwise.
    _slack_token = os.environ.get("FIRM_SLACK_BOT_TOKEN")
    _notifier = None
    if _slack_token:
        try:
            from slack_sdk import WebClient as _WebClient  # lazy import — only bites when env requests Slack
            from firm.hitl.notify import SlackNotifier as _SlackNotifier
            _notifier = _SlackNotifier(
                web_client=_WebClient(token=_slack_token),
                channel=policy.hitl.slack_channel,
                approver_id=policy.hitl.slack_approver_id,
                clock=clock,
                internal_secret=nonce_secret,
            )
        except Exception as _exc:
            click.echo(f"[firm run] Slack notifier unavailable: {_exc}", err=True)

    hitl = make_hitl(db_path=db, clock=clock, notifier=_notifier)
    execution = make_execution(
        db_path=db, broker=broker, clock=clock, nonce_secret=nonce_secret
    )
    reporter = make_reporter(reports_root=_reports_root(), clock=clock, db_path=db)

    graph = build_graph(
        db_path=db, monitor_node=monitor, research_node=research, pm_node=pm,
        risk_node=risk_node, hitl_node=hitl, execution_node=execution, reporter_node=reporter,
    )

    def _do_heartbeat(seq: int) -> None:
        # T24 (Plan 4): HITL aging sweep before each heartbeat — reaps any
        # hitl_queue row whose deadline has elapsed (REFUSE / UNAPPROVED_HIGH_RISK).
        # No-op when nothing has aged out.
        reap_expired_hitl_entries(db_path=db, clock=clock, nonce_secret=nonce_secret)

        # Unique thread_id per heartbeat. ``clock.now()`` is fixed under
        # ReplayClock so we suffix with ``seq`` to avoid checkpoint collisions
        # when running multiple ticks in deterministic replay mode.
        thread_id = f"{clock.now().isoformat()}-{seq}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        # Resume an interrupted graph (HITL waiting on approval) rather than
        # restarting. LangGraph: None → resume from checkpoint; dict → fresh run.
        existing = graph.get_state(config)
        invoke_input: dict[str, Any] | None = None if existing.next else {}

        # T03: one outer ``agent.heartbeat`` span so every child node span
        # shares a single trace_id (otherwise each node becomes its own root).
        with agent_span("heartbeat"):
            final = graph.invoke(invoke_input, config=config)
        click.echo(f"Heartbeat #{seq} complete. Report: {final.get('report_path')}")

    if once:
        _do_heartbeat(1)
        return

    # ---- Loop mode -------------------------------------------------------
    # Continuous operation per home-assignment §2.2. SIGINT/SIGTERM stops
    # cleanly after the current heartbeat; second signal exits immediately.
    import signal

    stop = {"requested": False}

    def _on_signal(signum: int, _frame: Any) -> None:
        if stop["requested"]:
            click.echo("\n[firm run] second signal — exiting now.", err=True)
            sys.exit(130)
        stop["requested"] = True
        click.echo(f"\n[firm run] signal {signum} — stopping after current heartbeat.")

    signal.signal(signal.SIGINT, _on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_signal)

    click.echo(
        f"[firm run] loop mode: heartbeat every {interval_seconds}s. Ctrl-C to stop."
    )
    _run_heartbeat_loop(
        _do_heartbeat,
        interval_seconds=interval_seconds,
        should_stop=lambda: stop["requested"],
    )


def _run_heartbeat_loop(
    do_heartbeat: Callable[[int], None],
    *,
    interval_seconds: int,
    should_stop: Callable[[], bool],
    sleep: Callable[[float], None] = _time_module.sleep,
    echo: Callable[..., None] = click.echo,
    failure_threshold: int = _LOOP_FAILURE_THRESHOLD,
) -> int:
    """Drive heartbeats until ``should_stop()`` returns True or breaker trips.

    Per-heartbeat exceptions are logged and the loop continues so a single
    transient failure (broker hiccup, DNS blip) doesn't kill a multi-day run.
    But ``failure_threshold`` consecutive *identical* failures (same exception
    class + same message) re-raise — that signature pattern means the issue is
    structural, not transient, and silent retries just delay the diagnosis.

    Returns the number of heartbeats attempted.
    """
    seq = 0
    last_signature: tuple[str, str] | None = None
    consecutive_failures = 0
    while not should_stop():
        seq += 1
        try:
            do_heartbeat(seq)
            last_signature = None
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001
            signature = (type(exc).__name__, str(exc))
            if signature == last_signature:
                consecutive_failures += 1
            else:
                last_signature = signature
                consecutive_failures = 1
            echo(
                f"[firm run] heartbeat #{seq} failed "
                f"({consecutive_failures}/{failure_threshold}): "
                f"{signature[0]}: {signature[1]}",
                err=True,
            )
            if consecutive_failures >= failure_threshold:
                echo(
                    f"[firm run] {failure_threshold} consecutive identical "
                    f"failures — aborting loop.",
                    err=True,
                )
                raise
        if should_stop():
            break
        # Sleep in 1s ticks so a stop signal is responsive across long intervals.
        slept = 0
        while slept < interval_seconds and not should_stop():
            sleep(1)
            slept += 1

    echo(f"[firm run] stopped after {seq} heartbeats.")
    return seq


def _is_test_environment() -> bool:
    """Return True iff running inside a pytest session (PYTEST_CURRENT_TEST is set)."""
    return "PYTEST_CURRENT_TEST" in os.environ


def _check_dev_ack_gate(dev_ack: bool) -> None:
    """Enforce the --dev-ack gate for non-test environments.

    In test environments (pytest sets PYTEST_CURRENT_TEST) this is a no-op so
    existing tests keep working without any flag.  In production the operator
    must pass --dev-ack to acknowledge that the CLI is a developer-only escape
    hatch; the production-correct path is the Slack /trading-hitl workflow.
    """
    if _is_test_environment():
        return
    if dev_ack:
        return
    click.echo(
        "Use Slack /trading-hitl to approve in production. "
        "Override with --dev-ack if you really mean it."
    )
    sys.exit(1)


@cli.command()
@click.argument("decision_id")
@click.option("--approver", default="cli-user")
@click.option("--dev-ack", is_flag=True, default=False, help="Developer override: bypass Slack gate.")
def ack(decision_id: str, approver: str, dev_ack: bool) -> None:
    """Approve a queued HITL decision (Plan 1 stand-in for Slack)."""
    _check_dev_ack_gate(dev_ack)
    mark_approved(db_path=_db_path(), decision_id=decision_id, approver=approver, clock=_resolve_clock())
    click.echo(f"approved: {decision_id}")


@cli.command()
@click.argument("decision_id")
@click.option("--approver", default="cli-user")
@click.option("--dev-ack", is_flag=True, default=False, help="Developer override: bypass Slack gate.")
def reject(decision_id: str, approver: str, dev_ack: bool) -> None:
    """Reject a queued HITL decision."""
    _check_dev_ack_gate(dev_ack)
    mark_rejected(db_path=_db_path(), decision_id=decision_id, approver=approver, clock=_resolve_clock())
    click.echo(f"rejected: {decision_id}")


@cli.command()
def reconcile() -> None:
    """Run boot reconciliation against the broker and print the result."""
    db = _db_path()
    init_db(db)
    clock = _resolve_clock()
    broker = make_broker(clock=clock)
    _seed_db_from_broker(db, broker, clock)
    result = reconcile_on_boot(db, broker, clock)
    click.echo(f"status: {result.status}")
    if result.diff:
        click.echo(f"diff: {result.diff}")


@cli.command()
@click.option("--date", "date_str", required=True, help="Report date in YYYY-MM-DD format.")
def report(date_str: str) -> None:
    """Generate the daily report bundle for the given date."""
    # Parse and validate date.
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise click.BadParameter(
            f"Expected YYYY-MM-DD, got: {date_str!r}", param_hint="'--date'"
        ) from exc

    db = _db_path()
    clock = _resolve_clock()
    broker = make_broker(clock=clock)
    init_db(db)

    reports_root = _reports_root()
    out_dir = reports_root / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    reconcile_block = render_reconcile_block(db_path=db, broker=broker, clock=clock)

    render_daily_report(
        date=parsed_date,
        db_path=db,
        broker=broker,
        traces_path=out_dir / "traces.jsonl",  # reserved placeholder; T16 doesn't read it
        reports_root=reports_root,
        reconcile_block=reconcile_block,
    )

    render_daily_html(
        date=parsed_date,
        db_path=db,
        broker=broker,
        traces_path=out_dir / "traces.jsonl",
        reports_root=reports_root,
        reconcile_block=reconcile_block,
    )

    as_of = datetime.combine(parsed_date, time.max, tzinfo=timezone.utc)
    write_positions_xlsx(
        path=out_dir / "positions.xlsx",
        broker=broker,
        db_path=db,
        as_of=as_of,
    )

    click.echo(f"Report bundle written: {out_dir}")
    click.echo("  - daily_report.md")
    click.echo("  - daily_report.html")
    click.echo("  - positions.xlsx")


def _make_qdrant_client() -> "Any":
    """QdrantClient backed by QDRANT_LOCAL_PATH (test override) or QDRANT_URL.

    ``QDRANT_URL`` defaults to ``http://localhost:6333`` — the same port the
    bundled ``docker-compose.yml`` publishes for the qdrant service. Lets a
    host-side ``python -m firm.cli ingest`` work out of the box after
    ``docker compose up -d qdrant`` without an explicit env export.
    """
    from qdrant_client import QdrantClient  # lazy import

    local_path = os.environ.get("QDRANT_LOCAL_PATH")
    if local_path:
        return QdrantClient(path=local_path)
    return QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"))


def _make_fixture_loader(fixture_path: str) -> Any:
    """Callable that loads FinanceBench rows from a local JSON file (test override)."""

    def _load() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = json.loads(
            Path(fixture_path).read_text(encoding="utf-8")
        )
        return rows

    return _load


@cli.command()
@click.option(
    "--config",
    default="config/rag.yaml",
    envvar="FIRM_RAG_CONFIG",
    help="Path to rag.yaml config file.",
    show_default=True,
)
@click.option(
    "--max-docs",
    default=None,
    type=int,
    envvar="FIRM_INGEST_MAX_DOCS",
    help="Override corpus.financebench.max_docs from config.",
)
@click.option(
    "--source",
    "source_name",
    type=click.Choice(["financebench", "transcripts", "news", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Which corpus source(s) to ingest. 'all' runs every configured source.",
)
def ingest(config: str, max_docs: int | None, source_name: str) -> None:
    """Ingest one or more corpora into Qdrant. Idempotent — already-indexed docs are skipped."""
    from firm.llm.anthropic_client import CachedAnthropicClient
    from firm.llm.cache import LlmCache
    from firm.rag.chunk import chunk_document
    from firm.rag.contextual import ContextualAugmenter
    from firm.rag.embed import BM25Sparse, NomicEmbedder
    from firm.rag.financebench import FinanceBenchSource
    from firm.rag.ingest import run_ingest
    from firm.rag.news import NewsCorpusSource
    from firm.rag.preprocess import tables_to_prose
    from firm.rag.qdrant_store import VectorStore
    from firm.rag.source import CorpusSource, FilingDoc
    from firm.rag.transcripts import TranscriptsCorpusSource

    db = _db_path()
    init_db(db)
    clock = _resolve_clock()

    rag_config = load_rag_config(Path(config))

    # -----------------------------------------------------------------------
    # Build the list of sources to run, in fixed order (deterministic output).
    # -----------------------------------------------------------------------
    sources: list[CorpusSource] = []

    want_financebench = source_name in ("financebench", "all")
    want_transcripts = source_name in ("transcripts", "all")
    want_news = source_name in ("news", "all")

    if want_financebench:
        # Determine effective max_docs: CLI flag wins, else fall back to config.
        effective_max_docs: int | None = (
            max_docs if max_docs is not None else rag_config.corpus.financebench.max_docs
        )
        # Use fixture loader if env var is set.
        fixture_env = os.environ.get("FIRM_FINANCEBENCH_FIXTURE")
        if fixture_env:
            fb_source: FinanceBenchSource = FinanceBenchSource(
                dataset_loader=_make_fixture_loader(fixture_env)
            )
        else:
            fb_source = FinanceBenchSource()

        if effective_max_docs is not None:
            _limit = effective_max_docs

            class _LimitedSource:
                name: str = fb_source.name

                def iter_docs(self) -> Iterator[FilingDoc]:
                    yield from itertools.islice(fb_source.iter_docs(), _limit)

            sources.append(_LimitedSource())
        else:
            sources.append(fb_source)

    if want_transcripts:
        if rag_config.corpus.transcripts is not None:
            sources.append(
                TranscriptsCorpusSource(path=Path(rag_config.corpus.transcripts.path))
            )
        else:
            click.echo(
                "warning: --source transcripts requested but corpus.transcripts is not "
                "configured in rag.yaml — skipping",
                err=True,
            )

    if want_news:
        news_cfg = rag_config.corpus.news
        if news_cfg is not None and news_cfg.enabled:
            # TODO: wire universe tickers in once firm.yaml exposes them via rag.yaml.
            # For now, pass empty tickers list; the env gate (FIRM_NEWS_ENABLED) and
            # the empty iter handle the no-op case gracefully.
            sources.append(NewsCorpusSource(tickers=[], clock=clock))
        # news is opt-in by design; skip silently when not configured or not enabled.

    if not sources:
        click.echo("No sources configured or selected — nothing to ingest.", err=True)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Qdrant setup (shared collection across all sources).
    # -----------------------------------------------------------------------
    qdrant_client = _make_qdrant_client()
    store = VectorStore(qdrant_client)
    collection = rag_config.qdrant.collection
    dense_dim = rag_config.embedding.dense_dim
    store.create_collection(collection, dense_dim=dense_dim)

    # -----------------------------------------------------------------------
    # Embedders.
    # -----------------------------------------------------------------------
    embedder = NomicEmbedder()
    sparse = BM25Sparse()

    # BM25 pre-pass: collect chunk texts from ALL selected sources so IDF is
    # comparable across corpora. Fit once on the union.
    # NB: news source is iterated twice (pre-pass + run_ingest). With tickers=[]
    # this is free; if T21+ wires real tickers, exclude news from the pre-pass
    # so its TokenBucket budget isn't double-burned.
    click.echo("Building BM25 vocabulary (pre-pass)...")
    all_texts: list[str] = _collect_chunk_texts(sources, rag_config, chunk_document, tables_to_prose)
    if all_texts:
        sparse.fit(all_texts)
    else:
        # Empty corpus: fit on a placeholder so transform() doesn't raise.
        sparse.fit(["placeholder"])

    # LLM client + contextual augmenter.
    llm_cache = LlmCache(db_path=db, clock=clock)
    llm_client = CachedAnthropicClient.from_env(cache=llm_cache, clock=clock)
    augmenter = ContextualAugmenter(
        client=llm_client,
        model=rag_config.contextual.summary_model,
    )

    # -----------------------------------------------------------------------
    # Per-source ingest — one run_ingest call per source.
    # -----------------------------------------------------------------------
    total_docs = 0
    total_chunks = 0
    any_failed = False

    for src in sources:
        result = run_ingest(
            source=src,
            store=store,
            embedder=embedder,
            sparse=sparse,
            augmenter=augmenter,
            db_path=db,
            clock=clock,
            rag_config=rag_config,
        )
        click.echo(
            f"ingest {result.status}: corpus={result.corpus} "
            f"docs_completed={result.docs_completed}/{result.docs_total} "
            f"chunks_written={result.chunks_written}"
        )
        if result.status == "failed":
            click.echo(f"error ({result.corpus}): {result.error}", err=True)
            any_failed = True
        total_docs += result.docs_completed
        total_chunks += result.chunks_written

    click.echo(
        f"total: {len(sources)} corpora, {total_docs} docs, {total_chunks} chunks"
    )
    if any_failed:
        sys.exit(1)


@cli.command()
@click.option(
    "--db",
    "db_path_str",
    default=None,
    envvar="FIRM_DB_PATH",
    help="Path to firm.db (defaults to data/firm.db).",
)
@click.option(
    "--litestream-dir",
    default="data/litestream/firm",
    envvar="FIRM_LITESTREAM_DIR",
    show_default=True,
    help="Path to litestream file-replica directory.",
)
@click.option(
    "--rag-config",
    "rag_config_path_str",
    default=None,
    envvar="FIRM_RAG_CONFIG",
    help="Path to rag.yaml (defaults to config/rag.yaml).",
)
def doctor(db_path_str: str | None, litestream_dir: str, rag_config_path_str: str | None) -> None:
    """Print a one-line health check for each operational concern.

    Exit code equals the number of non-OK checks (0 = fully healthy).
    Ops wires this to a cron job and alerts on any non-zero exit.
    """
    from firm.ops.doctor import format_results, run_doctor

    db = Path(db_path_str) if db_path_str else _db_path()
    init_db(db)
    rag_cfg_path = Path(rag_config_path_str) if rag_config_path_str else _rag_config_path()
    rag_config = load_rag_config(rag_cfg_path)
    collection = rag_config.qdrant.collection

    try:
        qdrant_client: "Any | None" = _make_qdrant_client()
        qdrant_error: str | None = None
    except Exception as exc:
        qdrant_client = None
        qdrant_error = f"{type(exc).__name__}: {exc}"
    clock = _resolve_clock()

    results = run_doctor(
        db_path=db,
        litestream_dir=Path(litestream_dir),
        qdrant_client=qdrant_client,
        qdrant_error=qdrant_error,
        collection_name=collection,
        clock=clock,
    )
    click.echo(format_results(results))
    non_ok = sum(1 for r in results if r.status != "OK")
    sys.exit(non_ok)


@cli.command(name="red-team")
@click.option(
    "--vcr-mode",
    default="replay",
    type=click.Choice(["replay", "record", "live"]),
    show_default=True,
    help="VCR cassette mode passed to FIRM_VCR_MODE.",
)
@click.option(
    "--timeout",
    default=60,
    type=int,
    show_default=True,
    help="Soft wall-clock budget in seconds (warning only, not a hard fail).",
)
def red_team(vcr_mode: str, timeout: int) -> None:
    """Run the red-team injection corpus (50 cases × 10 classes)."""
    import time

    import pytest

    os.environ["FIRM_VCR_MODE"] = vcr_mode

    # 10 per-class test files (T07) — enumerate explicitly to exclude schema/helper files.
    test_files = [
        "tests/red_team/test_direct_override.py",
        "tests/red_team/test_role_hijack.py",
        "tests/red_team/test_delimiter_break.py",
        "tests/red_team/test_unicode_homoglyph.py",
        "tests/red_team/test_encoded_payload.py",
        "tests/red_team/test_indirect_tool_output.py",
        "tests/red_team/test_multi_step_chain.py",
        "tests/red_team/test_citation_forgery.py",
        "tests/red_team/test_spoofed_approval.py",
        "tests/red_team/test_confused_deputy.py",
    ]

    class _CountPlugin:
        passed: int = 0
        failed: int = 0
        total: int = 0

        def pytest_runtest_logreport(self, report: Any) -> None:
            if report.when == "call":
                self.total += 1
                if report.outcome == "passed":
                    self.passed += 1
                elif report.outcome != "skipped":
                    self.failed += 1

    plugin = _CountPlugin()
    start = time.monotonic()
    rc = pytest.main(["-q", *test_files], plugins=[plugin])
    elapsed = time.monotonic() - start

    click.echo(f"{plugin.passed}/{plugin.total} passed")
    if elapsed > timeout:
        click.echo(
            f"WARN: suite took {elapsed:.1f}s > {timeout}s budget", err=True
        )
    if rc != 0 or plugin.failed > 0:
        sys.exit(1)


_EVAL_DEFAULT_ENV: dict[str, str] = {
    # Deterministic defaults for ``firm eval``. Set ONLY if the env var is
    # unset — operator overrides win. The HMAC default is 64 hex zeros; it
    # is fixture-grade and MUST NOT be used in production.
    "FIRM_LLM_MODE": "cached",
    "FIRM_VCR_MODE": "replay",
    "FIRM_PRICES_MODE": "replay",
    "FIRM_RANDOM_SEED": "42",
    "FIRM_HMAC_SECRET": "0" * 64,
}

# Env vars the eval command may mutate. Captured + restored by
# ``_with_env_restored`` so CliRunner.invoke (or any in-process re-entry)
# doesn't leak state — most importantly FIRM_REPORTS_ROOT, which the
# command points into a tmp-scoped artifacts dir that the test harness
# may delete after the invocation.
_EVAL_MUTATED_ENV: tuple[str, ...] = (
    "FIRM_REPORTS_ROOT",
    "FIRM_LLM_MODE",
    "FIRM_VCR_MODE",
    "FIRM_PRICES_MODE",
    "FIRM_RANDOM_SEED",
    "FIRM_HMAC_SECRET",
    "FIRM_EVAL_SKIP_MISCONFIG",
)


def _apply_eval_env_defaults() -> None:
    """Set each :data:`_EVAL_DEFAULT_ENV` key iff the env var is unset.

    Determinism for ``firm eval`` rests on cached LLM / replayed VCR /
    replayed prices / pinned RNG seed. The CLI sets these once at entry
    time so the heartbeat factory + run_regime never see drift between
    calls within the same process.
    """
    for k, v in _EVAL_DEFAULT_ENV.items():
        os.environ.setdefault(k, v)


@contextmanager
def _with_env_restored(*keys: str) -> Iterator[None]:
    """Capture env vars in *keys* on entry, restore them on exit.

    Vars that were absent on entry are deleted on exit (not left as the
    empty string). Used to scope eval-command mutations to the duration
    of the invocation so in-process re-entry (CliRunner.invoke + tests)
    doesn't see stale FIRM_REPORTS_ROOT etc.
    """
    sentinel: object = object()
    prior: dict[str, object] = {k: os.environ.get(k, sentinel) for k in keys}
    try:
        yield
    finally:
        for k, v in prior.items():
            if v is sentinel:
                os.environ.pop(k, None)
            else:
                assert isinstance(v, str)
                os.environ[k] = v


def _load_summary_template() -> Any:
    """Load the ``summary.md.j2`` template from ``firm/reports/templates``."""
    templates_dir = Path(__file__).resolve().parent / "reports" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env.get_template("summary.md.j2")


@cli.command(name="eval")
@click.option(
    "--regime",
    "regime_id",
    type=click.Choice(["r1", "r2", "r3", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Which regime to run (r1/r2/r3) or 'all' for the full sweep.",
)
@click.option(
    "--output-dir",
    "output_dir_str",
    default=None,
    show_default=True,
    help="Where to write per-regime + summary reports (default: reports/eval).",
)
@click.option(
    "--db-dir",
    "db_dir_str",
    default=None,
    help="Where to write per-regime sqlite DBs (default: <output-dir>/_dbs).",
)
def eval_cmd(
    regime_id: str, output_dir_str: str | None, db_dir_str: str | None
) -> None:
    """Run the eval harness on one or all regimes; write reports/eval/.

    Idempotent: deleting the output directory and re-running produces
    byte-identical files. Determinism is pinned by env-var defaults
    (see :data:`_EVAL_DEFAULT_ENV`); operator overrides are respected.

    The body is wrapped in ``_with_env_restored`` so the env vars the
    command mutates (notably ``FIRM_REPORTS_ROOT``) are restored on exit;
    this prevents in-process re-entry (CliRunner.invoke in tests) from
    inheriting a stale path that points into a deleted tmp dir.
    """
    from firm.eval.aggregate import build_summary_context
    from firm.eval.benchmarks import (
        PriceCassetteMissError,
        compute_basket_return,
        compute_spy_return,
    )
    from firm.eval.heartbeat import make_eval_heartbeat
    from firm.eval.regimes import (
        ALL_REGIMES,
        R1_EARNINGS,
        R2_DRAWDOWN,
        R3_QUIET,
        RegimeConfig,
    )
    from firm.eval.runner import RegimeReport, run_regime

    with _with_env_restored(*_EVAL_MUTATED_ENV):
        # Determinism defaults BEFORE any subsystem reads them.
        _apply_eval_env_defaults()

        # Resolve output + db dirs.
        output_dir = (
            Path(output_dir_str)
            if output_dir_str
            else Path("reports") / "eval"
        )
        db_dir = Path(db_dir_str) if db_dir_str else (output_dir / "_dbs")

        # Redirect reporter side-effects into the eval-scoped artifacts dir so
        # ``firm eval`` never mutates ``data/reports/`` from ``firm run``.
        artifacts_root = output_dir / "_artifacts"
        os.environ["FIRM_REPORTS_ROOT"] = str(artifacts_root)

        # Idempotency: nuke + recreate output + db dirs so a second invocation
        # starts from the same blank slate as the first.
        for d in (output_dir, db_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        artifacts_root.mkdir(parents=True, exist_ok=True)

        # Resolve the regime list.
        regime_map: dict[str, RegimeConfig] = {
            "r1": R1_EARNINGS,
            "r2": R2_DRAWDOWN,
            "r3": R3_QUIET,
        }
        if regime_id.lower() == "all":
            regimes_to_run: list[RegimeConfig] = list(ALL_REGIMES)
        else:
            regimes_to_run = [regime_map[regime_id.lower()]]

        # Per-regime: pre-resolve benchmarks at the TOP of the loop so a
        # missing cassette fails the run BEFORE the heartbeat does any work
        # (Plan 4 T16.1 fix-up). Previously a missing cassette emitted a
        # silent 0.0 stub which masked broken determinism gates; the run
        # now raises click.ClickException with operator-actionable guidance.
        prices_dir = Path("data/prices_eval")
        reports: list[RegimeReport] = []
        for regime in regimes_to_run:
            try:
                spy_return = compute_spy_return(
                    regime.start_date,
                    regime.end_date,
                    prices_dir=prices_dir,
                )
            except PriceCassetteMissError as exc:
                raise click.ClickException(
                    f"Price cassette missing for SPY benchmark in regime "
                    f"{regime.regime_id}: {exc}. Run `python "
                    f"firm/ops/eval_capture.py --stub` to generate fixture "
                    f"cassettes, or `python firm/ops/eval_capture.py` (with "
                    f"ANTHROPIC_API_KEY) for production-fidelity recording. "
                    f"See docs/runbook.md §'make eval'."
                ) from exc
            try:
                basket_return = compute_basket_return(
                    list(regime.universe),
                    regime.start_date,
                    regime.end_date,
                    prices_dir=prices_dir,
                )
            except PriceCassetteMissError as exc:
                raise click.ClickException(
                    f"Price cassette missing for basket benchmark in regime "
                    f"{regime.regime_id}: {exc}. Run `python "
                    f"firm/ops/eval_capture.py --stub` to generate fixture "
                    f"cassettes, or `python firm/ops/eval_capture.py` (with "
                    f"ANTHROPIC_API_KEY) for production-fidelity recording. "
                    f"See docs/runbook.md §'make eval'."
                ) from exc

            heartbeat = make_eval_heartbeat(
                regime, reports_root=artifacts_root / regime.regime_id
            )
            report = run_regime(
                regime,
                output_dir=output_dir,
                db_path=db_dir / f"{regime.regime_id}.db",
                heartbeat=heartbeat,
                spy_return=spy_return,
                basket_return=basket_return,
                final_marks={},
            )
            reports.append(report)

        # Cross-regime summary.
        template = _load_summary_template()
        ctx = build_summary_context(reports)
        rendered = template.render(**ctx)
        summary_path = output_dir / "summary.md"
        summary_path.write_bytes(rendered.encode("utf-8"))

        click.echo(
            f"Eval done. {len(reports)} regimes. Summary: {summary_path}"
        )


def _collect_chunk_texts(
    sources: list[Any],
    rag_config: Any,
    chunk_document: Any,
    tables_to_prose: Any,
) -> list[str]:
    """Pre-pass over all selected sources to collect chunk texts for BM25 fitting."""
    all_texts: list[str] = []
    for src in sources:
        for pre_doc in src.iter_docs():
            processed_html = tables_to_prose(pre_doc.html)
            pre_doc_processed = pre_doc.model_copy(update={"html": processed_html})
            chunks = chunk_document(
                pre_doc_processed,
                target_tokens=rag_config.chunk.target_tokens,
                overlap_tokens=rag_config.chunk.overlap_tokens,
                source=src.name,
            )
            all_texts.extend(c.text for c in chunks)
    return all_texts


def _load_dotenv_if_available() -> None:
    """Auto-load ``.env`` from cwd so host-side CLI invocations pick up
    ``ANTHROPIC_API_KEY`` / ``FIRM_LLM_MODE`` / ``FIRM_HMAC_SECRET`` without
    the operator having to source the file manually (cmd.exe and bare
    PowerShell do not). ``override=False`` so an explicit env export still
    wins over .env.

    Invoked only from ``main()``, not at module import, so importing
    ``firm.cli`` from a test does not pollute ``os.environ`` with whatever
    secrets live in the developer's .env. ``python-dotenv`` is in
    ``pyproject.toml`` but the import is guarded so a minimal install still
    runs.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def main() -> None:
    _load_dotenv_if_available()
    cli()


if __name__ == "__main__":
    main()
