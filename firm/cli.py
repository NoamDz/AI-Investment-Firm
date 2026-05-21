"""CLI entry points. See spec §3.1, §3.8."""
from __future__ import annotations

import functools
import itertools
import json
import os
import sys
from contextlib import closing
from datetime import datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import click

from firm.agents.execution import make_execution
from firm.agents.hitl import make_hitl, mark_approved, mark_rejected
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
from firm.reports.reconcile_block import render_reconcile_block
from firm.reports.xlsx import write_positions_xlsx

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
@click.option("--once/--loop", default=True, help="Single heartbeat (default) or loop (loop is Plan 3+).")
def run(once: bool) -> None:
    """Run one heartbeat of the firm end-to-end."""
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
            # TODO(Plan 2): wire live quote_age_seconds / trades_today / daily_pnl_pct; stubs disable those checks
            decision = evaluate_risk(RiskInput(
                proposal=proposal, quote_price=quote.price, quote_age_seconds=0,
                cash=broker.get_cash(), positions=positions, sector_map=universe.sector_map,
                trades_today=0, nav=broker.get_cash() + sum((p.shares * broker.get_quote(p.ticker).price for p in broker.list_positions()), Decimal("0")),
                daily_pnl_pct=0.0, policy=policy,
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
    execution = make_execution(db_path=db, broker=broker, clock=clock)
    reporter = make_reporter(reports_root=_reports_root(), clock=clock, db_path=db)

    graph = build_graph(
        db_path=db, monitor_node=monitor, research_node=research, pm_node=pm,
        risk_node=risk_node, hitl_node=hitl, execution_node=execution, reporter_node=reporter,
    )

    config: RunnableConfig = {"configurable": {"thread_id": clock.now().isoformat()}}

    # Resume an interrupted graph (e.g. pending HITL approval) rather than
    # restarting it with a fresh empty-state invoke.  LangGraph distinguishes
    # "resume" from "new run" by the input argument: None → resume from the
    # saved checkpoint; dict → start a new run (clobbering the checkpoint).
    # T31 surfaced that invoke({}) always restarts, so approved HITL decisions
    # were never processed — the graph re-entered research and hit the interrupt
    # again instead of continuing to execution.
    existing = graph.get_state(config)
    if existing.next:
        # Pending checkpoint: resume (HITL gate waiting for approval, etc.).
        invoke_input: dict[str, Any] | None = None
    else:
        # No checkpoint or completed checkpoint: start a fresh heartbeat.
        invoke_input = {}

    # T03: wrap a single heartbeat in one outer ``agent.heartbeat`` span so
    # every child node span (research/pm/risk/reporter + their llm/retrieval
    # children) shares a single trace_id.  Without this each node becomes its
    # own trace root and a heartbeat fragments into 4+ disjoint traces.
    with agent_span("heartbeat"):
        final = graph.invoke(invoke_input, config=config)
    click.echo(f"Heartbeat complete. Report: {final.get('report_path')}")


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

    as_of = datetime.combine(parsed_date, time.max, tzinfo=timezone.utc)
    write_positions_xlsx(
        path=out_dir / "positions.xlsx",
        broker=broker,
        db_path=db,
        as_of=as_of,
    )

    click.echo(f"Report bundle written: {out_dir}")


def _make_qdrant_client() -> "Any":
    """QdrantClient backed by QDRANT_LOCAL_PATH (test override) or QDRANT_URL."""
    from qdrant_client import QdrantClient  # lazy import

    local_path = os.environ.get("QDRANT_LOCAL_PATH")
    if local_path:
        return QdrantClient(path=local_path)
    return QdrantClient(url=os.environ["QDRANT_URL"])


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
    from collections.abc import Iterator

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


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
