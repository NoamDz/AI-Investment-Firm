"""CLI entry points. See spec §3.1, §3.8."""
from __future__ import annotations

import os
import sys
from contextlib import closing
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import click

from firm.agents.execution import make_execution
from firm.agents.hitl import make_hitl, mark_approved, mark_rejected
from firm.agents.monitor import make_monitor
from firm.agents.pm import make_pm
from firm.agents.reporter import make_reporter
from firm.agents.research import make_research
from firm.agents.risk import RiskInput, evaluate_risk
from firm.broker.alpaca_paper import make_broker
from firm.broker.protocol import Broker
from firm.core.clock import Clock, ReplayClock, WallClock
from firm.core.config import load_policy, load_universe
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.orchestrator.graph import build_graph
from firm.orchestrator.state import WorkingState
from firm.reconcile.boot import reconcile_on_boot

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

    _seed_db_from_broker(db, broker, clock)
    recon = reconcile_on_boot(db, broker, clock)
    if recon.status == "mismatch":
        click.echo(f"BOOT RECONCILIATION MISMATCH: {recon.diff}", err=True)
        click.echo("Resolve the mismatch and re-run.", err=True)
        sys.exit(1)

    monitor = make_monitor(clock)
    research = make_research(clock=clock, broker=broker, universe=universe)
    pm = make_pm()

    def risk_node(state: WorkingState) -> dict[str, Any]:
        proposal = state["pm_decision"]
        ticker = proposal.payload.ticker if hasattr(proposal.payload, "ticker") else "AAPL"
        quote = broker.get_quote(ticker)
        positions = {p.ticker: p.shares for p in broker.list_positions()}
        # TODO(Plan 2): wire live quote_age_seconds / trades_today / daily_pnl_pct; stubs disable those checks
        decision = evaluate_risk(RiskInput(
            proposal=proposal, quote_price=quote.price, quote_age_seconds=0,
            cash=broker.get_cash(), positions=positions, sector_map=universe.sector_map,
            trades_today=0, nav=broker.get_cash() + sum((p.shares * broker.get_quote(p.ticker).price for p in broker.list_positions()), Decimal("0")),
            daily_pnl_pct=0.0, policy=policy,
        ))
        return {"risk_decision": decision}

    hitl = make_hitl(db_path=db, clock=clock)
    execution = make_execution(db_path=db, broker=broker, clock=clock)
    reporter = make_reporter(reports_root=_reports_root(), clock=clock, db_path=db)

    graph = build_graph(
        db_path=db, monitor_node=monitor, research_node=research, pm_node=pm,
        risk_node=risk_node, hitl_node=hitl, execution_node=execution, reporter_node=reporter,
    )

    config: RunnableConfig = {"configurable": {"thread_id": clock.now().isoformat()}}
    final = graph.invoke({}, config=config)
    click.echo(f"Heartbeat complete. Report: {final.get('report_path')}")


@cli.command()
@click.argument("decision_id")
@click.option("--approver", default="cli-user")
def ack(decision_id: str, approver: str) -> None:
    """Approve a queued HITL decision (Plan 1 stand-in for Slack)."""
    mark_approved(db_path=_db_path(), decision_id=decision_id, approver=approver, clock=_resolve_clock())
    click.echo(f"approved: {decision_id}")


@cli.command()
@click.argument("decision_id")
@click.option("--approver", default="cli-user")
def reject(decision_id: str, approver: str) -> None:
    """Reject a queued HITL decision."""
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


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
