from decimal import Decimal
from firm.agents.research import make_research
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.config import load_universe
from datetime import datetime, timezone
from pathlib import Path


def test_research_returns_buy_decision_for_universe_ticker():
    broker = FakeBroker(initial_cash=Decimal("100000"))
    universe = load_universe(Path("config/universe.yaml"))
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    research = make_research(clock=clock, broker=broker, universe=universe)
    state = research({"heartbeat_at": "2024-03-13T14:30:00+00:00"})
    d = state["research_decision"]
    assert d.action.value == "BUY"
    assert d.payload.ticker in universe.tickers
    assert d.payload.shares == Decimal("10")
    assert "stub" in d.rationale.lower()
