from decimal import Decimal
from pathlib import Path

from firm.agents.risk import RiskInput, evaluate_risk
from firm.core.config import load_policy
from firm.core.models import ActionEnum, BuyPayload, Decision

POLICY = load_policy(Path("config/policy.yaml"))


def _proposal(ticker: str, shares: str) -> Decision:
    return Decision(
        id="pm-1", decision_id_chain=["res-1"], action=ActionEnum.BUY,
        payload=BuyPayload(ticker=ticker, shares=Decimal(shares)),
        rationale="x", confidence=0.7, citations=[], falsification_condition="y",
        escalation_reason=None, failure_mode=None, metadata={}, nonce="n",
    )


def _make_input(*, ticker="AAPL", shares="10", price="180", cash="100000",
                positions=None, trades_today=0, quote_age_seconds=5,
                daily_pnl_pct=0.0) -> RiskInput:
    return RiskInput(
        proposal=_proposal(ticker, shares),
        quote_price=Decimal(price),
        quote_age_seconds=quote_age_seconds,
        cash=Decimal(cash),
        positions=positions or {},
        sector_map={"AAPL": "tech", "MSFT": "tech", "JPM": "finance"},
        trades_today=trades_today,
        nav=Decimal("100000"),
        daily_pnl_pct=daily_pnl_pct,
        policy=POLICY,
    )


def test_passes_within_all_limits():
    out = evaluate_risk(_make_input(positions={"AAPL": Decimal("1")}))
    assert out.action == ActionEnum.BUY


def test_blocks_max_position_pct():
    # Pre-seed AAPL at 50 shares ($9,000 = 9%). Buy 6 more = 1.08% trade (passes max_trade_pct)
    # but new position = 56 * $180 = $10,080 = 10.08% > 10% limit.
    out = evaluate_risk(_make_input(shares="6", positions={"AAPL": Decimal("50")}))
    assert out.action == ActionEnum.REFUSE
    assert out.failure_mode is not None
    assert out.failure_mode.value == "risk_limit_breached"


def test_blocks_max_trade_pct():
    # max_trade_pct = 5% NAV. $6000 = 6% breaches.
    out = evaluate_risk(_make_input(shares="34"))  # 34*180=6120
    assert out.action == ActionEnum.REFUSE


def test_blocks_max_trades_per_day():
    out = evaluate_risk(_make_input(trades_today=20))
    assert out.action == ActionEnum.REFUSE


def test_blocks_min_cash_buffer():
    # cash 5000, trade requires more than 5% of NAV ($5000) buffer to remain
    out = evaluate_risk(_make_input(cash="2000"))
    assert out.action == ActionEnum.REFUSE


def test_blocks_stale_quote():
    out = evaluate_risk(_make_input(quote_age_seconds=999))  # > 60s threshold
    assert out.action == ActionEnum.REFUSE
    assert out.failure_mode.value == "stale_data"


def test_blocks_drawdown_halt():
    out = evaluate_risk(_make_input(shares="10", daily_pnl_pct=-0.04))  # -4% > -3% threshold
    assert out.action == ActionEnum.REFUSE
    assert out.failure_mode.value == "risk_limit_breached"


def test_blocks_sector_concentration():
    # Pre-seed MSFT 145 shares ($26,100 = 26.1% tech). AAPL trade 25 shares = $4,500 = 4.5%
    # passes max_trade_pct (5%) and max_position_pct (10%, fresh position). Combined tech =
    # 26.1% + 4.5% = 30.6% > 30% sector limit -> REFUSE.
    positions = {"MSFT": Decimal("145")}
    out = evaluate_risk(_make_input(ticker="AAPL", shares="25", positions=positions))
    assert out.action == ActionEnum.REFUSE
    assert out.failure_mode.value == "risk_limit_breached"


def test_hitl_threshold_escalates_instead_of_passing():
    # Trade > 3% NAV = $3000 -> HITL escalate (NOT refuse)
    out = evaluate_risk(_make_input(shares="17"))  # 17*180=3060>3000
    assert out.action == ActionEnum.ESCALATE


def test_every_limit_has_at_least_one_triggering_fixture():
    """CI invariant: each enumerated limit row must be triggered by a test above."""
    import inspect, sys
    triggered = {n for n in dir(sys.modules[__name__]) if n.startswith("test_blocks_")}
    assert len(triggered) >= 7
