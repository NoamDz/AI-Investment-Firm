"""Risk agent — deterministic hard limits enforcement. See spec §3.7."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from firm.core.config import PolicyConfig
from firm.core.ids import ulid_new
from firm.core.models import (
    ActionEnum, BuyPayload, Decision, EscalatePayload, FailureMode, RefusePayload, SellPayload,
)


@dataclass
class RiskInput:
    proposal: Decision
    quote_price: Decimal
    quote_age_seconds: int
    cash: Decimal
    positions: dict[str, Decimal]      # ticker → shares
    sector_map: dict[str, str]         # ticker → sector
    trades_today: int
    nav: Decimal
    daily_pnl_pct: float
    policy: PolicyConfig


def _decision_from_breach(input: RiskInput, reason: str) -> Decision:
    return Decision(
        id=ulid_new(), decision_id_chain=[input.proposal.id],
        action=ActionEnum.REFUSE,
        payload=RefusePayload(reason=reason),
        rationale=f"hard limit: {reason}", confidence=1.0, citations=[],
        falsification_condition="never (deterministic limit)",
        escalation_reason=None, failure_mode=FailureMode.RISK_LIMIT_BREACHED,
        metadata={"agent": "risk"}, nonce="risk-deterministic",
    )


def _decision_stale(input: RiskInput, reason: str) -> Decision:
    return Decision(
        id=ulid_new(), decision_id_chain=[input.proposal.id],
        action=ActionEnum.REFUSE,
        payload=RefusePayload(reason=reason),
        rationale=f"data freshness violation: {reason}", confidence=1.0, citations=[],
        falsification_condition="never", escalation_reason=None,
        failure_mode=FailureMode.STALE_DATA, metadata={"agent": "risk"}, nonce="risk-stale",
    )


def _escalate(input: RiskInput, reason: str) -> Decision:
    assert isinstance(input.proposal.payload, (BuyPayload, SellPayload))
    return Decision(
        id=ulid_new(), decision_id_chain=[input.proposal.id],
        action=ActionEnum.ESCALATE,
        payload=EscalatePayload(proposed=input.proposal.payload, reason=reason),
        rationale=f"HITL required: {reason}", confidence=1.0, citations=[],
        falsification_condition="HITL approval timeout",
        escalation_reason=reason, failure_mode=None,
        metadata={"agent": "risk"}, nonce="risk-escalate",
    )


def _pass(input: RiskInput) -> Decision:
    return Decision(
        id=ulid_new(), decision_id_chain=[input.proposal.id],
        action=input.proposal.action,
        payload=input.proposal.payload,
        rationale="all hard limits pass", confidence=input.proposal.confidence,
        citations=input.proposal.citations,
        falsification_condition=input.proposal.falsification_condition,
        escalation_reason=None, failure_mode=None,
        metadata={"agent": "risk"}, nonce="risk-pass",
    )


def evaluate_risk(input: RiskInput) -> Decision:
    """Apply every hard limit in policy.yaml. Returns one Decision."""
    p = input.policy
    proposal = input.proposal

    if input.daily_pnl_pct <= -p.limits.max_daily_loss_pct:
        return _decision_from_breach(input, "daily loss drawdown halt")

    if input.quote_age_seconds > p.limits.stale_quote_seconds:
        return _decision_stale(input, f"quote age {input.quote_age_seconds}s")

    if proposal.action == ActionEnum.HOLD:
        return _pass(input)

    if not isinstance(proposal.payload, (BuyPayload, SellPayload)):
        return _pass(input)

    trade_value = input.quote_price * proposal.payload.shares
    trade_pct = float(trade_value / input.nav)

    if input.trades_today >= p.limits.max_trades_per_day:
        return _decision_from_breach(input, "max trades per day")

    if trade_pct > p.limits.max_trade_pct:
        return _decision_from_breach(input, f"trade size {trade_pct:.3f} > {p.limits.max_trade_pct}")

    # Max-position-pct check (after applying the proposed trade)
    ticker = proposal.payload.ticker
    cur_shares = input.positions.get(ticker, Decimal("0"))
    new_shares = cur_shares + proposal.payload.shares if proposal.action == ActionEnum.BUY else cur_shares - proposal.payload.shares
    new_position_value = new_shares * input.quote_price
    pos_pct = float(new_position_value / input.nav)
    if pos_pct > p.limits.max_position_pct:
        return _decision_from_breach(input, f"position {pos_pct:.3f} > {p.limits.max_position_pct}")

    # Min cash buffer (only for buys)
    if proposal.action == ActionEnum.BUY:
        cash_after = input.cash - trade_value
        cash_pct = float(cash_after / input.nav)
        if cash_pct < p.limits.min_cash_pct:
            return _decision_from_breach(input, f"cash buffer {cash_pct:.3f} < {p.limits.min_cash_pct}")

    # Sector concentration. Plan 1: single quote_price applied to all sector positions.
    # Per-ticker pricing arrives with the multi-quote bus in Plan 2.
    sector = input.sector_map.get(ticker, "unknown")
    sector_value = sum(
        input.positions.get(t, Decimal("0")) * input.quote_price
        for t, s in input.sector_map.items() if s == sector
    )
    if proposal.action == ActionEnum.BUY:
        sector_value += proposal.payload.shares * input.quote_price
    sector_pct = float(sector_value / input.nav)
    if sector_pct > p.limits.max_sector_pct:
        return _decision_from_breach(input, f"sector {sector} {sector_pct:.3f} > {p.limits.max_sector_pct}")

    # max_gross_exposure: deferred — Plan 1 has no shorts, so gross == net.
    # Re-enable when Plan 2 introduces short positions.

    # HITL threshold — emit distinct reasons so audit trail is unambiguous
    if trade_pct > p.hitl.trade_threshold_pct:
        return _escalate(input, f"trade size {trade_pct:.3f} > HITL threshold {p.hitl.trade_threshold_pct}")
    if p.hitl.escalate_new_ticker and cur_shares == Decimal("0") and proposal.action == ActionEnum.BUY:
        return _escalate(input, f"new ticker {ticker} (no existing position)")

    return _pass(input)
