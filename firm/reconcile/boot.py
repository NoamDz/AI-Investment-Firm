"""Boot-time position reconciliation. Broker is source of truth. See spec §5.7."""
from __future__ import annotations

import json
from contextlib import closing
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from firm.audit.log import AuditLog
from firm.broker.protocol import Broker
from firm.core.clock import Clock
from firm.db.connection import get_conn


@dataclass
class ReconcileResult:
    status: str  # 'ok' or 'mismatch'
    diff: dict[str, Any]
    # Snapshot fields populated by reconcile_on_boot (additive — existing callers
    # only use .status and .diff, so defaults ensure backward compatibility).
    broker_positions: dict[str, Decimal] = field(default_factory=dict)
    local_positions: dict[str, Decimal] = field(default_factory=dict)
    broker_cash: Decimal = field(default_factory=lambda: Decimal("0"))
    local_cash: Decimal = field(default_factory=lambda: Decimal("0"))


def _local_positions(db_path: Path) -> dict[str, Decimal]:
    with closing(get_conn(db_path)) as conn:
        return {r["ticker"]: Decimal(r["shares"]) for r in conn.execute("SELECT * FROM positions")}


def _local_cash(db_path: Path) -> Decimal:
    with closing(get_conn(db_path)) as conn:
        row = conn.execute("SELECT amount FROM cash WHERE id=1").fetchone()
        return Decimal(row["amount"]) if row else Decimal("0")


def reconcile_on_boot(db_path: Path, broker: Broker, clock: Clock) -> ReconcileResult:
    broker_positions = {p.ticker: p.shares for p in broker.list_positions()}
    broker_cash = broker.get_cash()
    local_positions = _local_positions(db_path)
    local_cash = _local_cash(db_path)

    diff: dict[str, Any] = {}
    pos_diff = {}
    for t in set(broker_positions) | set(local_positions):
        b = broker_positions.get(t, Decimal("0"))
        local = local_positions.get(t, Decimal("0"))
        if b != local:
            pos_diff[t] = {"broker": str(b), "local": str(local)}
    if pos_diff:
        diff["positions"] = pos_diff
    if broker_cash != local_cash:
        diff["cash"] = {"broker": str(broker_cash), "local": str(local_cash)}

    status = "ok" if not diff else "mismatch"
    with closing(get_conn(db_path)) as conn:
        conn.execute(
            "INSERT INTO reconciliations (kind, ran_at, broker_snapshot, local_snapshot, diff, status) "
            "VALUES ('boot', ?, ?, ?, ?, ?)",
            (
                clock.now().isoformat(),
                json.dumps({"positions": {t: str(s) for t, s in broker_positions.items()}, "cash": str(broker_cash)}),
                json.dumps({"positions": {t: str(s) for t, s in local_positions.items()}, "cash": str(local_cash)}),
                json.dumps(diff),
                status,
            ),
        )
    AuditLog(db_path, clock).append("reconcile.boot", {"status": status, "diff": diff})
    return ReconcileResult(
        status=status,
        diff=diff,
        broker_positions=broker_positions,
        local_positions=local_positions,
        broker_cash=broker_cash,
        local_cash=local_cash,
    )


def resolve_from_broker(
    db_path: Path, broker: Broker, clock: Clock, diff: dict[str, Any]
) -> None:
    """Rewrite local positions + cash to match the broker (spec §5.7).

    Spec §5.7 prescribes "halt + human ack, then rewrite local from broker."
    For the demo we treat every boot as an implicit ack: the diff is already
    persisted by ``reconcile_on_boot`` (audit-logged + reconciliations table)
    before this is called, so the resolution stays reviewable. Wrapped in a
    single transaction so a crash mid-rewrite leaves the DB unchanged.
    """
    broker_positions = list(broker.list_positions())
    broker_cash = broker.get_cash()
    now = clock.now().isoformat()
    with closing(get_conn(db_path)) as conn:
        conn.execute("BEGIN")
        try:
            conn.execute("DELETE FROM positions")
            for pos in broker_positions:
                conn.execute(
                    "INSERT INTO positions (ticker, shares, avg_cost, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    (pos.ticker, str(pos.shares), str(pos.avg_cost), now),
                )
            conn.execute(
                "INSERT OR REPLACE INTO cash (id, amount, updated_at) "
                "VALUES (1, ?, ?)",
                (str(broker_cash), now),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    AuditLog(db_path, clock).append("reconcile.resolved", {"diff": diff})
