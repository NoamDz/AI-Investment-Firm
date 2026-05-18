"""Boot-time position reconciliation. Broker is source of truth. See spec §5.7."""
from __future__ import annotations

import json
from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from firm.audit.log import AuditLog
from firm.broker.protocol import Broker
from firm.core.clock import Clock
from firm.db.connection import get_conn


@dataclass
class ReconcileResult:
    status: str  # 'ok' or 'mismatch'
    diff: dict


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

    diff: dict = {}
    pos_diff = {}
    for t in set(broker_positions) | set(local_positions):
        b = broker_positions.get(t, Decimal("0"))
        l = local_positions.get(t, Decimal("0"))
        if b != l:
            pos_diff[t] = {"broker": str(b), "local": str(l)}
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
    return ReconcileResult(status=status, diff=diff)
