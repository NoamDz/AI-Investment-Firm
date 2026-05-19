"""Transactional outbox for broker orders. See design spec §5.2."""
from __future__ import annotations

import hashlib
from contextlib import closing
from pathlib import Path

from firm.broker.protocol import Broker, OrderResult
from firm.core.clock import Clock
from firm.core.models import Decision
from firm.db.connection import get_conn


def _idempotency_key(decision: Decision) -> str:
    """sha256(decision.id:decision.nonce). Must match key sent to broker.submit."""
    return hashlib.sha256(f"{decision.id}:{decision.nonce}".encode()).hexdigest()


def place_order_via_outbox(
    decision: Decision, db_path: Path, broker: Broker, clock: Clock
) -> OrderResult:
    """Place an order with exactly-once semantics. See spec §5.2 crash semantics."""
    key = _idempotency_key(decision)
    now = clock.now().isoformat()

    with closing(get_conn(db_path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO outbox (key, decision_id, payload, status, created_at, updated_at) "
                "VALUES (?, ?, ?, 'pending', ?, ?) ON CONFLICT (key) DO NOTHING",
                (key, decision.id, decision.model_dump_json(), now, now),
            )
            row = conn.execute(
                "SELECT status, result FROM outbox WHERE key = ?", (key,)
            ).fetchone()
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        if row["status"] == "confirmed":
            return OrderResult.model_validate_json(row["result"])

        result = broker.submit(decision.payload.model_dump(mode="json"), idempotency_key=key)

        conn.execute(
            "UPDATE outbox SET status='confirmed', result=?, updated_at=? WHERE key=?",
            (result.model_dump_json(), clock.now().isoformat(), key),
        )
        return result


def recover_pending(db_path: Path, broker: Broker, clock: Clock) -> list[OrderResult]:
    """On boot, drive any `pending` outbox rows to terminal status."""
    with closing(get_conn(db_path)) as conn:
        rows = conn.execute(
            "SELECT key, decision_id, payload FROM outbox WHERE status='pending'"
        ).fetchall()
        results: list[OrderResult] = []
        for r in rows:
            decision = Decision.model_validate_json(r["payload"])
            result = broker.submit(decision.payload.model_dump(mode="json"), idempotency_key=r["key"])
            conn.execute(
                "UPDATE outbox SET status='confirmed', result=?, updated_at=? WHERE key=?",
                (result.model_dump_json(), clock.now().isoformat(), r["key"]),
            )
            results.append(result)
        return results
