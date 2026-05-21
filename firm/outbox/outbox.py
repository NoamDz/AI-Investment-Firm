"""Transactional outbox for broker orders. See design spec §5.2."""
from __future__ import annotations

import hashlib
from contextlib import closing
from pathlib import Path

from firm.broker.protocol import Broker, OrderResult
from firm.core.clock import Clock
from firm.core.models import Decision
from firm.db.connection import get_conn


# Spec §10.5 line 222: "After N retries, abort decision and surface unfilled
# order in EOD report".  N is small + in-heartbeat — long retry budgets
# delay the next heartbeat's risk re-check; ops can later swap in a
# backoff/jitter strategy if real latency profiles demand it (separate task).
_DEFAULT_MAX_BROKER_ATTEMPTS = 3


class BrokerUnavailableError(RuntimeError):
    """Broker submit failed after N attempts; outbox row stays 'pending'.

    Caller (execution agent) is responsible for emitting a REFUSE Decision
    stamped with :attr:`firm.core.models.FailureMode.BROKER_UNAVAILABLE` so
    the heartbeat surfaces the unfilled order in the audit log + EOD report
    (spec §10.5).  The outbox row keyed by ``idempotency_key`` is
    intentionally left ``'pending'`` so the next heartbeat's
    :func:`recover_pending` retries it with the SAME key (spec §5.2
    invariant) when the broker comes back.
    """

    def __init__(
        self, *, idempotency_key: str, attempts: int, underlying: BaseException
    ) -> None:
        super().__init__(
            f"broker.submit failed after {attempts} attempts "
            f"(idempotency_key={idempotency_key!r}); outbox row remains pending. "
            f"Underlying: {type(underlying).__name__}: {underlying}"
        )
        self.idempotency_key = idempotency_key
        self.attempts = attempts
        self.underlying = underlying


def _idempotency_key(decision: Decision) -> str:
    """sha256(decision.id:decision.nonce). Must match key sent to broker.submit."""
    return hashlib.sha256(f"{decision.id}:{decision.nonce}".encode()).hexdigest()


def place_order_via_outbox(
    decision: Decision,
    db_path: Path,
    broker: Broker,
    clock: Clock,
    *,
    max_attempts: int = _DEFAULT_MAX_BROKER_ATTEMPTS,
) -> OrderResult:
    """Place an order with exactly-once semantics. See spec §5.2 crash semantics.

    The outbox row is INSERTed inside a transaction that commits BEFORE the
    first ``broker.submit`` attempt — this is what makes the recovery story
    work after a crash mid-submit.  After the insert, attempt
    ``broker.submit`` up to ``max_attempts`` times.  Every attempt re-uses
    the SAME idempotency key (spec §5.2 invariant — broker dedupes by key).
    No sleep/backoff between attempts: the heartbeat budget is fixed, and
    spinning fast keeps test fixtures synchronous; ops can layer backoff
    later as a separate task.

    On success: UPDATE the row to ``'confirmed'`` with the broker result
    and return the :class:`OrderResult`.

    On exhaustion: raise :class:`BrokerUnavailableError`.  The outbox row
    intentionally stays ``'pending'`` so :func:`recover_pending` can retry
    it on the next heartbeat with the same key.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

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

        # Bounded retry loop. Spec §5.2 invariant: every attempt MUST use the
        # SAME idempotency_key so the broker can dedupe its own state on retry.
        # On exhaustion we DO NOT UPDATE the row — it stays 'pending' so the
        # next heartbeat's recover_pending picks it up.
        payload_dump = decision.payload.model_dump(mode="json")
        last_exc: BaseException | None = None
        for _ in range(max_attempts):
            try:
                result = broker.submit(payload_dump, idempotency_key=key)
            except Exception as exc:  # noqa: BLE001 -- broker exceptions are opaque/3rd-party
                last_exc = exc
                continue

            conn.execute(
                "UPDATE outbox SET status='confirmed', result=?, updated_at=? WHERE key=?",
                (result.model_dump_json(), clock.now().isoformat(), key),
            )
            return result

        # All attempts exhausted; surface the typed exception so the
        # execution agent can emit a REFUSE BROKER_UNAVAILABLE Decision.
        assert last_exc is not None  # loop ran at least once (max_attempts >= 1)
        raise BrokerUnavailableError(
            idempotency_key=key,
            attempts=max_attempts,
            underlying=last_exc,
        )


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
