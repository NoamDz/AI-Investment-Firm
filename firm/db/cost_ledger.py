"""Per-LLM-call cost ledger writer. See Plan 3 §10.2 / T09.

The ledger is append-only: one row per successful ``messages_create`` call
(cached or live). The :class:`firm.llm.router.CostRouter` invokes
:func:`write_cost_ledger_row` after each successful attempt; failed attempts
(transient errors that trigger the fallback ladder) are NOT logged here to
keep the spec's "one row per LLM call" wording precise. Failure-mode
attribution lives on the per-attempt OTel span (T03 / T04).

Column convention (mirrors T04's ``_stamp_llm_cost`` semantics):

* **Cached call** -> ``input_tokens=None``, ``output_tokens=None``,
  ``cached_tokens=<sum of cached usage>``, ``cost_usd=0.0``.
* **Live call** -> ``input_tokens=<int>``, ``output_tokens=<int>``,
  ``cached_tokens=None``, ``cost_usd=<computed from rate card>``.

Schema is declared in ``firm/db/schema.sql``; see also
:func:`firm.db.migrations.init_db`.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path

from firm.core.clock import Clock
from firm.db.connection import get_conn


def write_cost_ledger_row(
    *,
    db_path: Path,
    decision_id: str,
    agent: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_tokens: int | None,
    cost_usd: float,
    clock: Clock,
) -> None:
    """Append one row to the ``cost_ledger`` table.

    Uses a short-lived connection (mirroring :class:`firm.audit.log.AuditLog`)
    so the writer is safe to call from any thread without coordinating a
    shared connection lifetime.
    """
    created_at = clock.now().isoformat()
    with closing(get_conn(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO cost_ledger (
                decision_id, agent, model,
                input_tokens, output_tokens, cached_tokens,
                cost_usd, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                agent,
                model,
                input_tokens,
                output_tokens,
                cached_tokens,
                cost_usd,
                created_at,
            ),
        )


__all__ = ["write_cost_ledger_row"]
