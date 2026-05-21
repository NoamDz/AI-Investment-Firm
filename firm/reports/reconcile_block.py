"""EOD reconciliation block renderer. See Plan 3 §T17 and spec §5.7."""
from __future__ import annotations

from contextlib import closing
from decimal import Decimal
from pathlib import Path
from typing import Any

from firm.broker.protocol import Broker
from firm.core.clock import Clock
from firm.db.connection import get_conn
from firm.reconcile.boot import reconcile_on_boot


# ---------------------------------------------------------------------------
# Private formatters
# ---------------------------------------------------------------------------


def _format_decimal_shares(value: Decimal) -> str:
    """Stringify shares, stripping trailing zeros for integer-valued Decimals.

    Decimal("100.00") → "100", Decimal("12.5") → "12.5".

    Decimal.normalize() can produce scientific notation (e.g., 1E+2) for
    round-number values, so we use a manual rstrip approach on the string
    representation instead.
    """
    s = str(value)
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _format_position_dict(positions: dict[str, Decimal]) -> str:
    """Format a ticker→shares dict per spec §5.7: { TICKER: SHARES, ... } sorted."""
    if not positions:
        return "{}"
    items = ", ".join(
        f"{ticker}: {_format_decimal_shares(shares)}"
        for ticker, shares in sorted(positions.items())
    )
    return f"{{ {items} }}"


def _format_cash(value: Decimal) -> str:
    """Format cash as $X,XXX.XX (unsigned)."""
    return f"${value:,.2f}"


def _format_signed_cash(value: Decimal) -> str:
    """Format signed cash diff: negative if local > broker, positive if broker > local.

    value = broker_cash - local_cash.
    """
    if value < Decimal("0"):
        return f"-${abs(value):,.2f}"
    return f"${value:,.2f}"


def _format_position_diff(diff_positions: dict[str, Any]) -> str:
    """Format position diff section: { TICKER: broker=X local=Y, ... } sorted."""
    items = ", ".join(
        f"{ticker}: broker={entry['broker']} local={entry['local']}"
        for ticker, entry in sorted(diff_positions.items())
    )
    return f"{{ {items} }}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_reconcile_block(*, db_path: Path, broker: Broker, clock: Clock) -> str:
    """Run reconcile_on_boot and return the §5.7 body block (no header).

    The returned string has no trailing newline; T16's template adds one.
    On mismatch the status line embeds a Markdown footnote reference linking
    to the audit_log row written by reconcile_on_boot.
    """
    result = reconcile_on_boot(db_path, broker, clock)

    # Retrieve the audit_log row id just written — append-only table so the
    # highest id is the one we just wrote.
    audit_id: int | None = None
    audit_ts: str | None = None
    with closing(get_conn(db_path)) as conn:
        row = conn.execute(
            "SELECT id, ts FROM audit_log WHERE event='reconcile.boot' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is not None:
            audit_id = int(row["id"])
            audit_ts = str(row["ts"])

    # Build body lines per spec §5.7 (two-space indent, aligned columns).
    pos_diff_raw: dict[str, Any] = result.diff.get("positions", {})
    pos_diff_str = _format_position_diff(pos_diff_raw) if pos_diff_raw else "none"

    cash_diff = result.broker_cash - result.local_cash

    if result.status == "ok":
        status_line = "✓ books tie to broker"
    else:
        # Footnote reference; safe to use "N" placeholder if table is empty
        # (defensive — should not happen in practice).
        ref = f"[^audit-{audit_id}]" if audit_id is not None else "[^audit-?]"
        status_line = f"❌ MISMATCH {ref}"

    lines: list[str] = [
        f"  Broker positions:   {_format_position_dict(result.broker_positions)}",
        f"  Local positions:    {_format_position_dict(result.local_positions)}",
        f"  Position diff:      {pos_diff_str}",
        f"  Broker cash:        {_format_cash(result.broker_cash)}",
        f"  Local cash:         {_format_cash(result.local_cash)}",
        f"  Cash diff:          {_format_signed_cash(cash_diff)}",
        f"  Status:             {status_line}",
    ]

    body = "\n".join(lines)

    # Append footnote definition on mismatch so the reference is resolvable.
    if result.status == "mismatch" and audit_id is not None and audit_ts is not None:
        footnote = f"[^audit-{audit_id}]: audit_log row {audit_id} (event=reconcile.boot, ts={audit_ts})"
        body = f"{body}\n\n{footnote}"

    return body
