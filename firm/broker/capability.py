"""Capability-restricted broker wrapper (Plan 4 T23).

Enforces least-privilege MCP-style tool access: only callers who declare
``agent_role="execution"`` may invoke the order-placing surface
(``submit`` / ``broker.place_order``). All other roles are rejected with
``ToolPermissionDeniedError`` carrying the ``(role, tool)`` pair.

Read-only methods (``get_quote``, ``get_cash``, ``list_positions``) pass
through unconditionally — they are safe for any agent to call. Only the
order-placing surface is gated.

The wrapper structurally satisfies the :class:`firm.broker.protocol.Broker`
Protocol so it is a drop-in replacement for ``FakeBroker`` /
``AlpacaPaperBroker`` at any agent boundary.

Note this is per-call enforcement bound at construction time: a single
``RestrictedBroker`` instance is bound to ONE role.  Cross-agent privilege
laundering (e.g. Research handing its broker reference to Execution) is
out of scope — the orchestrator is responsible for constructing the right
wrapper for each agent.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from firm.broker.protocol import Broker, OrderResult, Position, Quote


# Single source of truth for the privileged-agent allowlist.  Mirrors
# ``tests/red_team/conftest.py::_BROKER_ORDER_AGENTS`` (the red-team
# defense-in-depth test-time invariant).  Both must stay in sync; this
# layer adds production enforcement, the red-team check adds audit-channel
# enforcement.
_PRIVILEGED_AGENTS: frozenset[str] = frozenset({"execution"})


class ToolPermissionDeniedError(PermissionError):
    """Raised when an agent invokes a tool outside its capability allowlist.

    Attributes:
      role: The agent role of the caller (e.g. ``"research"``).
      tool: The tool/method name that was rejected
        (e.g. ``"broker.place_order"``).
    """

    def __init__(self, *, role: str, tool: str) -> None:
        self.role = role
        self.tool = tool
        super().__init__(
            f"agent_role={role!r} is not permitted to invoke tool={tool!r}; "
            f"privileged roles: {sorted(_PRIVILEGED_AGENTS)}"
        )


class RestrictedBroker:
    """Broker wrapper that gates privileged calls by agent role.

    Constructed with the inner broker and the caller's agent role.
    Read-only methods (``list_positions``, ``get_cash``, ``get_quote``)
    pass through unconditionally.  The order-placing surface (``submit``)
    raises :class:`ToolPermissionDeniedError` unless the role is in
    :data:`_PRIVILEGED_AGENTS`.

    Structurally satisfies :class:`firm.broker.protocol.Broker`, so it can
    be used anywhere the protocol is expected.
    """

    # The conceptual tool name exposed to the capability layer.  We
    # deliberately use ``broker.place_order`` rather than the underlying
    # protocol method name (``submit``) because the red-team /
    # audit-log convention everywhere else in the codebase
    # (``CallLoggingBroker``, ``assert_no_privileged_action``) uses
    # ``"place_order"`` as the externally-visible tool identifier.  Keeping
    # them aligned makes the security audit trivial.
    _PLACE_ORDER_TOOL: str = "broker.place_order"

    def __init__(self, inner: Broker, *, agent_role: str) -> None:
        self._inner = inner
        self._agent_role = agent_role

    # ------------------------------------------------------------------
    # Read-only methods — pass through unconditionally.
    # ------------------------------------------------------------------

    def list_positions(self) -> list[Position]:
        return self._inner.list_positions()

    def get_cash(self) -> Decimal:
        return self._inner.get_cash()

    def get_quote(self, ticker: str) -> Quote:
        return self._inner.get_quote(ticker)

    # ------------------------------------------------------------------
    # Privileged method — capability check before delegation.
    # ------------------------------------------------------------------

    def submit(
        self,
        decision_payload: dict[str, Any],
        idempotency_key: str,
    ) -> OrderResult:
        """Place a broker order.

        Raises:
          ToolPermissionDeniedError: if the bound ``agent_role`` is not in
            :data:`_PRIVILEGED_AGENTS`.  The inner broker is NOT called in
            this case — rejection happens before any side effect reaches
            the broker.
        """
        if self._agent_role not in _PRIVILEGED_AGENTS:
            raise ToolPermissionDeniedError(
                role=self._agent_role,
                tool=self._PLACE_ORDER_TOOL,
            )
        return self._inner.submit(decision_payload, idempotency_key)


__all__ = [
    "RestrictedBroker",
    "ToolPermissionDeniedError",
]
