"""Router-backed :class:`AnthropicMessagesClient` adapter (Plan 3 T08).

The agent layer (Research extractor + judge, PM voter) constructs each LLM
collaborator with a client implementing the
:class:`firm.llm.citations.AnthropicMessagesClient` Protocol and stashes the
reference at ``__init__`` time. To route each call through the cost-aware
:class:`firm.llm.router.CostRouter` (with its fallback ladder and per-call
ledger writes) without churning the existing extractor / judge / voter shapes,
this module exposes :class:`RouterBackedMessagesClient` — a stateful adapter
the agent rebinds per heartbeat.

Why ``.bind()`` (stateful) rather than a fresh adapter per heartbeat
------------------------------------------------------------------
The extractor / judge / voter take a single ``client=`` in ``__init__`` and
keep it as ``self._client`` for the lifetime of the instance. Constructing
a fresh adapter per heartbeat would require either rebuilding those
collaborators on every tick (wasteful) or threading an adapter-factory
through every call site (intrusive). Mutating ``bind()`` on a long-lived
adapter keeps the LLM client interface unchanged and contains the mutability
to one object owned by a single LangGraph thread per heartbeat.

``messages_create``'s ``model`` / ``max_tokens`` / ``temperature`` arguments
are IGNORED — the router derives those from the profile config and may
downgrade ``max_tokens`` through the fallback ladder. The values the
collaborators pass are vestigial under routing, but kept on the Protocol so
non-router callers (tests, ad-hoc scripts) keep working.
"""
from __future__ import annotations

from firm.core.models import ProfileName
from firm.llm.router import CostRouter


class RouterBackedMessagesClient:
    """Adapter making :class:`CostRouter` satisfy ``AnthropicMessagesClient``.

    Per-heartbeat the agent layer calls :meth:`bind` with the chosen profile,
    the decision id, and the agent name. The next :meth:`messages_create`
    call forwards to :meth:`CostRouter.call_with_fallback` with those bound
    values. The caller's ``model`` / ``max_tokens`` / ``temperature`` are
    ignored — the router derives those from the profile config (and may
    downgrade ``max_tokens`` through the fallback ladder).

    Raises :class:`RuntimeError` on :meth:`messages_create` if :meth:`bind`
    was never called — fail-loud rather than silently routing to a stale or
    default profile.
    """

    def __init__(self, *, router: CostRouter) -> None:
        self._router = router
        self._profile: ProfileName | None = None
        self._decision_id: str | None = None
        self._agent: str | None = None

    def bind(
        self,
        *,
        profile: ProfileName,
        decision_id: str,
        agent: str,
    ) -> None:
        """Set the routing context for the next ``messages_create`` call(s).

        Callable repeatedly; each call replaces the previous binding. The
        agent layer rebinds at the start of every heartbeat before any LLM
        call so cross-heartbeat state cannot leak through this adapter.
        """
        self._profile = profile
        self._decision_id = decision_id
        self._agent = agent

    def messages_create(
        self,
        *,
        model: str,  # noqa: ARG002 — vestigial under routing; see module docstring
        system: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        max_tokens: int,  # noqa: ARG002 — vestigial under routing; see module docstring
        temperature: float,  # noqa: ARG002 — vestigial under routing; see module docstring
    ) -> dict[str, object]:
        if (
            self._profile is None
            or self._decision_id is None
            or self._agent is None
        ):
            raise RuntimeError(
                "RouterBackedMessagesClient.messages_create() called before bind(); "
                "the agent layer must call bind(profile=..., decision_id=..., agent=...) "
                "per heartbeat before invoking any LLM collaborator."
            )
        return self._router.call_with_fallback(
            self._profile,
            system=system,
            messages=messages,
            tools=tools,
            decision_id=self._decision_id,
            agent=self._agent,
        )


__all__ = ["RouterBackedMessagesClient"]
