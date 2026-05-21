"""Cost-aware LLM router with fallback ladder (Plan 3 T07).

:class:`CostRouter` chooses an Anthropic model *profile* per request
(:meth:`route_for_decision`) and dispatches the call through a fallback
ladder (:meth:`call_with_fallback`):

1. **Primary attempt** — use the profile's ``model_id`` /
   ``max_tokens`` / ``temperature``.
2. **Same-profile retry with truncated documents** — drop ~50% of
   Anthropic Citations ``document`` content blocks (keep the first
   ``ceil(N * 0.5)``) and retry the SAME profile once.
3. **Downgrade** — fall to the next profile in
   :attr:`RouterConfig.fallback_chain` with ``max_tokens *= 0.5``.
   Each downgraded profile gets exactly one attempt (no further
   truncation pass); the ladder is otherwise iterated to completion.
4. **Exhausted** — raise :class:`LLMUnavailableError` with a tail of the
   underlying error reprs.

The primary's ``max_tokens *= 0.5`` interpretation: applied to the
DESTINATION profile's own ``max_tokens``, not the originating primary's.
The spec wording "downgrades to Haiku with ``max_tokens *= 0.5``" is
ambiguous between the two readings; we chose this one because it (a)
respects each profile's documented per-request budget as the upper
bound, (b) is symmetric across primaries (sonnet→haiku and opus→sonnet
both halve the destination's headroom rather than tying behavior to the
caller), and (c) avoids the degenerate case where a primary with a
small ``max_tokens`` would force the downgraded profile to use an even
smaller (sometimes < 1) effective token budget.

Observability: each underlying ``messages_create`` call runs inside its
own :func:`firm.obs.llm_span`, tagged with the per-attempt ``model`` and
a ``fallback_attempt`` integer (0 = primary, 1 = truncated retry, 2+ =
downgrades). This makes the cost/token attribution stamped by
:class:`firm.llm.anthropic_client.CachedAnthropicClient` land on the
correct attempt span rather than collapsing onto the outer caller span.

Exception policy: any exception except :class:`LLMUnavailableError`
(don't re-fallback our own raise) and
:class:`firm.llm.anthropic_client.LlmCacheMissError` (a logic error in
CACHED mode, not a transient failure) triggers the next ladder step.
The two named exceptions are re-raised immediately.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from firm.core.config import RouterConfig
from firm.core.models import ProfileName, RouterFeatures
from firm.llm.anthropic_client import (
    CachedAnthropicClient,
    LlmCacheMissError,
)
from firm.obs import llm_span


class LLMUnavailableError(Exception):
    """Raised after the fallback ladder is exhausted — no profile served the call."""


@dataclass(frozen=True)
class ProfileChoice:
    """Result of :meth:`CostRouter.route_for_decision`.

    ``primary`` is the profile name that :meth:`RouterFeatures.score`
    returned; ``ladder`` is the ordered tuple of profile names the
    router will try in :meth:`CostRouter.call_with_fallback` (primary
    first, then ``fallback_chain`` entries, with duplicates removed
    while preserving first-occurrence order).
    """

    primary: ProfileName
    ladder: tuple[ProfileName, ...]


# Maximum number of underlying calls per ladder profile.  The primary
# profile gets 2 (initial + truncated-retry); downgraded profiles get 1.
# Kept as constants (rather than dispersed magic numbers) so the limit is
# easy to find and tune in one place.
_PRIMARY_ATTEMPTS = 2
_DOWNGRADE_ATTEMPTS = 1
_DOWNGRADE_TOKEN_FRACTION = 0.5


class CostRouter:
    """Routes one LLM call to a profile and retries down a fallback ladder.

    Construct once per process (or per run) with the loaded
    :class:`RouterConfig` and a configured
    :class:`CachedAnthropicClient`; call :meth:`route_for_decision` to
    pick a profile and :meth:`call_with_fallback` to dispatch.
    """

    def __init__(
        self,
        *,
        router_cfg: RouterConfig,
        anthropic_client: CachedAnthropicClient,
    ) -> None:
        self._router_cfg = router_cfg
        self._client = anthropic_client

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def route_for_decision(self, features: RouterFeatures) -> ProfileChoice:
        """Return the :class:`ProfileChoice` for *features*.

        The primary is the result of
        :meth:`RouterFeatures.score(router_cfg.weights)`. The ladder is
        ``(primary, *router_cfg.fallback_chain)`` with duplicates removed
        while preserving order (so a primary that already appears in
        ``fallback_chain`` does not waste an attempt on itself).
        """
        primary: ProfileName = features.score(self._router_cfg.weights)
        seen: set[ProfileName] = set()
        ladder: list[ProfileName] = []
        for name in (primary, *self._router_cfg.fallback_chain):
            if name in seen:
                continue
            seen.add(name)
            ladder.append(name)
        return ProfileChoice(primary=primary, ladder=tuple(ladder))

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------
    def call_with_fallback(
        self,
        profile: ProfileName,
        system: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        """Dispatch one call through the fallback ladder rooted at *profile*.

        Returns the raw Anthropic response dict (with ``_cache_hit``
        attached by the lower layer). Raises :class:`LLMUnavailableError`
        once every profile in the ladder has failed.
        """
        # Recompute the ladder rooted at *profile*. Callers usually pass
        # the same name they got from :meth:`route_for_decision`, but the
        # API accepts an arbitrary profile name so we don't assume.
        ladder = self._ladder_rooted_at(profile)

        errors: list[Exception] = []
        attempt_idx = 0  # cumulative across profiles, stamped on spans.

        for ladder_pos, profile_name in enumerate(ladder):
            profile_cfg = self._lookup_profile(profile_name)

            if ladder_pos == 0:
                # Primary profile: try original messages, then truncated.
                messages_for_attempt = messages
                max_tokens = profile_cfg.max_tokens
                attempts_for_this_profile = _PRIMARY_ATTEMPTS
            else:
                # Downgraded profile: single attempt with halved max_tokens.
                messages_for_attempt = messages
                # int() truncates toward zero; max(1, ...) guarantees we
                # never call with max_tokens=0 even if the destination
                # profile has max_tokens=1 (which would never pass T06
                # validation but we stay defensive).
                max_tokens = max(1, int(profile_cfg.max_tokens * _DOWNGRADE_TOKEN_FRACTION))
                attempts_for_this_profile = _DOWNGRADE_ATTEMPTS

            for sub_attempt in range(attempts_for_this_profile):
                # On the SECOND attempt of the primary profile, truncate
                # document blocks. Only ever applied once per call_with_
                # fallback invocation.
                if ladder_pos == 0 and sub_attempt == 1:
                    messages_for_attempt = _truncate_document_blocks(
                        messages, fraction=0.5
                    )

                try:
                    return self._invoke(
                        profile_name=profile_name,
                        model_id=profile_cfg.model_id,
                        system=system,
                        messages=messages_for_attempt,
                        tools=tools,
                        max_tokens=max_tokens,
                        temperature=profile_cfg.temperature,
                        attempt_idx=attempt_idx,
                    )
                except (LLMUnavailableError, LlmCacheMissError):
                    # Don't fallback over our own raise or a CACHED-mode
                    # cache miss (which is a logic error, not transient).
                    raise
                except Exception as exc:  # noqa: BLE001 — intentional broad catch
                    errors.append(exc)
                    attempt_idx += 1
                    # Fall through to the next sub_attempt or next ladder
                    # profile.

        # Ladder exhausted.
        tail = ", ".join(repr(e) for e in errors[-3:])
        raise LLMUnavailableError(
            f"all profiles exhausted for primary={profile}: ladder={ladder} "
            f"last_errors=[{tail}]"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ladder_rooted_at(self, profile: ProfileName) -> tuple[ProfileName, ...]:
        """Compute the ladder for an arbitrary primary profile name.

        Mirrors :meth:`route_for_decision`'s dedup logic so callers that
        bypass routing (e.g. a forced profile) still get the same
        fallback semantics.
        """
        seen: set[ProfileName] = set()
        ladder: list[ProfileName] = []
        for name in (profile, *self._router_cfg.fallback_chain):
            if name in seen:
                continue
            seen.add(name)
            ladder.append(name)
        return tuple(ladder)

    def _lookup_profile(self, name: ProfileName):
        """Look up a profile by name; raise KeyError if absent.

        T06's :class:`RouterConfig` validator guarantees the three
        canonical profile names are present, so this only fires on a
        truly malformed config that bypassed the loader.
        """
        try:
            return self._router_cfg.profiles[name]
        except KeyError as exc:
            raise KeyError(
                f"router profile {name!r} not declared in router.yaml — "
                f"known profiles: {sorted(self._router_cfg.profiles)}"
            ) from exc

    def _invoke(
        self,
        *,
        profile_name: ProfileName,
        model_id: str,
        system: str,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        max_tokens: int,
        temperature: float,
        attempt_idx: int,
    ) -> dict[str, object]:
        """Issue one underlying ``messages_create`` call inside its own span.

        The span is a child of whatever span is currently active (T03
        wraps each agent op in its own :func:`llm_span`); we open a
        NESTED ``llm_span`` so the per-attempt model id and
        ``fallback_attempt`` attribute are recorded on a dedicated span
        rather than overwritten on the outer one.
        """
        with llm_span("anthropic", model_id) as span:
            span.set_attribute("fallback_attempt", attempt_idx)
            span.set_attribute("profile", profile_name)
            return self._client.messages_create(
                model=model_id,
                system=system,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_document_blocks(
    messages: list[dict[str, object]], *, fraction: float = 0.5
) -> list[dict[str, object]]:
    """Return a NEW message list with the last *fraction* of document blocks dropped.

    Only blocks whose ``type == "document"`` (Anthropic Citations API
    chunks) are affected; text / tool_use / etc. blocks pass through
    unchanged in their original positions. The first ``ceil(N * (1 -
    fraction))`` documents are kept, where ``N`` is the original count;
    if there are no document blocks the messages are returned
    structurally copied but logically unchanged. At least one document
    block is always kept when ``N >= 1`` (so a 1-document message stays
    1-document, not 0).

    The input list and its nested dicts are NOT mutated — callers can
    pass the same ``messages`` value into both the original call and
    the truncated retry safely.
    """
    if not 0.0 < fraction < 1.0:
        # Defensive: caller should never request a no-op or full-drop.
        return [dict(m) for m in messages]

    new_messages: list[dict[str, object]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            # Non-block content (e.g. plain-string user messages): pass
            # through untouched.
            new_messages.append(dict(msg))
            continue

        doc_indices = [
            i
            for i, blk in enumerate(content)
            if isinstance(blk, dict) and blk.get("type") == "document"
        ]
        n_docs = len(doc_indices)
        if n_docs == 0:
            new_messages.append(dict(msg))
            continue

        # Keep ceil(N * (1 - fraction)); always >= 1 for N >= 1.
        keep = max(1, math.ceil(n_docs * (1.0 - fraction)))
        keep_set = set(doc_indices[:keep])
        new_content: list[object] = []
        for i, blk in enumerate(content):
            if isinstance(blk, dict) and blk.get("type") == "document":
                if i in keep_set:
                    new_content.append(blk)
                # else: drop this document block.
            else:
                new_content.append(blk)
        new_msg = dict(msg)
        new_msg["content"] = new_content
        new_messages.append(new_msg)

    return new_messages


__all__ = [
    "CostRouter",
    "LLMUnavailableError",
    "ProfileChoice",
]
