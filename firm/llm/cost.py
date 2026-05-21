"""Per-call USD cost computation from the ``config/router.yaml`` rate card.

Plan 3 T04: the :class:`firm.llm.anthropic_client.CachedAnthropicClient` stamps
``cost_usd`` onto the active OTel span on every live call.  The rate card
lives under ``profiles:`` in ``config/router.yaml`` so the same file is reused
by T06's router for weights / fallbacks.

Tolerance: an unknown model returns ``0.0`` rather than raising, so newly-added
models in flight do not break runs — the cost ledger will simply under-report
for that model until its rate card entry is added.

T06 upgrade: ``get_router_config`` now returns a typed :class:`RouterConfig`
(or ``None`` when the YAML file is missing — preserving T04's graceful-
degradation contract).  A *present-but-malformed* file still raises at load
time (we won't silently swallow real misconfiguration).

T09 addition: :func:`extract_cost_fields` is a single source of truth for
turning a raw Anthropic response dict into the four cost columns. Both T04's
:func:`firm.llm.anthropic_client._stamp_llm_cost` (span attributes) and
T09's :func:`firm.db.cost_ledger.write_cost_ledger_row` writer (DB row)
consume it, so the span attribute and the ledger row can never drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from firm.core.config import RouterConfig, load_router_config

# Default location of the rate card.  Kept as a module-level constant so test
# overrides can monkeypatch it; the helper itself takes the path as an
# argument for explicit-config callers.
DEFAULT_ROUTER_PATH = Path("config/router.yaml")


@lru_cache(maxsize=1)
def _load_cached(path_str: str) -> RouterConfig | None:
    """LRU-cached loader keyed on the path string.

    Cached because every LLM call hits the rate card; re-parsing YAML on the
    hot path would be wasteful.  Keyed on a ``str`` (not :class:`Path`) so
    ``lru_cache`` can hash it.  Missing file -> ``None`` (graceful
    degradation: every cost lookup then returns 0.0).  Malformed file still
    raises ``ValidationError`` — silent fallback only applies to *absence*.
    """
    path = Path(path_str)
    if not path.exists():
        return None
    return load_router_config(path)


def get_router_config(path: Path | None = None) -> RouterConfig | None:
    """Return the parsed router config, using the LRU-cached loader."""
    p = path if path is not None else DEFAULT_ROUTER_PATH
    return _load_cached(str(p))


def compute_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    router_cfg: RouterConfig | None,
) -> float:
    """Return the USD cost of one call given the rate card.

    Iterates ``router_cfg.profiles`` looking for a profile whose ``model_id``
    matches *model*.  Returns ``0.0`` when *router_cfg* is ``None`` (no rate
    card available) or when no profile matches the model (unknown-model
    tolerance).

    Pricing convention: rates in ``router.yaml`` are USD per 1,000,000
    tokens, so we divide token counts by ``1_000_000`` before scaling.
    """
    if router_cfg is None:
        return 0.0
    for profile in router_cfg.profiles.values():
        if profile.model_id != model:
            continue
        return (input_tokens / 1_000_000.0) * profile.input_per_mtok_usd + (
            output_tokens / 1_000_000.0
        ) * profile.output_per_mtok_usd
    return 0.0


@dataclass(frozen=True)
class CostFields:
    """The four cost columns derivable from one raw response dict.

    ``cached_tokens`` is mutually exclusive with ``input_tokens`` /
    ``output_tokens`` — see :func:`extract_cost_fields` for the convention.
    """

    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None
    cost_usd: float


def extract_cost_fields(
    raw: dict[str, object],
    *,
    model: str,
    router_cfg: RouterConfig | None,
) -> CostFields:
    """Turn a raw Anthropic response dict into the four cost columns.

    Single source of truth for the (cache hit | live call) -> cost-fields
    mapping. Cache hit is detected via the synthetic ``_cache_hit`` boolean
    that :class:`firm.llm.anthropic_client.CachedAnthropicClient` attaches.

    Convention:
      * Cache hit -> ``input_tokens=None``, ``output_tokens=None``,
        ``cached_tokens=<sum of cached usage>``, ``cost_usd=0.0``. The
        cached-tokens count uses the cached row's recorded usage so that
        downstream "how much work the cache saved" reporting is accurate.
      * Live call -> ``input_tokens=<usage.input_tokens>``,
        ``output_tokens=<usage.output_tokens>``, ``cached_tokens=None``,
        ``cost_usd=compute_cost_usd(...)``.

    Missing or non-dict ``usage`` is treated as 0/0; ``_cache_hit`` must be
    the boolean ``True`` to be treated as a cache hit (any other value is live).
    """
    usage_raw = raw.get("usage", {})
    usage = usage_raw if isinstance(usage_raw, dict) else {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)

    cache_hit = raw.get("_cache_hit") is True

    if cache_hit:
        return CostFields(
            input_tokens=None,
            output_tokens=None,
            cached_tokens=input_tokens + output_tokens,
            cost_usd=0.0,
        )

    cost = compute_cost_usd(model, input_tokens, output_tokens, router_cfg)
    return CostFields(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=None,
        cost_usd=cost,
    )


__all__ = [
    "CostFields",
    "DEFAULT_ROUTER_PATH",
    "compute_cost_usd",
    "extract_cost_fields",
    "get_router_config",
]
