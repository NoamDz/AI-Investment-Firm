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
"""
from __future__ import annotations

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


__all__ = [
    "DEFAULT_ROUTER_PATH",
    "compute_cost_usd",
    "get_router_config",
]
