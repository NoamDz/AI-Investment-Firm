"""Per-call USD cost computation from the ``config/router.yaml`` rate card.

Plan 3 T04: the :class:`firm.llm.anthropic_client.CachedAnthropicClient` stamps
``cost_usd`` onto the active OTel span on every live call.  The rate card
lives under ``profiles:`` in ``config/router.yaml`` so the same file is reused
by T06's router for weights / fallbacks (with this module only consuming the
``profiles`` subset).

Tolerance: an unknown model returns ``0.0`` rather than raising, so newly-added
models in flight do not break runs — the cost ledger will simply under-report
for that model until its rate card entry is added.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from firm.core.config import load_router_config

# Default location of the rate card.  Kept as a module-level constant so test
# overrides can monkeypatch it; the helper itself takes the path as an
# argument for explicit-config callers.
DEFAULT_ROUTER_PATH = Path("config/router.yaml")


@lru_cache(maxsize=1)
def _load_cached(path_str: str) -> dict:
    """LRU-cached loader keyed on the path string.

    Cached because every LLM call hits the rate card; re-parsing YAML on the
    hot path would be wasteful.  Keyed on a ``str`` (not :class:`Path`) so
    ``lru_cache`` can hash it.  Missing file -> empty dict (graceful
    degradation: every cost lookup then returns 0.0).
    """
    path = Path(path_str)
    if not path.exists():
        return {"profiles": {}}
    return load_router_config(path)


def get_router_config(path: Path | None = None) -> dict:
    """Return the parsed router config, using the LRU-cached loader."""
    p = path if path is not None else DEFAULT_ROUTER_PATH
    return _load_cached(str(p))


def compute_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    router_cfg: dict,
) -> float:
    """Return the USD cost of one call given the rate card.

    Iterates ``router_cfg["profiles"]`` looking for a profile whose
    ``model_id`` matches *model*.  Returns ``0.0`` when no match is found
    (unknown-model tolerance).

    Pricing convention: rates in ``router.yaml`` are USD per 1,000,000
    tokens, so we divide token counts by ``1_000_000`` before scaling.
    """
    profiles = router_cfg.get("profiles", {})
    if not isinstance(profiles, dict):
        return 0.0
    for profile in profiles.values():
        if not isinstance(profile, dict):
            continue
        if profile.get("model_id") != model:
            continue
        try:
            input_rate = float(profile.get("input_per_mtok_usd", 0.0) or 0.0)
            output_rate = float(profile.get("output_per_mtok_usd", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return (input_tokens / 1_000_000.0) * input_rate + (
            output_tokens / 1_000_000.0
        ) * output_rate
    return 0.0


__all__ = [
    "DEFAULT_ROUTER_PATH",
    "compute_cost_usd",
    "get_router_config",
]
