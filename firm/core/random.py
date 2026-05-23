"""Seeded random-number facade for deterministic firm runs.

A single ``get_rng()`` returns a ``random.Random`` instance seeded from
the ``FIRM_RANDOM_SEED`` environment variable (default ``42``). All firm
code that needs randomness should import this facade instead of using
``random.*`` directly, so the eval determinism gate (``firm/ops/check_reports_clean.sh``)
remains valid.

The seed resolution is **process-level**: the same Python interpreter
that returns RNG_A from ``get_rng()`` returns a DIFFERENT instance from
the next ``get_rng()`` call (the seed is re-read; the returned Random is
a fresh object). Callers that need a stable RNG across multiple operations
should bind the result once and reuse:

    rng = get_rng()
    rng.shuffle(items)
    rng.choice(items)

Re-seeding via the env var requires a process restart (the env is read
on every call, but each call returns a fresh Random — there is no module
cache to invalidate).
"""
from __future__ import annotations

import os
import random

_DEFAULT_SEED = 42


def _resolve_seed() -> int:
    """Read FIRM_RANDOM_SEED from env, returning the default on missing/invalid input."""
    raw = os.environ.get("FIRM_RANDOM_SEED")
    if raw is None:
        return _DEFAULT_SEED
    try:
        return int(raw)
    except ValueError:
        return _DEFAULT_SEED


def get_rng() -> random.Random:
    """Return a fresh ``random.Random`` seeded from FIRM_RANDOM_SEED (default 42)."""
    return random.Random(_resolve_seed())
