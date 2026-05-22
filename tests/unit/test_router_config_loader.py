"""Tests for the typed router-config loader (Plan 3 T06).

These pin the shape of ``config/router.yaml`` as consumed by
:func:`firm.core.config.load_router_config`:

* All three profiles (``haiku`` / ``sonnet`` / ``opus``) are required
* ``weights`` keys must match the :class:`RouterFeatures` field names
* ``fallback_chain`` references must resolve to declared profiles
* The real repo file loads, and its weights drive
  :meth:`RouterFeatures.score` to sensible high/low profile selections.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from firm.core.config import (
    RouterConfig,
    RouterProfile,
    load_llm_config,
    load_router_config,
)
from firm.core.models import RouterFeatures


# ---------------------------------------------------------------------------
# 1. Loading the real repo file produces a typed RouterConfig
# ---------------------------------------------------------------------------


def test_load_router_yaml_returns_typed_config() -> None:
    cfg = load_router_config(Path("config/router.yaml"))
    assert isinstance(cfg, RouterConfig)
    # All three profiles present and typed.
    assert set(cfg.profiles.keys()) == {"haiku", "sonnet", "opus"}
    for name, profile in cfg.profiles.items():
        assert isinstance(profile, RouterProfile), name
        assert profile.model_id  # non-empty
        assert profile.max_tokens > 0
        assert 0.0 <= profile.temperature <= 2.0
        assert profile.input_per_mtok_usd >= 0.0
        assert profile.output_per_mtok_usd >= 0.0
    # Weights and fallback_chain are populated.
    assert set(cfg.weights.keys()) == {
        "risk_weight",
        "novelty",
        "complexity",
        "time_pressure",
    }
    assert cfg.fallback_chain == ["sonnet", "haiku"]


# ---------------------------------------------------------------------------
# 2. Profile model_ids reconcile with llm.yaml (with Opus exception)
# ---------------------------------------------------------------------------


def test_all_router_model_ids_match_llm_yaml_or_are_known_opus() -> None:
    """Every router profile's ``model_id`` must either appear in
    ``config/llm.yaml`` (the agent rate card) or be the known Opus identifier.

    The Plan 3 spec asks "every profile resolves to a real model_id from
    ``config/llm.yaml``", but the Opus model isn't currently wired into any
    agent slot in ``llm.yaml``; this test adapts to that practical reality
    while still pinning a known-good Opus name.
    """
    router_cfg = load_router_config(Path("config/router.yaml"))
    llm_cfg = load_llm_config(Path("config/llm.yaml"))

    llm_models = {llm_cfg.research.model, llm_cfg.judge.model, llm_cfg.pm.model}
    # Document the carve-out explicitly so a future llm.yaml change that
    # introduces opus doesn't accidentally make this test pass via a typo.
    known_opus = {"claude-opus-4-7"}

    for name, profile in router_cfg.profiles.items():
        assert profile.model_id in (llm_models | known_opus), (
            f"router profile {name!r} model_id={profile.model_id!r} "
            f"not in llm.yaml models {sorted(llm_models)} "
            f"and not a known opus id"
        )


# ---------------------------------------------------------------------------
# 3-7. Validation errors on malformed yaml
# ---------------------------------------------------------------------------


_GOOD_YAML = """
profiles:
  haiku:
    model_id: claude-haiku-4-5
    max_tokens: 1024
    temperature: 0.0
    input_per_mtok_usd: 0.80
    output_per_mtok_usd: 4.00
  sonnet:
    model_id: claude-sonnet-4-6
    max_tokens: 4096
    temperature: 0.0
    input_per_mtok_usd: 3.00
    output_per_mtok_usd: 15.00
  opus:
    model_id: claude-opus-4-7
    max_tokens: 4096
    temperature: 0.0
    input_per_mtok_usd: 15.00
    output_per_mtok_usd: 75.00
weights:
  risk_weight: 0.40
  novelty: 0.20
  complexity: 0.20
  time_pressure: 0.20
fallback_chain: [sonnet, haiku]
"""


def _write_yaml(tmp_path: Path, raw: dict) -> Path:
    p = tmp_path / "router.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return p


def _good_dict() -> dict:
    return yaml.safe_load(_GOOD_YAML)


def test_missing_profile_raises(tmp_path: Path) -> None:
    raw = _good_dict()
    del raw["profiles"]["opus"]
    path = _write_yaml(tmp_path, raw)
    with pytest.raises(ValidationError, match="missing"):
        load_router_config(path)


def test_extra_profile_raises(tmp_path: Path) -> None:
    raw = _good_dict()
    raw["profiles"]["extra_profile"] = {
        "model_id": "claude-fake-9-9",
        "max_tokens": 1024,
        "temperature": 0.0,
        "input_per_mtok_usd": 1.0,
        "output_per_mtok_usd": 1.0,
    }
    path = _write_yaml(tmp_path, raw)
    # Because ProfileName is a Literal["haiku","sonnet","opus"], pydantic
    # rejects "extra_profile" at the type layer with its own ValidationError
    # before our model_validator runs; either failure mode is acceptable so
    # long as load_router_config refuses to return a config.
    with pytest.raises(ValidationError):
        load_router_config(path)


def test_weights_mismatch_raises(tmp_path: Path) -> None:
    raw = _good_dict()
    raw["weights"] = {"risk_weight": 1.0}  # missing three
    path = _write_yaml(tmp_path, raw)
    with pytest.raises(ValidationError, match="weights mismatch"):
        load_router_config(path)


def test_fallback_chain_unknown_profile_raises(tmp_path: Path) -> None:
    raw = _good_dict()
    raw["fallback_chain"] = ["sonnet", "ghost"]
    path = _write_yaml(tmp_path, raw)
    # Literal-typed list rejects "ghost" at the field layer; either pydantic's
    # built-in ValidationError or our follow-up "unknown profiles" message is
    # acceptable.
    with pytest.raises(ValidationError):
        load_router_config(path)


def test_fallback_chain_empty_raises(tmp_path: Path) -> None:
    raw = _good_dict()
    raw["fallback_chain"] = []
    path = _write_yaml(tmp_path, raw)
    with pytest.raises(ValidationError, match="non-empty"):
        load_router_config(path)


# ---------------------------------------------------------------------------
# 8. Real router.yaml weights drive RouterFeatures.score sensibly
# ---------------------------------------------------------------------------


def test_router_weights_drive_score() -> None:
    """End-to-end check that the shipped weights map extreme feature inputs
    to the expected profiles via :meth:`RouterFeatures.score`."""
    cfg = load_router_config(Path("config/router.yaml"))

    # All features maxed -> normalized score = 1.0 -> opus.
    high = RouterFeatures(
        risk_weight=1.0, novelty=1.0, complexity=1.0, time_pressure=1.0
    )
    assert high.score(cfg.weights) == "opus"

    # All features at floor -> score = 0.0 -> haiku.
    low = RouterFeatures(
        risk_weight=0.0, novelty=0.0, complexity=0.0, time_pressure=0.0
    )
    assert low.score(cfg.weights) == "haiku"
