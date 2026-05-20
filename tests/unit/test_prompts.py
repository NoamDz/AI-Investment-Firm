"""Unit tests for prompt templates (Plan 2 §T17).

These tests assert each template contains its load-bearing phrases. They are
deliberately minimal because the bulk of behaviour is exercised end-to-end by
T18 (research extractor), T20 (sufficiency judge), and T25 (PM voter).
"""
from __future__ import annotations

import pytest

from firm.llm.prompts import (
    PM_VOTER_SYSTEM,
    RESEARCH_SYSTEM,
    SUFFICIENCY_SYSTEM,
    pm_voter_system,
)


def test_research_prompt_includes_arithmetic_ban() -> None:
    assert "must not perform arithmetic" in RESEARCH_SYSTEM


def test_research_prompt_includes_retrieved_content_safeguard() -> None:
    assert "<retrieved_content>" in RESEARCH_SYSTEM
    # The prompt must explicitly forbid following instructions inside the tags.
    lowered = RESEARCH_SYSTEM.lower()
    assert "do not follow" in lowered or "do not obey" in lowered
    assert "instruction" in lowered


def test_research_prompt_includes_tool_names() -> None:
    assert "fundamentals.get_ratio" in RESEARCH_SYSTEM
    assert "risk.get_metric" in RESEARCH_SYSTEM


def test_sufficiency_prompt_lists_three_status_values() -> None:
    assert "SUPPORTED|PARTIAL|UNSUPPORTED" in SUFFICIENCY_SYSTEM


def test_sufficiency_prompt_includes_retrieved_content_safeguard() -> None:
    assert "<retrieved_content>" in SUFFICIENCY_SYSTEM


def test_pm_voter_includes_all_three_lenses() -> None:
    # Raw template must mention all three lens names somewhere.
    assert "quality lens" in PM_VOTER_SYSTEM
    assert "valuation lens" in PM_VOTER_SYSTEM
    assert "catalyst lens" in PM_VOTER_SYSTEM
    # Each rendered prompt must contain its lens name.
    for lens in ("quality", "valuation", "catalyst"):
        rendered = pm_voter_system(lens)
        assert lens in rendered.lower()


def test_pm_voter_output_shape_specified() -> None:
    assert "BUY" in PM_VOTER_SYSTEM
    assert "HOLD" in PM_VOTER_SYSTEM
    assert "SELL" in PM_VOTER_SYSTEM
    assert "confidence" in PM_VOTER_SYSTEM
    assert "cited_claim_ids" in PM_VOTER_SYSTEM


def test_pm_voter_rejects_unknown_lens() -> None:
    with pytest.raises(ValueError):
        pm_voter_system("grimoire")
