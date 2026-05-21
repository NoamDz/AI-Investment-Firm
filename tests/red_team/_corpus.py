"""Red-team corpus loader and schema validator.

Exports RedTeamCase (Pydantic model), load_corpus(), and canonical-set constants.
Passes strict mypy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator

INJECTION_CLASSES: tuple[str, ...] = (
    "direct_override",
    "role_hijack",
    "delimiter_break",
    "unicode_homoglyph",
    "encoded_payload",
    "indirect_tool_output",
    "multi_step_chain",
    "citation_forgery",
    "spoofed_approval",
    "confused_deputy",
)

TARGET_AGENTS: tuple[str, ...] = (
    "research",
    "pm",
    "risk",
    "execution",
    "hitl",
    "reporter",
    "monitor",
)

INVARIANT_IDS: tuple[str, ...] = (
    "INV_NO_PRIVILEGED_ACTION",
    "INV_NO_SCHEMA_BYPASS",
    "INV_NO_UNAPPROVED_TRADE",
    "INV_NO_FORGED_CITATION",
    "INV_NO_FORGED_APPROVAL",
)

_INJECTION_CLASSES_SET: frozenset[str] = frozenset(INJECTION_CLASSES)
_TARGET_AGENTS_SET: frozenset[str] = frozenset(TARGET_AGENTS)
_INVARIANT_IDS_SET: frozenset[str] = frozenset(INVARIANT_IDS)


class RedTeamCase(BaseModel):
    """A single red-team injection test case."""

    case_id: str
    injection_class: str
    payload_text: str
    target_agent: str
    invariant_id: str

    @field_validator("injection_class")
    @classmethod
    def validate_injection_class(cls, v: str) -> str:
        if v not in _INJECTION_CLASSES_SET:
            raise ValueError(
                f"injection_class {v!r} not in allowed set: {sorted(_INJECTION_CLASSES_SET)}"
            )
        return v

    @field_validator("target_agent")
    @classmethod
    def validate_target_agent(cls, v: str) -> str:
        if v not in _TARGET_AGENTS_SET:
            raise ValueError(
                f"target_agent {v!r} not in allowed set: {sorted(_TARGET_AGENTS_SET)}"
            )
        return v

    @field_validator("invariant_id")
    @classmethod
    def validate_invariant_id(cls, v: str) -> str:
        if v not in _INVARIANT_IDS_SET:
            raise ValueError(
                f"invariant_id {v!r} not in allowed set: {sorted(_INVARIANT_IDS_SET)}"
            )
        return v


def load_corpus(path: Path) -> list[RedTeamCase]:
    """Read corpus.jsonl, validate each line, return list of RedTeamCase.

    Raises ValueError with the 1-based line number on any malformed entry.
    """
    cases: list[RedTeamCase] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data: Any = json.loads(line)
                case = RedTeamCase.model_validate(data)
            except Exception as exc:
                raise ValueError(f"Malformed corpus entry at line {lineno}: {exc}") from exc
            cases.append(case)
    return cases
