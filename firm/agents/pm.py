"""PM agent — deterministic pass-through stub for Plan 1.

Plan 2 swaps this for vote-of-3 self-consistency over LLM rationales.

T25 adds:  PmLens, PmVote, PmVoteSchemaError, PmVoter.
T27 will rewrite make_pm() to use PmVoter.
"""
from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError, model_validator

from firm.core.ids import ulid_new
from firm.core.models import ActionEnum, Claim, Decision
from firm.llm.citations import AnthropicMessagesClient
from firm.llm.prompts import pm_voter_system
from firm.orchestrator.state import WorkingState


# ---------------------------------------------------------------------------
# Plan 1 stub — preserved for T27
# ---------------------------------------------------------------------------


def make_pm() -> Callable[[WorkingState], dict[str, Any]]:
    def pm(state: WorkingState) -> dict[str, Any]:
        research: Decision = state["research_decision"]
        decision = Decision(
            id=ulid_new(), decision_id_chain=[research.id],
            action=research.action, payload=research.payload,
            rationale=f"pm pass-through: {research.rationale}",
            confidence=research.confidence, citations=research.citations,
            falsification_condition=research.falsification_condition,
            escalation_reason=None, failure_mode=None,
            metadata={"agent": "pm", "stub": True}, nonce="pm-stub",
        )
        return {"pm_decision": decision}
    return pm


# ---------------------------------------------------------------------------
# T25 — PM voter (single-lens)
# ---------------------------------------------------------------------------


class PmLens(StrEnum):
    QUALITY = "quality"
    VALUATION = "valuation"
    CATALYST = "catalyst"


class PmVote(BaseModel):
    lens: PmLens
    vote: ActionEnum  # constrained to BUY|HOLD|SELL via validator below
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    cited_claim_ids: list[str]

    @model_validator(mode="after")
    def _vote_is_buy_hold_or_sell(self) -> "PmVote":
        if self.vote not in (ActionEnum.BUY, ActionEnum.HOLD, ActionEnum.SELL):
            raise ValueError(
                f"PmVote.vote must be BUY, HOLD, or SELL; got {self.vote!r}"
            )
        return self


class PmVoteSchemaError(Exception):
    """Raised when the Sonnet response cannot be parsed into a PmVote."""


def _strip_markdown_fences(text: str) -> str:
    """Strip a single leading ```...``` markdown fence if present.

    Conservative: only removes fences when the text starts with three
    backticks and ends with three backticks.  Leaves everything else
    untouched so JSON values that embed ``` are not corrupted.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    if not stripped.endswith("```"):
        return stripped
    body = stripped[3:]
    newline_at = body.find("\n")
    if newline_at != -1 and body[:newline_at].strip().isalpha():
        body = body[newline_at + 1:]
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


class PmVoter:
    """Single-lens PM voter backed by Sonnet.

    Each ``vote()`` call:
    1. Renders a lens-specific system prompt via ``pm_voter_system``.
    2. Builds a user message with the research question, rationale, and
       cited claims wrapped in ``<retrieved_content>`` tags.
    3. Calls the Anthropic ``messages_create`` API.
    4. Parses and validates the JSON response into a :class:`PmVote`.
    5. Filters ``cited_claim_ids`` to the subset of provided ids.
    """

    def __init__(
        self,
        *,
        client: AnthropicMessagesClient,
        model: str,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def vote(
        self,
        *,
        lens: PmLens,
        question: str,
        claims: list[Claim],
        research_rationale: str,
    ) -> PmVote:
        """Cast a single-lens vote on the trade idea.

        Parameters
        ----------
        lens:
            Which analytical lens to apply.
        question:
            The original research question / trade idea.
        claims:
            Cited claims produced by the Research Extractor.  Each claim is
            assigned a positional id ``c1``, ``c2``, ... (1-indexed).
        research_rationale:
            The research agent's summary rationale for the proposed action.

        Returns
        -------
        PmVote
            Validated vote with ``cited_claim_ids`` filtered to the subset
            of provided ids.

        Raises
        ------
        PmVoteSchemaError
            If the LLM response cannot be parsed into a valid :class:`PmVote`.
        """
        system = pm_voter_system(lens.value)

        # Build the positional id mapping: c1, c2, ...
        claim_ids = [f"c{i + 1}" for i in range(len(claims))]
        valid_id_set: set[str] = set(claim_ids)

        claim_lines = "\n".join(
            f"[{cid}] {claim.text}"
            for cid, claim in zip(claim_ids, claims)
        )
        user_text = (
            f"Research question: {question}\n\n"
            f"Research rationale: {research_rationale}\n\n"
            "<retrieved_content>\n"
            f"{claim_lines}\n"
            "</retrieved_content>"
        )

        messages: list[dict[str, object]] = [
            {"role": "user", "content": user_text},
        ]

        response = self._client.messages_create(
            model=self._model,
            system=system,
            messages=messages,
            tools=None,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

        # Concatenate all text-type content blocks.
        text_parts: list[str] = []
        content = response.get("content", [])
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "text":
                    continue
                text_val = block.get("text", "")
                if isinstance(text_val, str):
                    text_parts.append(text_val)
        raw_text = _strip_markdown_fences("".join(text_parts))

        # Parse JSON.
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise PmVoteSchemaError(
                f"PM voter returned non-JSON text: {raw_text[:120]!r}"
            ) from exc

        if not isinstance(payload, dict):
            raise PmVoteSchemaError(
                f"PM voter JSON must be an object, got {type(payload).__name__}"
            )

        # Build and validate PmVote.
        try:
            vote_obj = PmVote(
                lens=lens,
                vote=ActionEnum(payload["vote"]),
                confidence=payload["confidence"],
                rationale=payload["rationale"],
                cited_claim_ids=payload["cited_claim_ids"],
            )
        except (ValidationError, KeyError, ValueError) as exc:
            raise PmVoteSchemaError(
                f"PM voter JSON failed schema validation: {exc!s}"
            ) from exc

        # Server-side filter: keep only ids that were actually provided.
        filtered_ids = [cid for cid in vote_obj.cited_claim_ids if cid in valid_id_set]
        # Re-instantiate via model_validate so all validators (e.g. the
        # BUY/HOLD/SELL invariant) re-run; model_copy(update=...) would
        # silently bypass them.
        try:
            vote_obj = PmVote.model_validate(
                {**vote_obj.model_dump(), "cited_claim_ids": filtered_ids}
            )
        except (ValidationError, KeyError, ValueError) as exc:
            raise PmVoteSchemaError(
                f"PM voter JSON failed schema validation: {exc!s}"
            ) from exc

        return vote_obj


__all__ = [
    "PmLens",
    "PmVote",
    "PmVoteSchemaError",
    "PmVoter",
    "make_pm",
]
