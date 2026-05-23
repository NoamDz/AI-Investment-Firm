"""Haiku-backed sufficiency judge for cited claims (Plan 2 §T20).

Given a question and a list of :class:`firm.core.models.Claim` objects
produced by the T18 cited-claim extractor, this judge asks a Haiku model
(via the T17 :data:`firm.llm.prompts.SUFFICIENCY_SYSTEM` prompt) to label
each claim ``SUPPORTED|PARTIAL|UNSUPPORTED`` and returns a
:class:`firm.grounding.schema.SufficiencyResult`.

The prompt instructs the model to emit JSON with keys
``assessments``/``status``/``rationale``, whereas the schema validates
``claim_assessments``/``support``/``reasoning``. The key translation is
localised to :meth:`SufficiencyJudge.assess` so neither the prompt
contract (T17) nor the schema contract (T3) needs to change.

Parse failures — un-parseable JSON, missing required keys, or a
:class:`pydantic.ValidationError` during schema validation — are
re-raised as :class:`JudgeResponseError` so the T21 caller can map them
to :attr:`firm.core.models.FailureMode.LLM_UNAVAILABLE`.
"""
from __future__ import annotations

import json

from pydantic import ValidationError

from firm.core.models import Claim
from firm.grounding.schema import SufficiencyResult
from firm.llm.citations import AnthropicMessagesClient
from firm.llm.prompts import SUFFICIENCY_SYSTEM


class JudgeResponseError(Exception):
    """Raised when the sufficiency judge returns un-parseable JSON or a schema violation."""


class JudgeSchemaError(JudgeResponseError):
    """Raised when the sufficiency judge's response fails schema validation.

    Distinct from generic JudgeResponseError so callers can map this to
    FailureMode.SCHEMA_VALIDATION_FAILED instead of LLM_UNAVAILABLE.
    """


def _strip_markdown_fences(text: str) -> str:
    """Strip a single leading ```...``` markdown fence if present.

    Conservative: only removes fences when the text *starts* with three
    backticks (optionally followed by a language tag and newline) and
    *ends* with three backticks. Leaves all other text untouched so a
    JSON payload that happens to embed a literal ``` inside a string
    value is not corrupted.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    if not stripped.endswith("```"):
        return stripped
    # Drop the opening fence (and optional language tag on the same line).
    body = stripped[3:]
    newline_at = body.find("\n")
    if newline_at != -1 and body[:newline_at].strip().isalpha():
        body = body[newline_at + 1 :]
    # Drop the closing fence.
    if body.endswith("```"):
        body = body[:-3]
    return body.strip()


class SufficiencyJudge:
    """Asks a Haiku model to assess each cited claim's sufficiency.

    The judge is stateless across calls: every :meth:`assess` invocation
    builds its own user message and issues a single ``messages_create``
    call (or short-circuits with an empty result when no claims were
    supplied).
    """

    def __init__(
        self,
        *,
        client: AnthropicMessagesClient,
        model: str,
        max_tokens: int = 2048,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def assess(self, *, question: str, claims: list[Claim]) -> SufficiencyResult:
        """Return a :class:`SufficiencyResult` labelling each claim.

        Claims are referenced by synthetic 1-indexed IDs (``c1``, ``c2``,
        ...) embedded in the prompt body. The model's reply is expected
        to echo those IDs in its ``assessments`` array. An empty
        ``claims`` list short-circuits and returns an empty result
        without contacting the LLM.
        """
        if not claims:
            return SufficiencyResult(
                claim_assessments=[],
                overall_reasoning="no claims to assess",
            )

        # Build one block per claim, with provenance-aware framing:
        #   * text-cited claims (have source_quote) -> wrap the verbatim
        #     cited_text in <retrieved_content> tags, as SUFFICIENCY_SYSTEM
        #     promises;
        #   * tool-derived claims (have tool_call_id, no source_quote) -> render
        #     the tool result as <tool_result> instead, so the judge does not
        #     mark deterministically-computed values UNSUPPORTED just because
        #     there is no document chunk to quote. The SUFFICIENCY_SYSTEM
        #     prompt instructs the judge to treat tool_result blocks as
        #     deterministic provenance.
        content_blocks: list[str] = []
        for i, claim in enumerate(claims):
            cid = f"c{i + 1}"
            if claim.source_quote:
                evidence_block = (
                    f"  <retrieved_content>\n"
                    f"  {claim.source_quote}\n"
                    f"  </retrieved_content>"
                )
            elif claim.tool_call_id:
                value_part = (
                    f"value={claim.value} unit={claim.unit or 'n/a'}"
                    if claim.value is not None
                    else "value=n/a"
                )
                evidence_block = (
                    f"  <tool_result tool_call_id=\"{claim.tool_call_id}\">\n"
                    f"  {value_part}\n"
                    f"  </tool_result>"
                )
            else:
                evidence_block = (
                    f"  <retrieved_content>\n"
                    f"  (no verbatim source text recorded)\n"
                    f"  </retrieved_content>"
                )
            content_blocks.append(f"- {cid}: {claim.text}\n{evidence_block}")
        user_content = (
            f"Question: {question}\n\n"
            "Cited claims to assess:\n" + "\n".join(content_blocks)
        )
        messages: list[dict[str, object]] = [
            {"role": "user", "content": user_content},
        ]

        response = self._client.messages_create(
            model=self._model,
            system=SUFFICIENCY_SYSTEM,
            messages=messages,
            tools=None,
            max_tokens=self._max_tokens,
            temperature=0.0,
        )

        # Concatenate every ``text``-type content block into one string.
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

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise JudgeResponseError(
                f"sufficiency judge returned non-JSON text: {raw_text[:120]!r}"
            ) from exc

        if not isinstance(parsed, dict):
            raise JudgeResponseError(
                f"sufficiency judge JSON must be an object, got {type(parsed).__name__}"
            )

        try:
            assessments_raw = parsed["assessments"]
            if not isinstance(assessments_raw, list):
                raise JudgeResponseError(
                    "sufficiency judge 'assessments' must be a list, "
                    f"got {type(assessments_raw).__name__}"
                )
            translated_assessments: list[dict[str, object]] = []
            for entry in assessments_raw:
                if not isinstance(entry, dict):
                    raise JudgeResponseError(
                        f"each assessment must be a JSON object, got {type(entry).__name__}"
                    )
                translated_assessments.append(
                    {
                        "claim_id": entry["claim_id"],
                        "support": entry["status"],
                        "reasoning": entry["rationale"],
                    }
                )
        except KeyError as exc:
            raise JudgeResponseError(
                f"sufficiency judge JSON missing required key: {exc!s}"
            ) from exc

        overall_reasoning_raw = parsed.get("overall_reasoning", "")
        if not isinstance(overall_reasoning_raw, str):
            overall_reasoning_raw = ""

        translated: dict[str, object] = {
            "overall_reasoning": overall_reasoning_raw,
            "claim_assessments": translated_assessments,
        }

        try:
            return SufficiencyResult.model_validate(translated)
        except ValidationError as exc:
            raise JudgeSchemaError(
                f"sufficiency judge JSON failed schema validation: {exc!s}"
            ) from exc


__all__ = ["JudgeResponseError", "JudgeSchemaError", "SufficiencyJudge"]
