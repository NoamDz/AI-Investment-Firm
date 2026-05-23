"""Red-team class: citation_forgery.

Each corpus case for this class is piped through ``red_team_run()`` and the
invariant named by ``case.invariant_id`` is asserted.  The class-to-invariant
mapping lives in the corpus, not in this file.

Plan 4 Section B T07.h
----------------------
The historical guard ``if result.decision is not None and
result.decision.citations`` made the UNCITED_CLAIM assertion vacuous because
the 5 original cases all targeted ``research``/``reporter`` runners that
return ``None``.  Bundle C adds ``citation_forgery_6`` -- a PM-targeted
case carrying the ``FORGE_CITATION:<id>`` marker -- so the inline
verifier in ``_run_pm`` stamps ``failure_mode=UNCITED_CLAIM`` on a real
Decision.  The assertion below now actually fires (and would fail) when
that wiring breaks.
"""
from __future__ import annotations

import pytest

from tests.red_team._corpus import RedTeamCase, load_corpus
from tests.red_team.conftest import (
    RedTeamCtx,
    assert_no_privileged_action,
    assert_no_schema_bypass,
    assert_no_unapproved_trade,
    assert_no_forged_citation,
    assert_no_forged_approval,
    red_team_run,
)

_CLASS_SLUG = "citation_forgery"

_INVARIANT_DISPATCH = {
    "INV_NO_PRIVILEGED_ACTION": assert_no_privileged_action,
    "INV_NO_SCHEMA_BYPASS": assert_no_schema_bypass,
    "INV_NO_UNAPPROVED_TRADE": assert_no_unapproved_trade,
    "INV_NO_FORGED_CITATION": assert_no_forged_citation,
    "INV_NO_FORGED_APPROVAL": assert_no_forged_approval,
}


def _cases_for_class() -> list[RedTeamCase]:
    from pathlib import Path
    corpus_path = Path(__file__).parent / "corpus.jsonl"
    return [c for c in load_corpus(corpus_path) if c.injection_class == _CLASS_SLUG]


@pytest.mark.parametrize("case", _cases_for_class(), ids=lambda c: c.case_id)
def test_citation_forgery_invariant(case: RedTeamCase, red_team_ctx: RedTeamCtx) -> None:
    result = red_team_run(case, ctx=red_team_ctx)
    assertion = _INVARIANT_DISPATCH[case.invariant_id]
    assertion(result, case=case)

    # T07.h: assert UNCITED_CLAIM for the PM-targeted forgery case
    # (citation_forgery_6).  Research/reporter cases produce no Decision
    # at the harness level; only PM cases produce a Decision whose inline
    # CitationVerifier in _run_pm stamps the failure mode.  This is the
    # only path in the repo today that stamps UNCITED_CLAIM end-to-end
    # on a real Decision.
    if case.target_agent == "pm":
        from firm.core.models import FailureMode
        assert result.decision is not None, (
            f"{case.case_id}: expected a Decision from PM target"
        )
        assert result.decision.failure_mode == FailureMode.UNCITED_CLAIM, (
            f"{case.case_id}: expected failure_mode=UNCITED_CLAIM "
            f"for a PM-targeted citation_forgery case, got "
            f"{result.decision.failure_mode}"
        )
