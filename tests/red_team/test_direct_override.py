"""Red-team class: direct_override.

Each of the 5 corpus cases for this class is piped through
red_team_run() and the invariant named by case.invariant_id is
asserted. The class-to-invariant mapping is determined by the
corpus, not hardcoded here.
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

_CLASS_SLUG = "direct_override"

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
def test_direct_override_invariant(case: RedTeamCase, red_team_ctx: RedTeamCtx) -> None:
    result = red_team_run(case, ctx=red_team_ctx)
    assertion = _INVARIANT_DISPATCH[case.invariant_id]
    assertion(result, case=case)
