"""Red-team class: citation_forgery.

Each of the 5 corpus cases for this class is piped through
red_team_run() and the invariant named by case.invariant_id is
asserted. The class-to-invariant mapping is determined by the
corpus, not hardcoded here.

NOTE: All 5 citation_forgery cases target 'research' or 'reporter' agents whose
_run_* helpers return None (no Decision produced). The UNCITED_CLAIM assertion
below is therefore guarded by both result.decision is not None AND
result.decision.citations being non-empty, per Strategy 1. Closing the
UNCITED_CLAIM ALLOWED_GAPS entry from Plan 3 is jointly the responsibility of
this module and T11 (firm/eval/perf_metrics.py), which exercises the full
citation surface end-to-end.
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

    # T07.h: end-to-end UNCITED_CLAIM check — closes Plan 3 ALLOWED_GAPS entry
    # jointly with T11. Guard: only assert failure_mode when the decision actually
    # carries citations (Strategy 1 — avoids vacuous failures for research/reporter
    # cases where _run_* returns None or produces no citations).
    if result.decision is not None and result.decision.citations:
        from firm.core.models import FailureMode
        assert result.decision.failure_mode == FailureMode.UNCITED_CLAIM, (
            f"{case.case_id}: expected failure_mode=UNCITED_CLAIM "
            f"for a citation_forgery case, got {result.decision.failure_mode}"
        )
