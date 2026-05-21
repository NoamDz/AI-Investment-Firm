"""Plan 3 T25 — Full FailureMode CI invariant (partial).

Maps each enum value to a "triggering fixture": a test in this repo that
exercises the failure mode end-to-end. The meta-test below enumerates the
FailureMode enum and asserts each value (except the explicitly deferred
UNCITED_CLAIM end-to-end fixture) appears as a key in the registry. The
referenced test is then collected by pytest in the normal way; this meta-
test does not re-run them.

UNCITED_CLAIM is enum-only by design (Plan 2 §T25 deferral). Its
end-to-end fixture ships in Plan 4 with the red-team corpus.

The following Plan 2-era modes have no triggering site in the source yet
(enum-only) and are documented as allowed gaps pending future plan tasks:
  PROMPT_INJECTION_DETECTED, HITL_TIMEOUT, UNGROUNDED_CLAIM,
  TOOL_PERMISSION_DENIED, UNAPPROVED_HIGH_RISK, BROKER_UNAVAILABLE.
Each is registered in ALLOWED_GAPS below with a note; adding a triggering
site will require moving it into FAILURE_MODE_FIXTURES.
"""
from __future__ import annotations

from pathlib import Path

from firm.core.models import FailureMode

# ---------------------------------------------------------------------------
# Registry: FailureMode → "tests/<path>::<test_function_name>"
#
# Each locator must point to a real function that asserts the failure mode.
# Locators starting with "<" are allowed-gap sentinels (see ALLOWED_GAPS).
# ---------------------------------------------------------------------------

FAILURE_MODE_FIXTURES: dict[FailureMode, str] = {
    # LLM_UNAVAILABLE: router exhaustion → REFUSE in the wired e2e test.
    FailureMode.LLM_UNAVAILABLE: (
        "tests/integration/test_router_wired_e2e.py"
        "::test_research_refuses_with_llm_unavailable_when_ladder_exhausted"
    ),
    # INSUFFICIENT_EVIDENCE: empty retrieval → REFUSE in research-agent unit test.
    FailureMode.INSUFFICIENT_EVIDENCE: (
        "tests/unit/test_research_agent.py"
        "::test_research_refuses_when_retriever_returns_empty"
    ),
    # RISK_LIMIT_BREACHED: gross-exposure breach → REFUSE in risk-limits unit test.
    FailureMode.RISK_LIMIT_BREACHED: (
        "tests/unit/test_risk_limits.py"
        "::test_blocks_max_gross_exposure"
    ),
    # STALE_DATA: stale quote → REFUSE in risk-limits unit test.
    FailureMode.STALE_DATA: (
        "tests/unit/test_risk_limits.py"
        "::test_blocks_stale_quote"
    ),
    # SCHEMA_VALIDATION_FAILED: malformed PM-voter JSON → REFUSE in pm_agent test.
    FailureMode.SCHEMA_VALIDATION_FAILED: (
        "tests/unit/test_pm_agent.py"
        "::test_pm_maps_pm_vote_schema_error_to_refuse_schema_validation_failed"
    ),
    # RECONCILIATION_DRIFT: boot mismatch → failure_mode stamped (T25 new).
    FailureMode.RECONCILIATION_DRIFT: (
        "tests/integration/test_reconciliation_drift_failure_mode.py"
        "::test_boot_reconcile_mismatch_emits_reconciliation_drift"
    ),
    # SIGNED_APPROVAL_INVALID: tampered internal HMAC → audit_log entry (T25 new).
    FailureMode.SIGNED_APPROVAL_INVALID: (
        "tests/integration/test_hitl_invalid_signature_failure_mode.py"
        "::test_invalid_internal_signature_audit_logs_signed_approval_invalid"
    ),
    # UNKNOWN: catch-all; no specific triggering fixture required.
    FailureMode.UNKNOWN: "<allowed gap — UNKNOWN is a catch-all, no triggering fixture required>",
}

# ---------------------------------------------------------------------------
# Allowed gaps: enum-only values with no triggering site yet.
# Moving a value here documents the gap explicitly so CI catches regressions
# (i.e., if someone removes the value from the enum, the gap entry breaks).
# ---------------------------------------------------------------------------

ALLOWED_GAPS: dict[FailureMode, str] = {
    # Plan 4 target: red-team corpus supplies end-to-end UNCITED_CLAIM fixture.
    FailureMode.UNCITED_CLAIM: "deferred to Plan 4 — tied to red-team corpus",
    # Plan 2-era modes without triggering sites yet (enum-only).
    FailureMode.PROMPT_INJECTION_DETECTED: (
        "enum-only — no triggering site in codebase yet; add fixture in future plan"
    ),
    FailureMode.HITL_TIMEOUT: (
        "enum-only — no triggering site in codebase yet; add fixture in future plan"
    ),
    FailureMode.UNGROUNDED_CLAIM: (
        "enum-only — no triggering site in codebase yet; add fixture in future plan"
    ),
    FailureMode.TOOL_PERMISSION_DENIED: (
        "enum-only — no triggering site in codebase yet; add fixture in future plan"
    ),
    FailureMode.UNAPPROVED_HIGH_RISK: (
        "enum-only — no triggering site in codebase yet; add fixture in future plan"
    ),
    FailureMode.BROKER_UNAVAILABLE: (
        "enum-only — no triggering site in codebase yet; add fixture in future plan"
    ),
}


# ---------------------------------------------------------------------------
# Meta-tests
# ---------------------------------------------------------------------------


def test_every_failure_mode_has_a_fixture_entry_or_documented_gap() -> None:
    """Every FailureMode value must appear in FAILURE_MODE_FIXTURES or ALLOWED_GAPS."""
    enum_values = set(FailureMode)
    registered = set(FAILURE_MODE_FIXTURES) | set(ALLOWED_GAPS)
    missing = enum_values - registered
    assert not missing, (
        f"FailureMode(s) {sorted(m.value for m in missing)} are not covered by "
        f"any triggering fixture and are not documented in ALLOWED_GAPS. Either "
        f"add a fixture and register it in FAILURE_MODE_FIXTURES, or document "
        f"the deferral in ALLOWED_GAPS."
    )


def test_each_fixture_path_resolves_to_an_existing_test() -> None:
    """Each non-gap entry in FAILURE_MODE_FIXTURES must point to a real test function."""
    repo_root = Path(__file__).parent.parent.parent
    for mode, locator in FAILURE_MODE_FIXTURES.items():
        if locator.startswith("<") and locator.endswith(">"):
            # Sentinel — documented allowed gap (e.g., UNKNOWN catch-all).
            continue
        path_str, _, func_name = locator.partition("::")
        path = repo_root / path_str
        assert path.exists(), (
            f"{mode.value}: fixture file '{path_str}' not found. "
            f"Either create the file or move the mode to ALLOWED_GAPS."
        )
        src = path.read_text(encoding="utf-8")
        assert f"def {func_name}" in src, (
            f"{mode.value}: function '{func_name}' not found in '{path_str}'. "
            f"Check for renames or move the mode to ALLOWED_GAPS."
        )


def test_allowed_gaps_are_all_real_enum_values() -> None:
    """Every key in ALLOWED_GAPS must be a valid FailureMode (catches stale entries)."""
    all_modes = set(FailureMode)
    for mode in ALLOWED_GAPS:
        assert mode in all_modes, (
            f"ALLOWED_GAPS contains {mode!r} which is not in the FailureMode enum. "
            f"Remove the stale gap entry."
        )


def test_no_overlap_between_fixtures_and_gaps() -> None:
    """A FailureMode cannot appear in both FAILURE_MODE_FIXTURES and ALLOWED_GAPS."""
    overlap = set(FAILURE_MODE_FIXTURES) & set(ALLOWED_GAPS)
    assert not overlap, (
        f"FailureMode(s) {sorted(m.value for m in overlap)} appear in both "
        f"FAILURE_MODE_FIXTURES and ALLOWED_GAPS. Remove from one."
    )
