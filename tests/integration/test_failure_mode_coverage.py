"""Plan 4 T26 — Full FailureMode CI invariant (strict).

Strict invariant: every :class:`FailureMode` value EXCEPT
:attr:`FailureMode.UNKNOWN` (a catch-all whose triggering site by
definition is "anywhere we don't recognise") has a triggering
fixture — a test in this repo that exercises the failure mode
end-to-end and stamps the enum onto a real :class:`Decision`.

``ALLOWED_GAPS`` is empty by design.  Adding a new enum value to
:class:`FailureMode` requires either:

  1. Adding a triggering fixture and registering it in
     :data:`FAILURE_MODE_FIXTURES`, OR
  2. In exceptional, documented cases, re-introducing an
     ``ALLOWED_GAPS`` entry with an explicit justification AND
     loosening the
     :func:`test_every_failure_mode_except_unknown_has_a_triggering_fixture`
     invariant.  Step 2 is intentionally painful so the default
     behaviour stays "every enum value has a fixture".

Final shape (Plan 4 T26): 14 fixtures + 1 catch-all UNKNOWN sentinel
= 15 enum values, the entirety of :class:`FailureMode`.

The meta-tests below enumerate the :class:`FailureMode` enum and
assert each value has a real path (or the explicit UNKNOWN
sentinel).  The referenced tests are collected by pytest in the
normal way; these meta-tests do not re-run them.
"""
from __future__ import annotations

from pathlib import Path

from firm.core.models import FailureMode

# ---------------------------------------------------------------------------
# Registry: FailureMode → "tests/<path>::<test_function_name>"
#
# Each locator must point to a real function that asserts the failure mode.
# Locators starting with "<" are allowed-gap sentinels — currently only the
# UNKNOWN catch-all uses this form.
# ---------------------------------------------------------------------------

FAILURE_MODE_FIXTURES: dict[FailureMode, str] = {
    # LLM_UNAVAILABLE: router exhaustion → REFUSE in the wired e2e test.
    FailureMode.LLM_UNAVAILABLE: (
        "tests/integration/test_router_wired_e2e.py"
        "::test_research_refuses_with_llm_unavailable_when_ladder_exhausted"
    ),
    # INSUFFICIENT_EVIDENCE: empty retrieval → REFUSE in Plan 4 integration
    # fixture (heartbeat-level e2e; promoted from the unit test in T18).
    FailureMode.INSUFFICIENT_EVIDENCE: (
        "tests/integration/test_failuremode_insufficient_evidence.py"
        "::test_heartbeat_emits_refuse_with_insufficient_evidence_on_empty_retrieval"
    ),
    # RISK_LIMIT_BREACHED: gross-exposure breach → REFUSE in risk-limits unit test.
    FailureMode.RISK_LIMIT_BREACHED: (
        "tests/unit/test_risk_limits.py"
        "::test_blocks_max_gross_exposure"
    ),
    # STALE_DATA: stale quote → REFUSE in Plan 4 integration fixture
    # (graph propagates REFUSE end-to-end; promoted from unit test in T21).
    FailureMode.STALE_DATA: (
        "tests/integration/test_failuremode_stale_data.py"
        "::test_graph_propagates_refuse_stale_data_and_writes_no_broker_call"
    ),
    # SCHEMA_VALIDATION_FAILED: malformed PM-voter JSON → REFUSE in Plan 4
    # integration fixture (graph propagates; promoted from unit test in T20).
    FailureMode.SCHEMA_VALIDATION_FAILED: (
        "tests/integration/test_failuremode_schema_validation.py"
        "::test_graph_propagates_refuse_schema_validation_failed_and_writes_no_broker_call"
    ),
    # RECONCILIATION_DRIFT: boot mismatch → failure_mode stamped (Plan 3 T25).
    FailureMode.RECONCILIATION_DRIFT: (
        "tests/integration/test_reconciliation_drift_failure_mode.py"
        "::test_boot_reconcile_mismatch_emits_reconciliation_drift"
    ),
    # SIGNED_APPROVAL_INVALID: tampered internal HMAC → audit_log entry (Plan 3 T25).
    FailureMode.SIGNED_APPROVAL_INVALID: (
        "tests/integration/test_hitl_invalid_signature_failure_mode.py"
        "::test_invalid_internal_signature_audit_logs_signed_approval_invalid"
    ),
    # PROMPT_INJECTION_DETECTED: delimiter-break payload → REFUSE in PM sanitiser test.
    FailureMode.PROMPT_INJECTION_DETECTED: (
        "tests/integration/test_failuremode_prompt_injection.py"
        "::test_pm_emits_refuse_with_prompt_injection_detected_on_tainted_claim"
    ),
    # UNCITED_CLAIM: red-team citation_forgery cases — extractor outputs a claim
    # without grounding citations (Plan 4 Section B T07.h).
    FailureMode.UNCITED_CLAIM: (
        "tests/red_team/test_citation_forgery.py"
        "::test_citation_forgery_invariant"
    ),
    # UNGROUNDED_CLAIM: extractor fabricates a chunk_id absent from retrieval (Plan 4 T22).
    FailureMode.UNGROUNDED_CLAIM: (
        "tests/integration/test_failuremode_ungrounded_claim.py"
        "::test_heartbeat_emits_refuse_with_ungrounded_claim_on_fabricated_chunk_id"
    ),
    # TOOL_PERMISSION_DENIED: research-role broker.place_order rejected by
    # the capability layer (Plan 4 T23).
    FailureMode.TOOL_PERMISSION_DENIED: (
        "tests/integration/test_failuremode_tool_permission_denied.py"
        "::test_research_role_broker_call_rejected_with_tool_permission_denied_failure_mode"
    ),
    # UNAPPROVED_HIGH_RISK: aged-out pending hitl_queue row keyed to a >3% NAV
    # ESCALATE emits REFUSE with conservative-default disposition (Plan 4 T24).
    FailureMode.UNAPPROVED_HIGH_RISK: (
        "tests/integration/test_failuremode_unapproved_high_risk.py"
        "::test_aged_pending_high_risk_hitl_row_emits_refuse_with_unapproved_high_risk_failure_mode"
    ),
    # BROKER_UNAVAILABLE: bounded broker-submit retries exhausted => REFUSE
    # with outbox row left 'pending' for next-heartbeat recovery (Plan 4 T25).
    FailureMode.BROKER_UNAVAILABLE: (
        "tests/integration/test_failuremode_broker_unavailable.py"
        "::test_broker_503_emits_refuse_with_broker_unavailable_and_leaves_outbox_pending"
    ),
    # HITL_TIMEOUT: Slack notifier raises during ``notifier.notify(...)``;
    # the message never arrived so no human can have acked it; test-scoped
    # reaper emits REFUSE on transport-grade grounds (Plan 4 T26).
    FailureMode.HITL_TIMEOUT: (
        "tests/integration/test_failuremode_hitl_timeout.py"
        "::test_failed_slack_notify_on_pending_hitl_row_emits_refuse_with_hitl_timeout"
    ),
    # UNKNOWN: catch-all; no specific triggering fixture required.
    FailureMode.UNKNOWN: "<allowed gap — UNKNOWN is a catch-all, no triggering fixture required>",
}

# ---------------------------------------------------------------------------
# Allowed gaps: enum-only values with no triggering site yet.
#
# T26 invariant: this dict is empty.  Every FailureMode value (except the
# UNKNOWN catch-all, which uses an in-registry sentinel) has a triggering
# fixture.  This dict is kept (rather than deleted) so the meta-tests stay
# self-documenting and so re-introducing a gap in an emergency is a
# small, visible edit rather than a structural change.
# ---------------------------------------------------------------------------

ALLOWED_GAPS: dict[FailureMode, str] = {}


# ---------------------------------------------------------------------------
# Meta-tests
# ---------------------------------------------------------------------------


def test_every_failure_mode_except_unknown_has_a_triggering_fixture() -> None:
    """Strict T26 invariant: every :class:`FailureMode` value except
    :attr:`FailureMode.UNKNOWN` must appear as a key in
    :data:`FAILURE_MODE_FIXTURES` with a real path (not a ``<...>``
    sentinel).  ``UNKNOWN`` is allowed to use the sentinel form because
    it is a catch-all whose triggering site is, by definition, "anywhere
    we don't recognise"."""
    missing: list[str] = []
    sentinel_non_unknown: list[str] = []
    for mode in FailureMode:
        if mode is FailureMode.UNKNOWN:
            # UNKNOWN may stay as the sentinel.
            continue
        locator = FAILURE_MODE_FIXTURES.get(mode)
        if locator is None:
            missing.append(mode.value)
            continue
        if locator.startswith("<") and locator.endswith(">"):
            sentinel_non_unknown.append(mode.value)
    assert not missing, (
        f"FailureMode(s) {sorted(missing)} are not covered by any "
        f"triggering fixture.  Add a fixture and register it in "
        f"FAILURE_MODE_FIXTURES; ALLOWED_GAPS is empty by design (T26 "
        f"strict invariant).  Re-introducing an ALLOWED_GAPS entry "
        f"requires loosening this test."
    )
    assert not sentinel_non_unknown, (
        f"FailureMode(s) {sorted(sentinel_non_unknown)} have a "
        f"placeholder ``<…>`` locator instead of a real fixture path.  "
        f"Only FailureMode.UNKNOWN may use the sentinel form."
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


def test_allowed_gaps_is_empty() -> None:
    """T26 strict-invariant lock: ``ALLOWED_GAPS`` must be empty.

    Re-introducing a gap requires editing both this test and the
    :func:`test_every_failure_mode_except_unknown_has_a_triggering_fixture`
    invariant — a deliberately visible structural change so the default
    "every enum value has a fixture" stays the default.
    """
    assert ALLOWED_GAPS == {}, (
        f"ALLOWED_GAPS must be empty (Plan 4 T26 strict invariant); "
        f"got {sorted(m.value for m in ALLOWED_GAPS)}"
    )


def test_no_overlap_between_fixtures_and_gaps() -> None:
    """A FailureMode cannot appear in both FAILURE_MODE_FIXTURES and ALLOWED_GAPS."""
    overlap = set(FAILURE_MODE_FIXTURES) & set(ALLOWED_GAPS)
    assert not overlap, (
        f"FailureMode(s) {sorted(m.value for m in overlap)} appear in both "
        f"FAILURE_MODE_FIXTURES and ALLOWED_GAPS. Remove from one."
    )
