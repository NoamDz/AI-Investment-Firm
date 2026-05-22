"""Plan 4 §T40 — AgentCore Reporter adapter byte-equivalence test.

Asserts that the on-disk ``decisions.jsonl`` row produced via the
AgentCore-served Reporter (``firm.agentcore.reporter_adapter``) is
byte-for-byte identical to the row produced via the LangGraph-served
Reporter (``firm.agents.reporter.make_reporter``) when both are driven
with the same :class:`firm.orchestrator.state.WorkingState` input.

Skip semantics: the test cleanly skips when ``bedrock_agentcore_sdk`` is
not installed. The SDK ships behind the optional ``[agentcore]`` extra
(T41), so a clean dev install (``pip install -e .``) does NOT pull it in
— hence the skip is the expected outcome on default install. When the
extra IS installed, the full byte-equivalence assertion runs.

Byte-equivalence preconditions:
  1. Both paths use the same clock (we monkeypatch the LangGraph
     closure's clock to match the adapter's, which the adapter builds at
     import time using :class:`firm.core.clock.WallClock`). We instead
     force both paths through a single :class:`ReplayClock` by injecting
     it into the adapter's module-level ``_reporter`` closure post-import.
  2. Both paths must emit the same ``trace_id`` field. The Reporter
     stamps ``format(get_current_span().get_span_context().trace_id, "032x")``
     onto every JSONL row. We monkeypatch ``get_current_span`` to return
     ``INVALID_SPAN`` (trace_id == 0) so both writes record ``trace_id=""``.
  3. Same input state, same ``db_path=None`` (so the conditional DB write
     is skipped on both paths).
"""
from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

# Skip the entire module when the optional AgentCore SDK is missing.
# This is the expected outcome on a default `pip install -e .` install;
# only when the operator has run `pip install -e .[agentcore]` (T41)
# will the SDK be importable and the full test exercise.
pytest.importorskip(
    "bedrock_agentcore_sdk",
    reason="bedrock_agentcore_sdk not installed (optional [agentcore] extra — see Plan 4 T41)",
)

from firm.agents.reporter import make_reporter
from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision


# Fixed reference time used by BOTH reporter invocations so the ``"ts"``
# field on the emitted JSONL row matches byte-for-byte.
_FIXED_NOW = datetime(2026, 5, 22, 14, 30, 0, tzinfo=timezone.utc)


def _fixture_decision() -> Decision:
    """Realistic Decision matching the Plan 1/2 shape (BUY w/ ticker + shares)."""
    return Decision(
        id="dec-agentcore-1",
        decision_id_chain=[],
        action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="Adapter-equivalence fixture.",
        confidence=0.75,
        citations=[],
        falsification_condition="If revenue declines.",
        escalation_reason=None,
        failure_mode=None,
        metadata={},
        nonce="agentcore-nonce-1",
    )


def _fixture_state() -> dict[str, Any]:
    """Plain-dict WorkingState fixture: this is what crosses the AgentCore
    boundary as JSON. The Reporter's ``_serialize_value`` calls
    ``model_dump(mode="json")`` on Pydantic objects and passes dicts
    through unchanged, so a dict carrying a pre-dumped Decision and a
    LangGraph-side WorkingState carrying the live Decision produce
    identical JSONL output. We use the dict form here and convert it to a
    typed-Decision form for the LangGraph leg by re-injecting the Decision
    object below.
    """
    return {
        "heartbeat_at": "2026-05-22T14:30:00+00:00",
        "execution_result": {"ticker": "AAPL", "filled_shares": "10"},
    }


def _read_bytes(p: Path) -> bytes:
    """Read on-disk JSONL exactly as written — no decoding, no normalisation."""
    return p.read_bytes()


def test_agentcore_reporter_byte_equivalent_to_langgraph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same input → byte-identical ``decisions.jsonl`` from both paths."""
    # --- Determinism: silence the OTel ``trace_id`` source so both
    # invocations stamp the same (empty) value into the JSONL row. We
    # patch the symbol used by ``firm.agents.reporter`` — that module
    # imports ``opentelemetry.trace`` as ``trace`` and calls
    # ``trace.get_current_span()``. Returning ``INVALID_SPAN`` makes
    # ``ctx.trace_id == 0`` so the reporter writes ``"trace_id": ""``.
    from opentelemetry.trace import INVALID_SPAN

    import firm.agents.reporter as reporter_mod

    monkeypatch.setattr(
        reporter_mod.trace, "get_current_span", lambda: INVALID_SPAN
    )

    # ------------------------------------------------------------------
    # Path A — LangGraph reporter (direct closure invocation).
    # ------------------------------------------------------------------
    lg_root = tmp_path / "langgraph"
    lg_clock = ReplayClock(_FIXED_NOW)
    lg_reporter = make_reporter(reports_root=lg_root, clock=lg_clock, db_path=None)

    state = _fixture_state()
    # Inject the live Decision on the LangGraph side. The adapter side
    # will receive the same Decision as a JSON-dumped dict (see below).
    decision = _fixture_decision()
    lg_state: dict[str, Any] = {**state, "risk_decision": decision}
    lg_out = lg_reporter(lg_state)
    lg_path = Path(lg_out["report_path"])
    assert lg_path.exists(), "LangGraph reporter must write decisions.jsonl"

    # ------------------------------------------------------------------
    # Path B — AgentCore-served reporter (through @agent decorator).
    # ------------------------------------------------------------------
    ac_root = tmp_path / "agentcore"
    # The adapter constructs its closure at import time from env vars, so
    # we set the env BEFORE (re)importing the module. ``FIRM_DB_PATH`` is
    # intentionally unset so the closure mirrors Path A.
    monkeypatch.setenv("FIRM_REPORTS_ROOT", str(ac_root))
    monkeypatch.delenv("FIRM_DB_PATH", raising=False)

    # Force re-import so the module-level ``_reporter`` closure picks up
    # the freshly-set env vars (importlib.reload re-runs module top-level).
    import firm.agentcore.reporter_adapter as adapter_mod

    adapter_mod = importlib.reload(adapter_mod)

    # Replace the adapter's WallClock-backed closure with the same
    # ReplayClock used on Path A so ``"ts"`` matches byte-for-byte. We
    # rebuild the closure rather than monkeypatch internal clock access
    # because the closure captures the Clock at construction time.
    adapter_mod._reporter = make_reporter(
        reports_root=ac_root, clock=ReplayClock(_FIXED_NOW), db_path=None
    )

    # JSON-marshal the same state the LangGraph leg received. The
    # Decision goes through ``model_dump(mode="json")`` so the dict
    # crossing the wire matches what ``_serialize_value`` would produce
    # for the live Decision object on the LangGraph leg.
    ac_payload: dict[str, Any] = {**state, "risk_decision": decision.model_dump(mode="json")}

    # Build an InvocationRequest. The SDK is pre-1.0 so the request
    # shape may drift; we use the SDK's actual class to keep the test
    # honest about what's deployed.
    InvocationRequest = adapter_mod.InvocationRequest
    try:
        req = InvocationRequest(payload=ac_payload)
    except TypeError:
        # SDK signature deviation — try positional. Document the actual
        # signature in this comment if this branch fires in CI:
        req = InvocationRequest(ac_payload)  # type: ignore[misc]

    resp = adapter_mod.reporter(req)

    # The response body is JSON-encoded ``{"report_path": str}``.
    body = resp.body if isinstance(resp.body, str) else resp.body.decode("utf-8")
    result = json.loads(body)
    ac_path = Path(result["report_path"])
    assert ac_path.exists(), "AgentCore-served reporter must write decisions.jsonl"

    # ------------------------------------------------------------------
    # Byte-equivalence assertion.
    # ------------------------------------------------------------------
    lg_bytes = _read_bytes(lg_path)
    ac_bytes = _read_bytes(ac_path)
    assert lg_bytes == ac_bytes, (
        "AgentCore-served decisions.jsonl must be byte-equivalent to the "
        "LangGraph-served decisions.jsonl for the same WorkingState input.\n"
        f"LangGraph ({lg_path}): {lg_bytes!r}\n"
        f"AgentCore ({ac_path}): {ac_bytes!r}"
    )
