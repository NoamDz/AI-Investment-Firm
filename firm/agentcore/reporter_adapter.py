"""firm/agentcore/reporter_adapter.py — AgentCore Runtime adapter for the Reporter.

Wraps the LangGraph Reporter closure (``firm.agents.reporter.make_reporter``)
so it can be served via AWS Bedrock AgentCore Runtime. The adapter is a
thin marshalling layer: ``InvocationRequest.payload`` (JSON dict mirroring
``WorkingState``) is fed straight into the closure, and the resulting
``{"report_path": str}`` dict is JSON-serialised back into the response
body. All reporting logic stays in ``firm/agents/reporter.py`` so the
AgentCore-served and LangGraph-served outputs land byte-for-byte
identical on disk (asserted in ``tests/integration/test_agentcore_reporter.py``).

Lazy-imports ``bedrock_agentcore_sdk``: if the optional ``[agentcore]``
extra is not installed (T41), importing this module raises a helpful
``ImportError`` pointing the operator at ``pip install -e .[agentcore]``.
The core LangGraph path never touches this module, so callers without the
extra are unaffected.

Marshalling contract (input → adapter → Reporter):
  AgentCore ``InvocationRequest.payload`` (JSON) ──┐
                                                   ├─→ reporter(state: WorkingState)
                                                   │     (closure built once at module load)
                                                   └─→ dict {"report_path": str}
                                                          │
                                                          └─→ AgentCore ``InvocationResponse.body`` (JSON str)

Environment variables consumed at import time:
  * ``FIRM_REPORTS_ROOT`` (default ``"reports"``) — root directory the
    closure writes ``<YYYY-MM-DD>/decisions.jsonl`` under.
  * ``FIRM_DB_PATH``      (default unset → ``None``) — when set, the
    closure also persists Decisions into the configured SQLite DB.

Re-import semantics: because the closure is built at import time, callers
that mutate the env vars after import must reload this module
(``importlib.reload``) for the new values to take effect. The
byte-equivalence test relies on this and is the canonical example.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

try:
    # Pre-1.0 SDK — actual decorator / request / response signatures may
    # drift. If the installed SDK exposes different names, update the
    # ``@agent`` call and the ``InvocationRequest`` / ``InvocationResponse``
    # marshalling sites below in lockstep with the SDK release notes.
    from bedrock_agentcore_sdk import (  # type: ignore[import-not-found]
        InvocationRequest,
        InvocationResponse,
        agent,
    )
except ImportError as e:  # pragma: no cover — optional extra
    raise ImportError(
        "AgentCore SDK is not installed. Run `pip install -e .[agentcore]` "
        "to install the optional dependency."
    ) from e

from firm.agents.reporter import make_reporter
from firm.core.clock import WallClock

# ---------------------------------------------------------------------------
# Module-level reporter closure — constructed once at import time.
# Reports root and db_path are read from env vars so the adapter is
# configurable without code changes (T41 wires container env). The
# ``Clock`` protocol's reference implementation is ``WallClock`` (UTC); the
# closure stamps ``clock.now()`` onto every JSONL row.
# ---------------------------------------------------------------------------
_REPORTS_ROOT = Path(os.environ.get("FIRM_REPORTS_ROOT", "reports"))
_DB_PATH_ENV = os.environ.get("FIRM_DB_PATH")
_DB_PATH = Path(_DB_PATH_ENV) if _DB_PATH_ENV else None

_reporter = make_reporter(
    reports_root=_REPORTS_ROOT,
    clock=WallClock(),
    db_path=_DB_PATH,
)


# The ``@agent`` decorator name (``"firm-reporter"``) is the contract with
# the Terraform-shipped ``module.bedrock`` outputs — see
# ``infra/terraform/modules/bedrock/outputs.tf`` (``agentcore_runtime_name``
# at line 21). If that output drifts, this decorator name must update in
# lockstep. ``memory_namespace=None`` because the Reporter is stateless —
# every invocation is fully driven by the incoming ``WorkingState`` and the
# on-disk JSONL.
@agent(name="firm-reporter", memory_namespace=None)  # type: ignore[untyped-decorator]
def reporter(request: "InvocationRequest") -> "InvocationResponse":
    """AgentCore entrypoint for the Reporter agent.

    Marshals ``request.payload`` (a JSON dict whose keys mirror
    :class:`firm.orchestrator.state.WorkingState`) into the LangGraph
    Reporter closure, then wraps the closure's ``{"report_path": str}``
    return value into an ``InvocationResponse`` with a JSON body.

    The closure performs the same on-disk effects as the LangGraph path:
    appending one row to ``<reports_root>/<YYYY-MM-DD>/decisions.jsonl``
    and (when ``FIRM_DB_PATH`` is set) persisting any top-level
    :class:`firm.core.models.Decision` values to the ``decisions`` table.
    Decision objects arrive across the AgentCore boundary as plain dicts
    (the JSON round-trip strips Pydantic types), which means DB
    persistence is a no-op for AgentCore-served invocations — the
    AgentCore deployment path treats the JSONL as the source of truth.
    """
    payload = request.payload
    state = json.loads(payload) if isinstance(payload, (str, bytes)) else payload

    result = _reporter(state)  # returns {"report_path": str}
    body = json.dumps(result)
    return InvocationResponse(body=body, content_type="application/json")
