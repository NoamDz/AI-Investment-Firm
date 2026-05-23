# AgentCore Migration Mapping

> Maps the AI Investment Firm's current LangGraph-based architecture to
> AWS Bedrock AgentCore primitives. Reference for Plan 4 §G and spec §11.1.
> Last updated: 2026-05-22 (Plan 4 T39).

## Status

| Migration scope | Status |
|---|---|
| Mapping documented | ✓ T39 |
| Reporter adapter shipped | ✓ T40 / T41 |
| Full migration applied | Out of scope (post-Plan-4) |

Only the Reporter agent ships on AgentCore in Plan 4. Other agents remain
LangGraph for the demo but are AgentCore-ready by interface (see "Migration
strategy" below).

---

## Migration Table (Expanded)

| Component (file:line) | LOC | AgentCore primitive | Adapter notes |
|---|---|---|---|
| `firm/orchestrator/graph.py:35` — `build_graph()` / `StateGraph` / `compile()` | 84 | AgentCore Runtime | Wrap each node in `@agent` decorator; Runtime handles dispatch, retries, timeouts. ~50 LOC. |
| `firm/tools/fundamentals.py:53`, `firm/tools/risk_metrics.py:34` — `FundamentalsTool`, `RiskMetricsTool` (MCP-style) | 226 + 205 | AgentCore Gateway (MCP-native) | Direct — tools already implement the MCP `tool_def` schema; Gateway ingests them without code change. |
| `firm/orchestrator/state.py:11` — `WorkingState` (TypedDict, 14 fields) | 47 | AgentCore Memory | Schema mapping: serialize `WorkingState` keys to Memory namespace JSON. ~50 LOC. |
| `firm/obs/tracer.py:60` — `JsonlFileExporter`, `init_tracer()`; `firm/obs/spans.py:66` — `agent_span()` | 381 + 158 | AgentCore Observability | Exporter swap: replace `JsonlFileExporter` with AgentCore's OTLP sink. `OTEL_EXPORTER=otlp` path already wired (line 281). |
| `firm/hitl/signing.py:172` — `verify_with_rotation()`; `firm/agents/hitl.py:22` — `make_hitl()` | 246 + 120 | AgentCore Identity | API binding: map `verify_with_rotation` HMAC checks to Identity provider attestation calls. ~30 LOC. |
| `firm/orchestrator/graph.py:83` — `SqliteSaver(conn, serde=_FIRM_SERDE)` | (within 84) | AgentCore Runtime checkpointer | Adapter: replace `SqliteSaver` with AgentCore's managed checkpointer; preserve `_FIRM_SERDE` msgpack allowlist. |

---

### LangGraph Orchestrator → AgentCore Runtime

**Current:** `firm/orchestrator/graph.py:35` — `build_graph()` assembles a
`StateGraph(WorkingState)` with seven nodes (monitor → research → pm → risk →
hitl|execution → reporter), sets the entry point, wires conditional edges, and
at line 83 calls `g.compile(checkpointer=saver, interrupt_before=["hitl"])`.
`firm/orchestrator/state.py:11` defines `WorkingState` (47 LOC). Total across
both files: 131 LOC.

**Migration:** Wrap each LangGraph node's invocation surface in AgentCore's
`@agent` decorator. The Runtime handles invocation dispatch, retries, and
timeouts. Estimated ~50 LOC adapter: one decorator per node function plus a
small dispatch shim that maps AgentCore's `InvocationRequest` to the node's
`WorkingState` slice.

**Blast radius:** High — the orchestrator is the central nervous system;
a bad migration silently breaks all heartbeats. Migrate last, after all
leaf agents are individually validated.

---

### MCP Servers → AgentCore Gateway (MCP-native)

**Current:** `firm/tools/fundamentals.py:53` — `FundamentalsTool` exposes a
`tool_def: ClassVar[ToolDef]` with `name`, `description`, and JSON `input_schema`
matching the Anthropic `tools=` payload format (MCP-style, per module docstring
line 1). `firm/tools/risk_metrics.py:34` — `RiskMetricsTool` mirrors the same
pattern for `volatility_30d`, `beta_180d`, `max_drawdown_90d`. Combined: 431 LOC.

**Migration:** Direct — AgentCore Gateway is MCP-native and ingests the existing
`ToolDef` descriptors without modification. No adapter LOC required. The only
operational change is deploying the tools as Gateway-managed endpoints rather
than in-process callables.

**Blast radius:** Low — tools are stateless PIT parquet lookups. The worst
outcome of a Gateway misconfiguration is a `KeyError` surfaced as
`INSUFFICIENT_EVIDENCE`, which the existing failure-mode chain handles.

---

### DeskState / WorkingState → AgentCore Memory

**Current:** `firm/orchestrator/state.py:11` — `WorkingState` (TypedDict with
14 keys: `heartbeat_at`, `research_decision`, `pm_decision`, `risk_decision`,
`retrieved_chunks`, `claims`, `sufficiency_result`, `tool_call_ids`, `pm_votes`,
`sufficiency_status`, `human_override_ack`, `hitl_required`, `hitl_approved`,
`execution_result`, `report_path`, `notes`). Spec §11.1 refers to this as
"DeskState / TradeJournal" — in the current codebase `WorkingState` is the
unified on-graph state that serves both roles.

**Migration:** Schema mapping: serialize each `WorkingState` key to the AgentCore
Memory namespace `firm-desk-state` (the namespace name is also the
`agentcore_memory_namespace` Terraform output at
`infra/terraform/modules/bedrock/outputs.tf:26`). The `Decision` values (Pydantic
models) serialize via `model_dump(mode="json")`; `add_messages`-annotated `notes`
need merge-semantics preserved. Estimated ~50 LOC.

**Blast radius:** Medium-high — a leaked namespace across tenants would expose
positions and rationales. Verify AgentCore Memory's isolation model before
migrating (see "Open questions / SDK risks" below).

---

### OpenTelemetry Traces → AgentCore Observability

**Current:** `firm/obs/tracer.py:60` — `JsonlFileExporter(SpanExporter)` writes
spans to `traces/YYYY-MM-DD/run-<run_id>.jsonl`. `init_tracer()` at line 255
already supports `OTEL_EXPORTER=otlp` via lazy import of
`OTLPSpanExporter` (lines 314–355). `firm/obs/spans.py:66` — `agent_span()`,
`llm_span()`, `tool_span()`, `retrieval_span()` are the four canonical span
helpers (158 LOC). Total observability surface: 539 LOC.

**Migration:** Exporter swap — set `OTEL_EXPORTER=otlp` and point the OTLP
endpoint at AgentCore's Observability collector. No code changes needed; the
`_init_otlp_provider()` path (tracer.py:314) is already implemented. The
`agentcore_log_group_name` output at `infra/terraform/modules/bedrock/outputs.tf:31`
names the CloudWatch log group that AgentCore Observability writes to.

**Blast radius:** Low — span emission is fire-and-forget; a misconfigured exporter
drops traces, it does not corrupt decisions or orders.

---

### HITL Signed Approvals → AgentCore Identity

**Current:** `firm/hitl/signing.py:172` — `verify_with_rotation()` implements
dual-key HMAC-SHA256 approval verification with a 24-hour grace window. The
primary secret lives at `firm/firm_hmac_secret` (Secrets Manager path, Plan 3
T35); the previous key at `firm/firm_hmac_secret_prev`. `firm/agents/hitl.py:22`
— `make_hitl()` orchestrates the ESCALATE branch: queues the decision in
`hitl_queue`, fires Slack notification, and reads the resulting `approved` status.
Combined: 366 LOC.

**Migration:** API binding — map `verify_with_rotation` HMAC checks to AgentCore
Identity's attestation API. The bespoke HMAC rotation scheme (`firm/firm_hmac_secret_prev`)
is not natively supported by AgentCore Identity; an adapter must bridge the two
schemes (~30 LOC). The `agentcore_identity_secret_arn` Terraform output at
`infra/terraform/modules/bedrock/outputs.tf:37` points to the primary secret.

**Blast radius:** High — a mismatch between HMAC verification and Identity
attestation could allow unsigned approvals through (false positive) or block
all legitimate approvals (false negative). Validate in staging with the Slack
approval E2E test before cutting over.

---

### SqliteSaver → AgentCore Runtime Checkpointer

**Current:** `firm/orchestrator/graph.py:8` — imports `SqliteSaver` from
`langgraph.checkpoint.sqlite`. Line 81–84 open a `sqlite3.connect()` and
construct `SqliteSaver(conn, serde=_FIRM_SERDE)`, where `_FIRM_SERDE` is a
`JsonPlusSerializer` with an msgpack allowlist for three `firm.core.models` types
(lines 20–26). Passed to `g.compile(checkpointer=saver, interrupt_before=["hitl"])`.

**Migration:** Adapter — replace `SqliteSaver` with AgentCore's managed
checkpointer. The `_FIRM_SERDE` msgpack allowlist must be re-registered with the
new checkpointer so that `firm.core.models.Decision`, `ActionEnum`, and
`FailureMode` survive round-trip serialization. LOC estimate: ~30 LOC adapter
plus validation of the checkpoint schema against AgentCore's expected format (see
"Open questions / SDK risks" below).

**Blast radius:** Medium — the HITL `interrupt_before=["hitl"]` pattern depends
on the checkpointer correctly parking and resuming graph state across process
restarts. A serialization mismatch would silently corrupt the resume payload.

---

## Why Reporter First

**1. No broker calls.** The Reporter agent is a pure projection: it reads
`WorkingState` values (decisions, cost ledger balance via `_cost_today_usd`,
OTel trace pointer) and emits a JSONL row plus a `report_path` string. It never
calls `firm.broker`, never writes to `hitl_queue`, and never mutates any upstream
state key. If the AgentCore adapter mis-marshals the payload, the worst outcome
is a malformed or missing summary row — never an unintended trade or a corrupted
approval record.

**2. No state mutation.** Reporter's `make_reporter()` closure (
`firm/agents/reporter.py:88`) is a pure projection: `WorkingState` input →
JSONL append → `{"report_path": str(path)}` output. It has no Memory namespace
to coordinate with, no Identity scopes to attest, and no checkpoint shape to
preserve. The smallest blast-radius slice possible: if the AgentCore-served
Reporter fails, the LangGraph-served Reporter continues to run unaffected while
the team debugs.

**3. Stable interface.** The Reporter's entry point is `make_reporter()` which
returns a closure with signature `reporter(state: WorkingState) -> dict[str, Any]`
(`firm/agents/reporter.py:91`). The closure has three dependencies baked in at
construction time (`reports_root`, `clock`, `db_path`) and one input at call
time (the full `WorkingState`). The AgentCore adapter marshals the JSON payload
into the state dict and calls the closure. Marshalling is straightforward: all
`WorkingState` values are either primitives, ISO strings, or Pydantic models
that serialize via `model_dump(mode="json")`.

**4. Byte-for-byte testable.** Because Reporter is deterministic given fixed
inputs (frozen `clock`, fixed `WorkingState`), the AgentCore-served output can
be compared byte-for-byte against the LangGraph-served output in a CI test. This
hard correctness gate (`tests/integration/test_agentcore_reporter.py`, per T40
spec) gives immediate signal if an SDK update changes marshalling behavior.

**5. Discardable on SDK churn.** `bedrock-agentcore-sdk` is pre-1.0 (per Plan 4
risk-mitigations §"AgentCore SDK churn → T41"). If the SDK breaks, the rest of
the firm keeps working: the `[agentcore]` optional extra (T41) isolates the
dependency from the core install. Removing the `[agentcore]` extra in CI restores
full test-suite green without touching any other agent.

---

## `firm/agentcore/reporter_adapter.py` — Design Sketch

This module does not exist yet; T40 implements it. The sketch below defines the
interface contract T40 must satisfy.

```python
"""firm/agentcore/reporter_adapter.py — AgentCore Runtime adapter for the Reporter.

Lazy-imports `bedrock_agentcore_sdk`: if the optional `[agentcore]` extra
is not installed, importing this module raises a helpful ImportError
pointing the operator at `pip install -e .[agentcore]`.

Marshalling contract (input → adapter → Reporter):
  AgentCore `InvocationRequest.payload` (JSON) ──┐
                                                  ├─→ reporter(state: WorkingState)
  decisions + cost_ledger + traces ──────────────┘
                                                  └─→ dict {"report_path": str}
                                                         │
                                                         └─→ AgentCore `InvocationResponse.body` (str)
"""

from __future__ import annotations

try:
    from bedrock_agentcore_sdk import agent, InvocationRequest, InvocationResponse
except ImportError as e:  # pragma: no cover — optional extra
    raise ImportError(
        "AgentCore SDK is not installed. Run `pip install -e .[agentcore]` "
        "to install the optional dependency."
    ) from e

from firm.agents.reporter import make_reporter   # construct closure at module load
from firm.core.clock import Clock
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Module-level reporter closure — constructed once at import time.
# Reports root and db_path are read from env vars so the adapter is
# configurable without code changes (see T41 for env var wiring).
# ---------------------------------------------------------------------------
import os as _os

_reporter = make_reporter(
    reports_root=Path(_os.environ.get("FIRM_REPORTS_ROOT", "reports")),
    clock=Clock(),
    db_path=Path(_os.environ.get("FIRM_DB_PATH", "firm.db")) if _os.environ.get("FIRM_DB_PATH") else None,
)


@agent(name="firm-reporter", memory_namespace=None)  # no Memory — Reporter is stateless
def reporter(request: InvocationRequest) -> InvocationResponse:
    """AgentCore entrypoint for the Reporter agent.

    Marshals `request.payload` (JSON dict whose keys mirror WorkingState)
    into the LangGraph Reporter's typed args, invokes the closure, returns
    the report_path as the response body.
    """
    payload = request.payload
    state = json.loads(payload) if isinstance(payload, str) else payload

    result = _reporter(state)          # returns {"report_path": str}
    body = json.dumps(result)
    return InvocationResponse(body=body, content_type="application/json")
```

The adapter is intentionally thin — all reporting logic stays in
`firm/agents/reporter.py` so the AgentCore-served and LangGraph-served
outputs are byte-for-byte identical. The byte-equality assertion lives in
`tests/integration/test_agentcore_reporter.py` (T40).

The `@agent` decorator name (`firm-reporter`) and the absence of a
`memory_namespace` argument match the Terraform-shipped `module.bedrock`
outputs at `infra/terraform/modules/bedrock/outputs.tf` —
`agentcore_runtime_name` (line 21) and `agentcore_memory_namespace` (line 26).
If those output names drift, this adapter must update in lockstep.

---

## Migration Strategy

Migrate agents in order of increasing blast radius: **Reporter → Research → PM
→ Risk → Execution**, with the orchestrator and SqliteSaver checkpointer migrated
last once all leaf agents have been individually validated. Each migration follows
the same gate: write a byte-equality or schema-equality integration test (modeled
on `tests/integration/test_agentcore_reporter.py`), land the adapter, confirm
the test is green in CI on the `main` workflow, then merge. The HITL signed-approval
bridge to AgentCore Identity and the Memory namespace migration for `WorkingState`
are pre-requisites for the Risk and Execution agents respectively; both carry
higher blast radius and require staging validation before production cut-over.
Full migration is out of scope for Plan 4 (post-Plan-4 track, see
`docs/path-to-production.md` T44).

---

## Open Questions / SDK Risks

- **`bedrock-agentcore-sdk` is pre-1.0.** Surface churn between releases is
  expected. The `[agentcore]` optional extra (T41) isolates the dependency so
  a breaking SDK change does not block CI for operators who have not installed it.

- **AgentCore Runtime checkpointer serialization shape.** It is not yet confirmed
  that the managed checkpointer accepts the `JsonPlusSerializer` msgpack allowlist
  that `SqliteSaver` uses for `firm.core.models` types. A shape mismatch would
  silently corrupt resume payloads for HITL-interrupted heartbeats. This is TBD
  and not blocking the Reporter migration (Reporter does not use the checkpointer).

- **AgentCore Memory namespace isolation.** Before migrating `WorkingState` to
  AgentCore Memory (`firm-desk-state` namespace), the namespace isolation model
  must be verified: a cross-tenant state leak would expose positions and rationales.
  This is a security gate, not a performance concern.

- **AgentCore Identity and HMAC rotation.** `firm/hitl/signing.py:172`'s
  `verify_with_rotation` uses a bespoke dual-key scheme (`firm/firm_hmac_secret`
  + `firm/firm_hmac_secret_prev`, Plan 3 T35). AgentCore Identity's attestation
  model may not natively support this rotation pattern; the adapter (~30 LOC
  per the migration table) must be validated against the Identity API before
  the HITL migration proceeds.

- **Multi-region AgentCore.** Out of scope for Plan 4. The path-to-production
  document (`docs/path-to-production.md`, T44) will cover multi-region
  AgentCore topology and the associated Terraform changes.

---

*Document generated for Plan 4 T39.*
