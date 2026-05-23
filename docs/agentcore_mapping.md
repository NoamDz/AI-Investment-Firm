# AgentCore Migration Mapping

Maps the firm's LangGraph architecture to AWS Bedrock AgentCore primitives. Spec §11.1.

**Plan 4 status:** Reporter adapter shipped (`firm/agentcore/reporter_adapter.py`, T40/T41). Other agents stay on LangGraph for the demo but are AgentCore-ready by interface.

## Migration table

| Component | File:line | AgentCore primitive | Adapter LOC | Blast radius |
|---|---|---|---|---|
| LangGraph orchestrator (`build_graph` + 7 nodes) | `firm/orchestrator/graph.py:35` | Runtime — `@agent` per node + dispatch shim | ~50 | **High** — central nervous system; migrate last |
| MCP tools (`FundamentalsTool`, `RiskMetricsTool`) | `firm/tools/fundamentals.py:53`, `risk_metrics.py:34` | Gateway (MCP-native) — ingests existing `ToolDef` | 0 | Low — stateless PIT lookups; failure → `INSUFFICIENT_EVIDENCE` |
| `WorkingState` (TypedDict, 14 keys) | `firm/orchestrator/state.py:11` | Memory — namespace `firm-desk-state` | ~50 | Medium-high — cross-tenant leak exposes positions |
| OTel traces (`JsonlFileExporter` + `agent_span`) | `firm/obs/tracer.py:60`, `spans.py:66` | Observability — exporter swap (`OTEL_EXPORTER=otlp` path already wired at `tracer.py:281`) | 0 | Low — fire-and-forget; misconfig drops traces, doesn't corrupt decisions |
| HITL signed approvals (`verify_with_rotation` + dual-key) | `firm/hitl/signing.py:172`, `agents/hitl.py:22` | Identity — API binding; bespoke dual-key not native | ~30 | **High** — mismatch admits unsigned approvals or blocks all |
| `SqliteSaver` checkpointer | `firm/orchestrator/graph.py:83` | Runtime managed checkpointer; re-register `_FIRM_SERDE` msgpack allowlist | ~30 | Medium — silent resume-payload corruption breaks HITL park/resume |

Terraform outputs at `infra/terraform/modules/bedrock/outputs.tf` name the corresponding AWS resources: `agentcore_runtime_name:21`, `agentcore_memory_namespace:26`, `agentcore_log_group_name:31`, `agentcore_identity_secret_arn:37`.

## Why Reporter first

Smallest blast-radius slice possible:

1. **No broker, no state mutation.** Pure projection: `WorkingState` → JSONL row + `report_path`. If marshalling breaks, worst case is a malformed summary — never an unintended trade.
2. **Stable interface.** `make_reporter() -> Callable[[WorkingState], dict]` with three construction-time deps (`reports_root`, `clock`, `db_path`). All inputs are primitives, ISO strings, or `model_dump(mode="json")`-able Pydantic models.
3. **Byte-for-byte testable.** Deterministic given fixed clock + state. `tests/integration/test_agentcore_reporter.py` asserts AgentCore-served output == LangGraph-served output byte-for-byte. SDK regressions surface immediately.
4. **Discardable on SDK churn.** `bedrock-agentcore-sdk` is pre-1.0. The `[agentcore]` optional extra (T41) isolates the dependency — uninstalling it restores green CI without touching any other agent.

## Adapter design

`firm/agentcore/reporter_adapter.py` is intentionally thin — all reporting logic stays in `firm/agents/reporter.py` so the two paths are byte-equivalent.

- Lazy-import `bedrock_agentcore_sdk`; missing extra raises a pointed `ImportError`.
- Construct `_reporter = make_reporter(...)` once at import (env vars `FIRM_REPORTS_ROOT`, `FIRM_DB_PATH`).
- `@agent(name="firm-reporter", memory_namespace=None)` — Reporter is stateless, no Memory.
- Entrypoint: `request.payload` (JSON) → `WorkingState` dict → `_reporter(state)` → `{"report_path": str}` → `InvocationResponse`.

The `@agent` name and the absence of `memory_namespace` track `agentcore_runtime_name` / `agentcore_memory_namespace` in the Terraform outputs above — drift those names and update in lockstep.

## Migration order

**Reporter → Research → PM → Risk → Execution → orchestrator/checkpointer last.** Each step:

1. Write a byte- or schema-equality integration test (model on `test_agentcore_reporter.py`).
2. Land the adapter.
3. Confirm CI green on `main` workflow.
4. Merge.

Risk and Execution have pre-requisites — Identity bridge for HITL, Memory namespace for `WorkingState`. Both require staging validation. Full migration is post-Plan-4 (see [`path-to-production.md`](path-to-production.md)).

## Open questions / SDK risks

- **SDK churn** — pre-1.0, expected. `[agentcore]` extra isolates blast.
- **Checkpointer serialization shape** — unconfirmed whether AgentCore's managed checkpointer accepts our `JsonPlusSerializer` msgpack allowlist. Shape mismatch → silent resume corruption. Not blocking Reporter (no checkpointer use).
- **Memory namespace isolation** — security gate before `WorkingState` migration. Cross-tenant leak exposes positions/rationales.
- **Identity vs HMAC rotation** — `verify_with_rotation`'s dual-key scheme may not be native to AgentCore Identity; adapter must bridge.
- **Multi-region AgentCore Memory** — out of scope. See [`path-to-production.md`](path-to-production.md) §3.
