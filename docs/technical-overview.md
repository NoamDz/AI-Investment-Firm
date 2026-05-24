# Technical overview

The complement to [`architecture.md`](architecture.md): what each agent contracts on, what flows through `WorkingState`, and how the system fails on purpose.

## Agents

Each agent is a `Callable[[WorkingState], dict[str, Any]]` factory (`make_*`) under `firm/agents/`. Factories take their dependencies (clock, db_path, broker, LLM client, …) and return the closure LangGraph calls per node.

| Agent | File | Reads | Emits | Notes |
|-------|------|-------|-------|-------|
| `monitor` | `firm/agents/monitor.py` | clock | `heartbeat_at` (ISO 8601) | One line. Stamps the thread id for the checkpointer. |
| `research` | `firm/agents/research.py` | universe, retriever, extractor, judge | `research_decision`, `claims`, `retrieved_chunks`, `sufficiency_*`, `tool_call_ids` | Grounded path: Citations API → sufficiency judge → BUY/HOLD/ESCALATE/REFUSE. Legacy stub path runs when any of the three deps is missing (used by Plan 1 unit tests). |
| `pm` | `firm/agents/pm.py` | `claims`, `sufficiency_status`, `human_override_ack` | `pm_decision`, `pm_votes` | Three single-lens voters (quality / valuation / catalyst) via `LangGraph Send`, deterministic `aggregate_votes`. Chinese-wall: never calls retrieval or tools. |
| `risk` | `firm/agents/risk.py` | `pm_decision`, quote, positions, policy | `risk_decision` | Deterministic gates: quote age, gross/net exposure, per-name & per-sector caps, drawdown, daily trade count. Breach → REFUSE with `RISK_LIMIT_BREACHED`. Above sizing thresholds → ESCALATE. |
| `hitl` | `firm/agents/hitl.py` | `risk_decision`, Slack signed payload | `hitl_approved`, possibly `risk_decision` rewrite | Reached only on `ESCALATE`. Gated by `interrupt_before=["hitl"]`; resumes on `make dev-ack` or signed Slack approval. Reaper closes stale items as `UNAPPROVED_HIGH_RISK` after `DEFAULT_HITL_TIMEOUT_SECONDS=1800`. |
| `execution` | `firm/agents/execution.py` | `risk_decision`, `hitl_approved`, broker | `execution_result` | Wraps the outbox. Idempotent on HMAC nonce. Broker retry exhaustion → REFUSE with `BROKER_UNAVAILABLE` chained to the unfilled risk decision. |
| `reporter` | `firm/agents/reporter.py` | full `WorkingState`, `firm.db` | `report_path`, persisted `decisions` rows | Pure projection — no state mutation other than persisting per-heartbeat decision rows. The Bedrock AgentCore adapter (`firm/agentcore/reporter_adapter.py`) hosts this exact callable; see [`agentcore_mapping.md`](agentcore_mapping.md). |

## WorkingState

`firm/orchestrator/state.py` — `TypedDict(total=False)` with 14 keys (file is the source of truth). Lifecycle:

1. `monitor` sets `heartbeat_at` — becomes the LangGraph thread id, so an interrupt resumes the **same** heartbeat instead of a fresh one.
2. `research` populates `research_decision` + the four grounding sidecars (`claims`, `retrieved_chunks`, `sufficiency_result`, `sufficiency_status`). PM and HITL reviewers read these directly so they can introspect without re-running retrieval.
3. `pm` appends `pm_decision` + `pm_votes` (one per lens).
4. `risk` appends `risk_decision` and (when escalating) `hitl_required=True`.
5. `hitl` mutates `hitl_approved`; if approved, downstream `execution` proceeds; if denied or reaped, the risk decision is rewritten in place to REFUSE.
6. `execution` appends `execution_result`.
7. `reporter` writes `report_path` and persists all Decisions to `firm.db`.

State serialization through the `SqliteSaver` uses a `JsonPlusSerializer` with `Decision`, `ActionEnum`, `FailureMode` in its msgpack allowlist (`firm/orchestrator/graph.py:20`) so checkpoint reloads do not emit deserialization warnings.

## Partial-failure model

Every refusal is **typed**, never a silent default. `FailureMode` (`firm/core/models.py:19`) is a 15-value `StrEnum` with `UNKNOWN` as the catch-all:

```
uncited_claim · insufficient_evidence · prompt_injection_detected · risk_limit_breached
hitl_timeout · schema_validation_failed · llm_unavailable · stale_data
ungrounded_claim · tool_permission_denied · unapproved_high_risk · broker_unavailable
reconciliation_drift · signed_approval_invalid · unknown
```

Every value has a triggering fixture in `tests/eval/failure_modes/`. The red-team suite (`python -m firm.cli red-team`) asserts each one fires on its intended adversarial input and never on a clean one. See [`eval.md`](eval.md) for the coverage matrix.

Decisions chain via `Decision.decision_id_chain` so a REFUSE always points back to the upstream decision it superseded — the audit log preserves the parent-id graph for EOD reconciliation.

## Cost router + LLM cache

Every LLM call goes through one chokepoint:

1. **Cost router** (`config/router.yaml`) — feature-gated model selection with `sonnet → haiku` fallback. Read by `firm/llm/router.py`.
2. **Content-addressed SQLite cache** — keyed by `hash_prompt(model_id, system, messages, tool_specs, …)` so the same logical call returns the same bytes regardless of which model the router picked at the time. Under `FIRM_LLM_MODE=cached` (eval, red-team, CI) the pipeline is byte-deterministic and offline. Cache rows live in the `llm_cache` table; cost rows in `cost_ledger`.

The hash is model-independent on purpose: cassette re-recording can swap models without invalidating siblings.

## Guardrails (where the safety budget is spent)

| Layer | Defense | Implementation |
|-------|---------|----------------|
| Retrieval | Verbatim grounding | Anthropic Citations API; each `Claim` carries `source_quote` lifted from the cited chunk's `cited_text` (no paraphrase). |
| Retrieval | Hybrid relevance | BM25 + Nomic-embed-text-v1.5 dense + BGE-reranker-v2-m3, in `firm/rag/retrieve.py`. |
| Reasoning | Sufficiency re-read | Haiku judge re-reads the exact passages used to extract each claim; labels per-claim `ok` / `partial` / `insufficient`. Aggregate `insufficient` → REFUSE. |
| Reasoning | Diversity | Three PM lenses in parallel, majority vote required to advance. |
| Action | Hard limits | Deterministic risk gates (not LLM-judged): gross/net exposure, per-name & per-sector caps, drawdown, quote-age. |
| Action | Human gate | `interrupt_before=["hitl"]` on ESCALATE; signed Slack approval (HMAC-SHA256 dual-key rotation in `firm/hitl/signing.py`); reaper for timeouts. |
| Execution | Idempotency | Orders keyed by HMAC nonce. Outbox table is append-only; reconciliation against the broker is per-day. |

## Observability

`firm/obs/spans.py` defines `agent_span(name, **attrs)` — a thin wrapper over OpenTelemetry. Default exporter is the `JsonlFileExporter` (`firm/obs/tracer.py:60`), which writes line-delimited spans to `data/traces/<date>.jsonl` so a reviewer can `tail -f` the run without standing up an OTLP collector. The OTLP path is wired at `firm/obs/tracer.py:281` (`OTEL_EXPORTER=otlp`) for prod.

Every agent span carries `decision_id`, `parent_decision_id`, and `failure_mode` when applicable, so a single grep over the JSONL reconstructs any heartbeat end-to-end.

## Deterministic replay

The end-to-end test (`tests/integration/test_end_to_end_grounded.py`) and the trading-demo loop test (`tests/integration/test_loop_trading_demo.py`) both work by:

1. Running real ingest into a tmpdir-local Qdrant.
2. Capturing prompt hashes via a stub LLM client.
3. Writing canned responses into `llm_cache` keyed by those hashes — for **every** router-eligible model id, since the hash is model-independent.
4. Running the full pipeline (or the `--loop` subprocess) with `FIRM_LLM_MODE=cached`.
5. Asserting on rows in `decisions`, `outbox`, `positions`.

This is the same pattern the eval harness uses, so the demo path and the eval path share their offline guarantees.

## Improving from past decisions

The firm doesn't update model weights at runtime — but every heartbeat reads what previous heartbeats decided, and several loops are closed automatically:

- **The risk gate stands on prior outcomes.** Current positions are the residue of every past trade. The 10%-per-name and 30%-per-sector limits are checked against those positions every tick, so yesterday's BUY automatically constrains today's sizing.
- **Research recognises familiar tickers.** Before each retrieval, research scans the `decisions` table for any prior trade on the same ticker (`firm/agents/research.py:186`). A novel ticker bumps the router to the stronger model (Sonnet) because there is no prior context to lean on; a familiar one stays on Haiku.
- **End-of-day reconciliation surfaces drift.** The reporter diffs local positions against the broker and writes the result to `reconciliations`. A mismatch is visible the next morning in the dashboard and blocks the next tick until acked.
- **The reversal-rate metric measures real mistakes.** The eval harness asks: of trades that opened in the last 5-day window, what percentage closed at a loss within 3 days? Threshold ≤30%. A rising number is the firm telling on itself.
- **Audit trail is the substrate for prompt iteration.** Every decision, every retrieval hit, every LLM call is on disk (the `decisions` table + the trace JSONL). An operator scanning a week of REFUSE outcomes by `failure_mode` can pinpoint whether the sufficiency judge is too strict, the citations are too weak, or a prompt needs work — then re-record cassettes and re-run the determinism gate.

What the system deliberately does **not** do: re-tune prompts at runtime, run a nightly reflection LLM over its own decisions, or update weights. Those are the next layer of work, called out in [`path-to-production.md`](path-to-production.md).

## Where to look next

- [`architecture.md`](architecture.md) — diagrammatic view of the same flow
- [`eval.md`](eval.md) — what we measure and what we explicitly do not
- [`threat_model.md`](threat_model.md) — STRIDE table + red-team coverage
- [`runbook.md`](runbook.md) — operator procedures for Slack approval, Litestream restore, Qdrant backup
