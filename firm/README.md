# `firm/` — module map

The firm package is a multi-agent paper-trading runtime. A single CLI
(`python -m firm.cli`) drives heartbeats through a LangGraph orchestrator.

| Module | Role |
|--------|------|
| `cli.py` | Click entry point — `run`, `ingest`, `report`, `ack`, `reconcile`, `eval`, `red-team`, `doctor` |
| `orchestrator/` | LangGraph state machine; SqliteSaver checkpointer; `interrupt_before=["hitl"]` |
| `agents/` | Per-node implementations — `monitor`, `research`, `pm`, `risk`, `hitl`, `execution`, `reporter` |
| `core/` | Domain models (`Claim`, `Decision`, `FailureMode`, …) + `ReplayClock` for deterministic time |
| `rag/` | FinanceBench ingest, chunking, BGE-rerank, Qdrant client (embedded or remote) |
| `grounding/` | Citations API extractor + Haiku sufficiency judge + claim schema |
| `llm/` | Anthropic client wrapper, cost router, prompt templates, LlmCache for replay |
| `tools/` | Deterministic tool functions exposed to the extractor (`fundamentals_get_ratio`, `risk_get_metric`) |
| `hitl/` | Pre-approve path + reaper (ages out pending rows → UNAPPROVED_HIGH_RISK REFUSE) |
| `risk/` | Policy engine — position limits, daily-PnL stop, exposure caps |
| `broker/` | Pluggable broker (`FAKE` simulator + `ALPACA` paper-trading) |
| `outbox/` | At-least-once event delivery with idempotency keys |
| `reports/` | Daily markdown report + xlsx positions sheet + EOD reconcile |
| `reconcile/` | Books-to-broker reconciliation (positions + cash diff) |
| `audit/` | Append-only audit log with chained `parent_chain` IDs |
| `obs/` | Structured logs, cost ledger, trace export |
| `eval/` | 3-regime eval harness + benchmark scoring (SPY + basket) |
| `agentcore/` | AWS Bedrock AgentCore Reporter adapter (opt-in via `[agentcore]` extra) |
| `db/` | SQLite connection + schema migrations |
| `ops/` | One-off operational scripts (precompute fundamentals/risk metrics) |

See the per-plan implementation summaries in [`docs/`](../docs/README.md) for
how the modules came together.
