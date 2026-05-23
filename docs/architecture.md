# Architecture

Two views: **logical** (the one heartbeat the LangGraph orchestrates) and **deployment** (the host + container topology the demo runs on).

## Logical — one heartbeat through seven nodes

```mermaid
flowchart LR
    M(monitor) --> R(research)
    R --> P(pm)
    P --> RK(risk)
    RK -- ESCALATE --> H(hitl)
    RK -- ok --> X(execution)
    H --> X
    X --> RP(reporter)
    RP --> END((END))

    subgraph WS [WorkingState — TypedDict checkpointed per heartbeat]
      direction TB
      WS1[heartbeat_at]
      WS2[research_decision · claims · retrieved_chunks · sufficiency_*]
      WS3[pm_decision · pm_votes]
      WS4[risk_decision · hitl_required]
      WS5[execution_result · report_path]
    end
```

`firm/orchestrator/graph.py:35` wires the edges. Conditional after risk: `route_after_risk` reads `risk_decision.action`; `ESCALATE` routes to `hitl` (interrupted by `interrupt_before=["hitl"]`) and waits for `make dev-ack` or a Slack approval. All other actions fall straight to `execution`.

State is persisted by a `SqliteSaver` (LangGraph checkpointer) into `firm.db` under thread id `heartbeat_at`. An interrupt resumes from the saved checkpoint on the next `--loop` tick rather than restarting the heartbeat.

### Where the safety nets sit

| Node | Defends against | Mechanism |
|------|-----------------|-----------|
| `research` | Hallucinated evidence | Anthropic Citations API → Qdrant (BM25 + Nomic + BGE rerank) → Haiku sufficiency judge re-reads the same passages; rejects with typed `INSUFFICIENT_EVIDENCE` / `UNCITED_CLAIM` |
| `pm` | Single-LLM groupthink | LangGraph `Send` fans out **three lenses in parallel** (quality / valuation / catalyst); majority vote required |
| `risk` | Silent policy bypass | Deterministic gates — gross & net exposure, per-name & per-sector caps, drawdown, quote-age. ESCALATE → HITL |
| `hitl` | Unsigned approvals | HMAC-SHA256 dual-key rotation (`firm/hitl/signing.py`); stale items reaped as `UNAPPROVED_HIGH_RISK` |
| `execution` | Duplicate fills | Idempotency keyed by HMAC nonce; chained `parent_id` in audit log |
| `reporter` | Drift between channels | Reads `firm.db` only; both dashboard and xlsx pull from the same source |

## Deployment — host + Docker

```mermaid
flowchart TB
    subgraph HOST [Host]
      INGEST([python -m firm.cli ingest<br/>GPU embed, one-time])
      LOOP([python -m firm.cli run --loop<br/>or `docker compose up firm`])
      DASH([streamlit run firm/dashboard.py])
      MAKE([make report DATE=YYYY-MM-DD<br/>writes positions.xlsx])
    end

    subgraph DOCKER [docker compose]
      QD[(Qdrant 1.11<br/>vector store)]
      FIRM[firm container<br/>LangGraph heartbeat]
    end

    subgraph STATE [Shared state — host bind-mounted]
      DB[(firm.db<br/>SQLite — decisions, outbox,<br/>positions, hitl_queue,<br/>cost_ledger, audit_log)]
      REP[(data/reports/<DATE>/<br/>daily_report.md<br/>positions.xlsx)]
    end

    subgraph EXT [External APIs]
      ANT[Anthropic<br/>Citations + LLM]
      ALP[Alpaca<br/>paper broker — optional]
      SLK[Slack<br/>HITL approvals — optional]
    end

    INGEST -- chunks --> QD
    INGEST -- ingest_runs --> DB
    LOOP --> FIRM
    FIRM -- retrieve --> QD
    FIRM -- read/write --> DB
    FIRM -- writes --> REP
    FIRM -- LLM calls<br/>(cost router + cache) --> ANT
    FIRM -- orders --> ALP
    FIRM -- escalations --> SLK
    SLK -- signed approvals --> FIRM
    DASH -- read-only --> DB
    DASH -- download --> REP
    MAKE -- writes --> REP
```

### Why this split

- **GPU ingest stays on the host.** Embedding 84 10-Ks with Nomic + BGE rerank wants CUDA; the firm runtime itself is CPU-bound and runs anywhere Docker does.
- **Qdrant in Docker, pinned to 1.11.** `qdrant-client` is bounded to `>=1.11,<1.13` in `pyproject.toml` so the wire format never drifts under us.
- **`firm.db` is the single source of truth.** The dashboard and the xlsx report read from it; they cannot disagree.
- **External APIs are optional.** Alpaca is gated by `FIRM_BROKER=ALPACA` (default `FAKE`); Slack by `FIRM_SLACK_BOT_TOKEN`. `FIRM_LLM_MODE=cached` removes the Anthropic dependency entirely (eval + red-team run offline).

## Where to look next

- [`technical-overview.md`](technical-overview.md) — agent contracts, state-key lifecycle, partial-failure model
- [`agentcore_mapping.md`](agentcore_mapping.md) — how each box above maps to a Bedrock AgentCore primitive
- [`path-to-production.md`](path-to-production.md) — what changes in this diagram when going to prod
