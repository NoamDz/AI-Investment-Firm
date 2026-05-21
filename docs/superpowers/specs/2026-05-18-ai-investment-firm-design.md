# AI Investment Firm — Design Specification

**Author:** noamdel
**Date:** 2026-05-18
**Status:** Approved (pending user review of this spec)
**Assignment:** Cato Networks — Agentic AI Engineer Home Task

---

## 1. Executive Summary

Build a multi-agent AI investment firm that operates a paper portfolio of US equities through US market hours, grounds decisions in retrieved evidence with citations, routes high-impact trades through human approval, and produces daily reports through two channels. The deliverable must be reproducible end-to-end on a historical replay window, observable per trace, and runnable by a reviewer in under ten minutes.

The target architecture is **B+: MCP-native firm with surgical SQLite durability**. The firm consists of five specialized agents (Research, PM, Risk, Execution, Reporter) plus a Position Monitor node, orchestrated by LangGraph with an SQLite checkpointer, communicating through typed contracts, served by MCP servers for tools. Persistence uses SQLite (WAL + `synchronous=FULL` + Litestream) for transactional state (portfolio, outbox, checkpoints, audit log) and Qdrant for vector retrieval. All five brief-defined bonuses are landed at honest depth.

The firm is deliberately scoped to demonstrate engineering discipline, not market alpha. Beating SPY is explicitly not the goal; reproducible, observable, auditable workflow is.

---

## 2. Goals and Non-Goals

### Goals
- Multi-agent system with ≥4 specialized agents, defensible decomposition, typed I/O contracts, defined failure modes
- Paper portfolio with realistic fills (slippage, commission, market hours) and crash-safe persistent state
- Continuous operation during US market hours with response to scheduled and event-driven triggers
- RAG layer with hybrid retrieval, reranking, point-in-time discipline, citation grounding, and refusal on insufficient evidence
- Human-in-the-loop approval for trades above configurable thresholds, with graph state surviving the wait
- Structured observability traces sufficient to replay any trade end-to-end
- Guardrails: schema validation, hallucination prevention, prompt-injection defenses, hard trading limits
- Eval harness running deterministic historical replay across three regime windows, reporting performance vs benchmark and process discipline
- Daily reports through two channels (Slack real-time + repo-committed Markdown/XLSX bundle)
- Documented path to production with one bonus integration (AWS Bedrock AgentCore for one agent)

### Non-Goals
- Beating the market (brief explicit)
- Real-money trading (paper portfolio only)
- Full cloud deployment (Terraform validated and plan-clean, not applied)
- Measuring investment-quality / alpha attribution (sample size too small)
- General-purpose agentic AI framework (this is a single-firm system)
- Live data feed integration in production sense (replay-first design)

---

## 3. Architecture Overview

### 3.1 The Firm (topology)

```
                    ┌─────────────────────────────────────────┐
                    │           Position Monitor              │
                    │  (heartbeat + event-driven triggers)    │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │   Research     │  ◄── RAG, market data
                            │  (Bull/Bear    │      MCP servers
                            │   internal)    │
                            └────────┬───────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │       PM       │  ◄── Memory: DeskState,
                            │  (vote-of-3,   │      TradeJournal, Playbook
                            │ one bounce ↑↓) │
                            └────────┬───────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │      Risk      │  ◄── Hard limits (Python)
                            │  (deterministic│      Soft policy (LLM)
                            │  + LLM gates)  │
                            └────────┬───────┘
                                     │
                            ┌────────┴───────┐
                            │  HITL Gate     │ ── escalate via signed
                            │  (if >cap)     │    Slack approval
                            └────────┬───────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │   Execution    │  ◄── Broker MCP
                            │ (mostly deter- │      (outbox-protected)
                            │  ministic)     │
                            └────────┬───────┘
                                     │
                                     ▼
                            ┌────────────────┐
                            │    Reporter    │  ──► Slack + Markdown/XLSX
                            │ (daily + HITL- │
                            │ gated reflect.)│
                            └────────────────┘
```

### 3.2 Agent decomposition rationale

Maps to real-fund Chinese-wall structure: Analyst (Research) → Portfolio Manager (PM) → Chief Risk Officer (Risk) → Trader (Execution) → Ops/Compliance (Reporter). Each agent has fiduciary boundaries enforced by least-privilege MCP tool access; Research cannot place orders, Execution cannot mutate research thesis, etc.

### 3.3 Deliberation patterns

Bounded deliberation, not free-form debate:
- **Inside Research:** Bull/Bear sub-agents argue, Research node synthesizes (one cycle)
- **PM → Research:** One-bounce only; PM can request clarification, Research responds once, PM decides
- **Self-consistency on PM:** Vote-of-3 with three parallel PM rationales, majority wins
- **No free-form inter-agent debate.** Roles preserve Chinese walls.

### 3.4 Decision envelope (the universal contract)

Every agent emits a typed `Decision` object:

```python
class Decision(BaseModel):
    id: str                              # ULID
    decision_id_chain: list[str]         # provenance from upstream agents
    action: ActionEnum                   # BUY, SELL, HOLD, ESCALATE, REFUSE
    payload: TypedPayload                # action-specific typed schema
    rationale: str                       # non-empty, structured
    confidence: float                    # 0..1
    citations: list[Citation]            # span-anchored
    falsification_condition: str         # what would invalidate this
    escalation_reason: str | None
    failure_mode: FailureMode | None
    metadata: dict                       # cost, model, timestamp
    nonce: str                           # HMAC(secret, id || timestamp)
```

### 3.5 Failure modes (enumerated)

```python
class FailureMode(StrEnum):
    UNCITED_CLAIM = "uncited_claim"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    HITL_TIMEOUT = "hitl_timeout"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    LLM_UNAVAILABLE = "llm_unavailable"
    STALE_DATA = "stale_data"
    UNGROUNDED_CLAIM = "ungrounded_claim"
    TOOL_PERMISSION_DENIED = "tool_permission_denied"
    UNAPPROVED_HIGH_RISK = "unapproved_high_risk"
    BROKER_UNAVAILABLE = "broker_unavailable"
    RECONCILIATION_DRIFT = "reconciliation_drift"      # added in Plan 3 (T17 EOD reconcile)
    SIGNED_APPROVAL_INVALID = "signed_approval_invalid" # added in Plan 3 (T11 Slack ingress)
    UNKNOWN = "unknown"
```

Every value must be triggered by at least one CI fixture **or** be present in the
`ALLOWED_GAPS` registry with a documented reason (enforced by
`tests/integration/test_failure_mode_coverage.py`).  As of Plan 3, the registry
covers 7 end-to-end triggering fixtures + 7 documented gaps + `UNKNOWN`
sentinel = full enumeration.  Plan 4 promotes the deferred modes (notably
`UNCITED_CLAIM`) from `ALLOWED_GAPS` to first-class fixtures.

### 3.6 Availability model

HA is per-component, not a single property. Take-home stays single-host; the design makes the production path concrete rather than aspirational.

| Component | Tier | Mechanism (take-home) | Production path |
|---|---|---|---|
| Position Monitor, Reporter | Stateless, replicatable | Docker restart policy + healthcheck | Run N replicas behind load balancer |
| Research, PM | Stateless, replicatable | Docker restart policy | Run N replicas; deliberation idempotent within a decision_id |
| Risk, Execution | **Must be singleton** (single-writer to broker) | Single process + restart policy | Leader election (etcd/Consul); broker idempotency keys make failover safe |
| SQLite (firm.db) | Single-writer fundamental | WAL + `synchronous=FULL` + Litestream continuous backup (RPO ~seconds, RTO ~seconds via restore) | Migrate to Postgres via SQLAlchemy `DATABASE_URL` + LangGraph `PostgresSaver` (one-import swap, §5.3) |
| Qdrant | Stateless w.r.t. business state (rebuildable from corpus) | Single container; on loss, re-ingest from source corpus | Qdrant Cloud or replicated cluster |
| MCP servers | Stateless | Restart on crash | Multi-replica; clients reconnect |

**Load-bearing insight:** Execution is fundamentally singleton — only one process can hold authoritative write access to the broker. HA for Execution is fast leader failover with idempotency-key dedupe on retry, **not** horizontal scaling. The outbox pattern (§5.2) is the enabling primitive: without it, hot-standby would double-fire orders; with it, failover is safe by construction.

**What the take-home actually delivers:** crash-safe recovery (§5.5), continuous SQLite backup (§5.6), process-level restart via Docker/systemd. Multi-host HA is explicitly out of scope, but every singleton constraint is named and the migration path for each component is one configuration change away.

### 3.7 Trading limits and risk policy

Hard limits the system cannot exceed, all enforced as **deterministic Python checks** in the Risk agent before any order reaches Execution. The LLM cannot override them.

| Limit | Default | Enforcement |
|---|---|---|
| Max single-position size | 10% NAV | Risk node, hard-fail pre-trade |
| Max sector concentration | 30% NAV | Risk node, hard-fail |
| Max gross exposure | 100% NAV (long-only, no leverage) | Risk node, hard-fail |
| Max single-trade size | 5% NAV | Risk node, hard-fail |
| Max trades per day | 20 | Risk node, hard-fail |
| Min cash buffer | 5% NAV | Risk node, hard-fail |
| Max daily loss (drawdown halt) | −3% NAV | Position Monitor halts new entries; exits-only mode |
| Stale-data refusal | quote > 60s old, OR thesis-source filing > 90 days | Risk node, refuse with `STALE_DATA` |
| HITL escalation threshold | trade > 3% NAV, OR new position in untraded ticker | Routed to HITL gate (not auto-rejected) |

**Hard limits vs. soft policy.** The table above is hard limits — deterministic Python, not LLM judgments. The "soft policy" LLM gate inside the Risk agent reasons about context ("is this thesis still valid given today's news, given today's vol regime?"). Soft policy can downgrade a trade or trigger escalation; it **cannot** lift a hard limit. Limit breach → `FailureMode.RISK_LIMIT_BREACHED`, decision rejected, audit-logged.

**Configurable.** All thresholds live in `config/policy.yaml`, validated by Pydantic on load (rejects malformed configs at startup, not at trade time). Changing a limit is a config change + redeploy, never a code change. Schema:

```yaml
limits:
  max_position_pct: 0.10
  max_sector_pct: 0.30
  max_gross_exposure: 1.00
  max_trade_pct: 0.05
  max_trades_per_day: 20
  min_cash_pct: 0.05
  max_daily_loss_pct: 0.03
  stale_quote_seconds: 60
  stale_filing_days: 90
hitl:
  trade_threshold_pct: 0.03
  escalate_new_ticker: true
```

**CI invariant.** `tests/integration/test_risk_limits.py` asserts: every limit row above has at least one fixture that triggers it and at least one that passes it. Ensures the enforcement code stays live.

### 3.8 Partial failure behavior

**Load-bearing rule: fail-closed.** Every partial failure either refuses the action or escalates to a human. No path silently degrades to "trade anyway with reduced confidence." This is the dominant invariant for any system that places orders.

| Failure | Detection | Behavior | FailureMode |
|---|---|---|---|
| LLM unavailable (all tiers exhausted) | retry ladder exhausted (§10.2) | Abort decision, default to HOLD, escalate | `LLM_UNAVAILABLE` |
| Qdrant down | connection error / timeout > 3s | Fail-closed: retrieval returns empty → sufficiency gate fails → refuse decision, escalate | `INSUFFICIENT_EVIDENCE` |
| Qdrant returns malformed chunks | schema validation on retrieval result | Drop malformed chunks, continue with rest; if too few remain, sufficiency gate fails | `INSUFFICIENT_EVIDENCE` |
| Market-data feed stale | quote timestamp > 60s old (§3.7) | Refuse new entries; exits-only mode until fresh data resumes | `STALE_DATA` |
| Slack down | webhook 5xx or timeout > 5s | Queue HITL approvals locally; on Slack recovery, replay queue. Graph state parked in checkpoint (§5.3). No HITL ack → no high-risk trade fires (fail-closed) | `HITL_TIMEOUT` if queue ages past threshold |
| HITL timeout (no human response) | configurable wait, default 30 min during market hours | Abort the trade, audit-log, do not auto-approve | `HITL_TIMEOUT` |
| Broker API down / timeout | submit timeout > 10s or 5xx | Outbox stays `pending`; retry with same idempotency key (§5.2). After N retries, abort decision and surface unfilled order in EOD report | `BROKER_UNAVAILABLE` |
| Single agent timeout | per-node deadline (default 30s) | Cancel node, emit failure mode, route to escalation. Independent agents continue | `UNKNOWN` (or specific where classifiable) |
| Crash (any host failure) | process restart | Outbox + checkpoint + reconciliation (§5.2, §5.3, §5.7) replay to a consistent state before any new decision | n/a (recovery is automatic) |

**State during partial failure.** All in-flight LangGraph state is checkpointed after every node (§5.3). HITL parks the graph in checkpoint; partial failures of upstream/downstream agents do not lose the decision context. Recovery resumes from the last checkpoint.

**Cross-references.** §5.5 (crash recovery test), §5.7 (boot reconciliation halts on broker drift), §10.2 (LLM fallback ladder), §3.7 (stale-data thresholds).

**CI invariant.** `tests/integration/test_partial_failure.py` injects each failure above and asserts: (a) the correct `FailureMode` is emitted, (b) no order reaches the broker, (c) graph state is recoverable from checkpoint. Wired to the `FailureMode coverage` invariant (§9.5).  As of Plan 3, the invariant is satisfied by a hybrid registry: 7 enum values have end-to-end triggering fixtures, the remaining 7 are listed in `ALLOWED_GAPS` with documented deferral reasons, plus the `UNKNOWN` sentinel.  Plan 4 promotes the deferrals to full fixtures, ending at 14/14 first-class coverage (15 enum values minus the catch-all `UNKNOWN`).

---

## 4. Memory and Caching

### 4.1 Memory taxonomy (LangMem-style)

Four stores, mapped to LangMem terminology:

| Store | LangMem type | Backend | Lifetime | Write path |
|---|---|---|---|---|
| WorkingState | (graph state) | LangGraph SqliteSaver | per-run | automatic |
| DeskState | semantic (facts) | append-only `desk_state` table | session/day | every node tick |
| TradeJournal | episodic | `trades` + `reflections` tables, embedded | permanent | post-trade, HITL-gated |
| Playbook | semantic (rules) | versioned markdown in repo, chunked + embedded | permanent | human PR-reviewed |

**HITL-gated reflections:** After a position closes, PM drafts a retrospective (`thesis, what_worked, what_didnt, lesson, tags`). Draft → Slack review → on approval, persists to TradeJournal and is embedded for future retrieval. Without the gate, hallucinated lessons would become rules.

**Out of scope:** procedural memory artifacts, auto-distillation, graph stores, cross-session user memory.

### 4.2 Caching (two layers only)

| Layer | Purpose |
|---|---|
| Anthropic prompt caching (server-side, 5-min TTL) | System prompt + tool definitions + RAG context cached at 4 breakpoints. 10× cheaper reads. |
| Local LLM response cache (replay mode only) | VCR-style cassettes keyed by `(model, messages_hash, tools_hash)`. Modes: `live`, `record`, `replay`. Enables deterministic CI eval. |

**Explicitly dropped:** retrieval cache, KG cache, embeddings cache, semantic LLM cache. At <50k chunks the marginal value is zero; senior anti-signal is stacking cache layers.

---

## 5. Durability and State

### 5.1 Storage substrate

| Component | Store | Why |
|---|---|---|
| Portfolio state, outbox, checkpoints, audit log | SQLite (WAL + `synchronous=FULL` + Litestream) | Single-writer, low TPS, ACID, audit-friendly |
| Vector store | Qdrant (Docker container) | Purpose-built; sqlite-vss is unmaintained with weak filtering |

**Pragmas (load-bearing):**
```python
db.execute("PRAGMA journal_mode = WAL")
db.execute("PRAGMA synchronous = FULL")  # NOT NORMAL
db.execute("PRAGMA foreign_keys = ON")
```

Litestream provides continuous backup to a local directory for point-in-time restore.

### 5.2 Outbox pattern (broker orders)

Surgical durability for the (DB write + external API call) pair. ~50 LOC.

```python
def place_order(decision: Decision) -> OrderResult:
    idempotency_key = sha256(f"{decision.id}:{decision.nonce}")
    with db.transaction():
        db.execute(
            "INSERT INTO outbox (key, payload, status) VALUES (?, ?, 'pending') "
            "ON CONFLICT (key) DO NOTHING",
            (idempotency_key, decision.to_json())
        )
        existing = db.execute("SELECT status, result FROM outbox WHERE key=?", (idempotency_key,)).fetchone()
        if existing["status"] == "confirmed":
            return OrderResult.from_json(existing["result"])
    result = broker.submit(decision, idempotency_key=idempotency_key)
    db.execute("UPDATE outbox SET status='confirmed', result=? WHERE key=?",
               (result.to_json(), idempotency_key))
    return result
```

**Crash semantics:**
- Crash before outbox insert → no order, no record, clean retry
- Crash between outbox insert and broker call → recovery sees pending, retries with same key, broker idempotency dedupes
- Crash after broker call → recovery sees pending, broker returns existing order, we confirm
- Concurrent attempts → unique key constraint collapses them

### 5.3 LangGraph checkpointer

`SqliteSaver` writes graph state after every node. HITL uses `interrupt_before=["hitl"]`. Graph parks in checkpoint until approval arrives, surviving any crash.

Migration path: `SqliteSaver` → `PostgresSaver` is a one-import swap. Documented in `docs/path-to-production.md`.

### 5.4 Clock injection

`Clock` Protocol with `WallClock` (production) and `ReplayClock` (eval). All time-dependent code accepts an injected clock. CI lint bans `datetime.now()` and `time.time()` in business code.

### 5.5 Crash-recovery test

`tests/integration/test_crash_recovery.py` asserts: kill-mid-order → restart → exactly-once at the broker.

### 5.6 Backup discipline

Litestream continuously replicates `firm.db` to `data/backups/`. Nightly cron runs `sqlite3 firm.db ".backup firm.db.bak"`. Last 7 days retained. Documented in `docs/runbook.md`.

### 5.7 Position reconciliation (broker is source of truth)

Order-level exactly-once (§5.2) prevents duplicate orders but does not by itself prove the firm's *position* view matches the broker's. Reconciliation closes the loop. Broker is the source of truth.

**Boot-time reconciliation (on every startup, before any decision):**

```python
def reconcile_on_boot() -> ReconcileResult:
    broker_positions = broker.list_positions()         # source of truth
    broker_cash = broker.get_cash()
    local_positions = db.load_positions()
    local_cash = db.load_cash()
    diff = compute_diff(broker_positions, local_positions, broker_cash, local_cash)
    audit.log("reconcile.boot", diff=diff)
    if diff.is_empty:
        return ReconcileResult.OK
    return ReconcileResult.MISMATCH  # halts new entries, requires Slack ack
```

**Mismatch protocol:** any non-empty diff → halt new entries (Position Monitor enters "exits-only" mode), push a Slack message to the Risk Committee with the diff, require human ack (reuses the existing HITL gate — no new infrastructure). After ack: local DB is rewritten to match broker view, the ack and resulting writes audit-logged.

**End-of-day reconciliation (in the daily report):**

After market close, the same reconciliation runs once more and the result is rendered into `reports/YYYY-MM-DD/daily_report.md` as a `RECONCILIATION` block:

```
RECONCILIATION (EOD)
  Broker positions:   { AAPL: 100, NVDA: 50, MSFT: 200 }
  Local positions:    { AAPL: 100, NVDA: 50, MSFT: 200 }
  Position diff:      none
  Broker cash:        $94,250.00
  Local cash:         $94,250.00
  Cash diff:          $0.00
  Status:             ✓ books tie to broker
```

Non-empty diff → block renders red with the discrepancy listed and the audit log entry referenced.

**Why this is in the report, not just the log.** The brief asks for an audit log of every decision (§1). Position reconciliation is the integrity check that the decision audit log corresponds to broker reality. Putting it in the committed daily report makes it a durable, reviewer-facing artifact, not an operator-only log line.

**CI invariant.** `tests/integration/test_reconciliation.py` simulates three drift scenarios (crash-induced stale local DB, broker-side mutation, clean match) and asserts the boot protocol halts/acks/proceeds correctly. The EOD report rendering is checked against a golden file.

---

## 6. Retrieval Pipeline (RAG)

### 6.1 Pipeline (4 stages + finance-specific ingest)

```
query → [1] hybrid retrieve (dense + sparse via Qdrant)
      → [2] PIT filter (published_at <= as_of)
      → [3] rerank (local bge-reranker-v2-m3)
      → [4] contextual pack
      → LLM
```

| Stage | Mechanism | Source |
|---|---|---|
| Hybrid | Qdrant named vectors: dense embeddings + sparse BM25-style | Native Qdrant hybrid |
| PIT filter | Payload filter `published_at <= as_of` at index time | Qdrant payload filter |
| Rerank | bge-reranker-v2-m3 on top-50 → top-8 | Local, free, ~50ms GPU |
| Contextual pack | Per-chunk pre-computed doc summary prefix | Anthropic Contextual Retrieval pattern, generated at ingest |

**Dropped:** HyDE query rewrite (gains compress to zero with strong reranker; saves Haiku call and ~500ms p50).

### 6.2 Finance-specific ingest

1. **Table → text preprocessing.** SEC filings are table-heavy. At ingest, detect tables and convert to prose narratives (e.g., *"In Q3 2024, NVDA reported total revenue of $18.1B, up 24.9% YoY..."*). Both versions stored: prose feeds retrieval, original HTML preserved for citation display.
2. **Ticker-aware tokenizer.** Qdrant sparse index configured to preserve patterns `\$[A-Z]+`, `[A-Z]+\.[A-Z]`, `\d+-[A-Z]` as single tokens so `BRK.B`, `$AAPL`, `10-K` survive tokenization.

### 6.3 Corpus

| Source | Coverage | Cadence |
|---|---|---|
| SEC 10-K, 10-Q, 8-K (EDGAR) | 30 tickers × last 5 years | Daily poll for new filings |
| Earnings call transcripts | 30 tickers × last 5 years | Per call |
| Curated news (Polygon or NewsAPI) | 30 tickers, rolling 12 months | Hourly |
| Internal Playbook | 5–10 hand-written trading rules, versioned in repo | On PR merge |

**Ingest discipline:** `published_at` required at ingest; chunks without it rejected. CI test asserts no NULL `published_at` in any chunk.

### 6.4 PIT enforcement

Index-level filter, not post-hoc. CI test: no future-dated chunk ever appears in any retrieval result, asserted across all eval queries.

**Known limitation (documented):** forward references inside otherwise-valid chunks ("as we'll see in Q4...") cannot be filtered automatically. Stated in `docs/eval.md`.

---

## 7. Citations and Grounding

### 7.1 Anthropic Citations API

All retrieval-grounded claims use the Citations API. Retrieved chunks formatted as `document` content blocks with `citations: {enabled: true}`. Server enforces span-anchoring; the model cannot output a claim without pointing to a chunk span.

Vendor lock contained behind one adapter:
```python
class CitedClaimExtractor(Protocol):
    def extract(self, query: str, chunks: list[Chunk]) -> list[Claim]: ...

class AnthropicCitationsExtractor: ...  # production
class GenericExtractor: ...              # JSON-prompted fallback for other providers
```

### 7.2 Claim schema

```python
class Claim(BaseModel):
    text: str
    value: Decimal | None
    unit: str | None
    source_chunk_id: str | None       # set if from a citation
    source_span: tuple[int, int] | None
    tool_call_id: str | None          # set if from a computed tool result
```

Every numeric or factual claim must have `source_chunk_id` OR `tool_call_id`. Schema validator enforces. Missing provenance → `FailureMode.UNCITED_CLAIM`.

### 7.3 Ban LLM arithmetic

The LLM cannot compute numbers. Any computed metric (P/E, growth %, ratios, vol-target sizing) must come from a tool call. Two new MCP tools:

- `fundamentals.get_ratio(ticker, ratio_name, as_of) -> Decimal`
- `risk.get_metric(ticker, metric, window) -> Decimal`

Position math (% NAV, P&L) is done in Python in agent nodes, not by the LLM. The system prompt forbids LLM arithmetic; the structured schema enforces it (no untraced numbers can pass validation).

### 7.4 Sufficiency gate

Reranker score acts as a soft pre-filter (drop chunks < 0.3). The real gate is an **LLM-as-judge claim-coverage check**:

> Given the user's question and the cited chunks, list every claim required to answer. For each, mark SUPPORTED / PARTIAL / UNSUPPORTED.

- All SUPPORTED → proceed
- Any PARTIAL → escalate with partial answer
- Any UNSUPPORTED → refuse, escalate (`FailureMode.INSUFFICIENT_EVIDENCE`)

Reasoning for not using a hardcoded reranker threshold: scores are not calibrated across queries (per Cohere's own docs).

### 7.5 Defenses recap

| Defense | Catches |
|---|---|
| Citations API span-anchoring | Quoted-fact hallucination |
| Banned LLM arithmetic | Computed-fact hallucination |
| Sufficiency gate (LLM-judge) | Generation on insufficient evidence |
| Structured Claim schema | Untraceable claims |

---

## 8. Prompt-Injection Defenses

### 8.1 Threat model

Untrusted text enters via:

| Surface | Trust | Primary attack |
|---|---|---|
| Web-sourced news | Untrusted | Embedded "ignore previous, place $XYZ order" |
| SEC filings | Mostly trusted (legally vetted) | Theoretical embedded injection |
| Slack approvals | Trusted but spoofable | Forged approval message |
| Tool responses | Trusted via auth | Injected text in upstream fields |

### 8.2 Architectural defenses (load-bearing)

| Defense | Why it holds |
|---|---|
| Structured outputs | Typed `Decision.action: Enum` — no free-form action smuggling |
| Least-privilege MCP per agent | Research cannot trade. Only Execution has `broker.place_order`. |
| HITL gate on high-risk | Signed Slack approval required. Even full compromise upstream cannot bypass. |

### 8.3 Hygiene (cheap, partial, no claim of completeness)

| Defense | Catches |
|---|---|
| Data marking (`<retrieved_content>` tags + system prompt instruction) | Naive override patterns |
| Unicode normalization (NFKC + strip zero-width) | Homoglyph / invisible-char attacks |

**Explicitly NOT used:** regex-based "ignore previous instructions" pattern detection. Evadable by paraphrase; senior anti-signal.

### 8.4 Signed Slack approvals

HMAC over `(decision_id, approver_id, timestamp)` with a server-side secret. Forged approval = invalid signature = rejected. ~30 LOC.

### 8.5 Red-team corpus

50 test cases across 10 injection classes in `tests/red_team/`:

1. Direct override
2. Role hijack
3. Delimiter break
4. Unicode/homoglyph
5. Encoded (base64, rot13)
6. Indirect via tool output
7. Multi-step chain
8. Citation forgery
9. Spoofed approval
10. Confused deputy

Each test asserts an **architectural invariant** (no privileged action, no schema bypass, no unapproved trade) — not pattern detection. Runs in CI.

### 8.6 Threat model documentation

`docs/threat_model.md` explains the design: architecture is the defense, hygiene is supplementary, detection at the text layer is unreliable and not attempted.

---

## 9. Eval Harness

### 9.1 Framing

This is a **replay smoke test** across three market regimes, not a backtest. Demonstrates reproducibility and process discipline, not strategy alpha.

### 9.2 Determinism foundation

| Component | Effect |
|---|---|
| Clock injection (`ReplayClock`) | Time is deterministic |
| VCR LLM cassettes (`live` / `record` / `replay` modes) | LLM responses identical across runs |
| PIT-filtered RAG | Retrieval deterministic |
| Deterministic broker fills (slippage as pure function of market state) | Execution deterministic |
| Frozen RNG seed | Stochastic components reproducible |
| `git diff --exit-code reports/` in CI | Determinism gate |

### 9.3 Three regime windows (declared upfront)

| Regime | Window | Character |
|---|---|---|
| 1 | 2024-03-11 to 2024-03-15 | Earnings-heavy (NVDA, ORCL, ADBE) |
| 2 | 2024-08-05 to 2024-08-09 | Drawdown (post-Aug-5 sell-off) |
| 3 | 2023-11-06 to 2023-11-10 | Low-volatility quiet |

Windows declared at spec time; no agent prompts or parameters tuned against them.

### 9.4 Performance metrics

**SPY is the primary benchmark** (per brief). We additionally report an **equal-weight basket of our 30-ticker universe** as a secondary comparison.

**Why both:** SPY is the S&P 500 (500 names, market-cap-weighted). Our firm picks from a 30-ticker universe. A direct firm-vs-SPY comparison therefore conflates two effects:

1. **Stock-picking skill** within our universe — what the firm is meant to demonstrate.
2. **Universe selection** (our 30 tickers vs SPY's 500) — not a skill claim.

The equal-weight basket answers *"what would equal-weight buy-and-hold of these 30 tickers have returned?"* — an apples-to-apples baseline for the firm's stock-picking and timing. Beating SPY but losing to the basket means the firm got lucky on universe choice, not skill. Beating the basket is the harder, more honest test.

```
PER-TRADE RETURNS (%)
  Trade 1: +2.8%, Trade 2: -1.4%, ...
  Median, best, worst.

HIT RATE
  X/N — n is small, not statistically significant.
  Reported for transparency.

TOTAL RETURN
  Firm vs SPY (primary, per brief) vs equal-weight basket (secondary), per regime and aggregate.
```

No Sharpe (meaningless at this N).

### 9.5 Process metrics (all mechanical except sufficiency gate)

| Metric | Mechanism |
|---|---|
| Groundedness | Schema check on `Claim` provenance — % with `source_chunk_id` or `tool_call_id` |
| Decision discipline | Schema check on `Decision` (rationale, ≥2 citations, falsification_condition non-empty) |
| Citation diversity | Distinct source_id count per decision |
| Reversal rate | % positions closed at loss within 3 days |
| Risk-policy compliance | Audit-log invariant (any policy violation that reached broker?) |
| HITL correctness | Trades > threshold paused with valid HMAC approval |
| Schema rejections | Count of decisions rejected by validator |
| Red-team pass | Architectural-invariant assertions on 50 injection tests |
| Sufficiency gate | Precision/recall on 30-query labeled dev set |
| FailureMode coverage | Every enum value triggered by ≥1 fixture |

### 9.6 "Not Measured" section (mandatory in report)

- Investment quality / alpha — N too small
- Generalization beyond 3 declared regimes
- Real-world fill quality — paper sim
- Forward references inside chunks — known RAG limitation
- Long-horizon learning effects — journal too sparse

### 9.7 Sample report shape

```
EVAL REPORT — Replay smoke test across 3 regimes

REGIME 1: Mar 11–15, 2024 (earnings-heavy)
  Total return:           -1.2%
  vs SPY (primary):       -2.0pp (SPY: +0.8%)
  vs equal-weight basket: -0.8pp (basket: -0.4%)
  Per-trade returns:      +2.8%, -1.4%, +0.6%, -0.9%, -2.1%
  Hit rate:               2/5 (40%) — n=5, not statistically significant

[Regimes 2 and 3 same shape]

PROCESS METRICS (aggregated)
  Groundedness:                  99.5%
  Decision discipline:           15/15
  Red-team pass:                 50/50
  Privileged-action attempts:    0
  HITL correctness:              12/12
  FailureMode coverage:          14/14 fixtures (ALLOWED_GAPS empty; UNKNOWN catch-all sentinel)

NOT MEASURED
  [list]
```

### 9.8 Sample run artifact

`sample_runs/2024-03-13/` contains:
- `daily_report.md` + `positions.xlsx` (two-channel deliverable)
- `trace.jsonl` (full OpenTelemetry trace)
- `decisions.jsonl` (append-only decision log)
- `cassettes/` (LLM cassettes for reproducibility)

Reviewer clones, runs `make replay`, sees one trading day reproduce identically.

**Runtime vs. committed paths.** `sample_runs/<date>/` above is the *committed*
reference artifact (one historical day, frozen, tracked in git). The *runtime*
output path that the running firm writes to on every heartbeat is
`data/reports/<date>/decisions.jsonl` — calendar-date-keyed, append-only,
not committed. The two are intentionally distinct: one is a reproducibility
snapshot, the other is rolling production output. Trace JSONL similarly
splits: committed sample at `sample_runs/<date>/trace.jsonl`, runtime at
`data/traces/<date>/run-<id>.jsonl`.

### 9.9 Inspect AI reference

`docs/eval.md` references Inspect AI (UK AISI) as the framework we would adopt at production scale, with rationale for why we used a custom pytest harness for the take-home scope.

---

## 10. Observability, Cost Routing, Output Channels

### 10.1 Observability

OpenTelemetry SDK. Every agent node and tool call emits one span with:

```python
{
    "trace_id", "span_id", "parent_span_id",
    "agent", "operation", "decision_id",
    "duration_ms", "model",
    "input_tokens", "output_tokens", "cached_tokens",
    "cost_usd",
    "citations", "failure_mode", "status",
}
```

Local exporter writes JSONL to `traces/YYYY-MM-DD/run-<id>.jsonl`. Production path: swap exporter to OTLP collector. One sample trace committed to `sample_runs/2024-03-13/trace.jsonl` for reviewer end-to-end replay.

### 10.2 Cost routing

`RouterFeatures` Pydantic model scores each decision on `(risk_weight, novelty, complexity, time_pressure)`. Routing matrix in `config/router.yaml`:

| Profile | Model |
|---|---|
| Low-risk repetitive (intraday rebalance check) | Haiku |
| Standard (most PM proposals) | Sonnet (default) |
| High-risk or first-of-kind (large position, illiquid ticker) | Opus |

**Fallback ladder:**
1. Sonnet failure → truncate retrieved chunks, retry
2. Still failing → downgrade to Haiku with reduced scope
3. Haiku failure → abort, return `FailureMode.LLM_UNAVAILABLE` with conservative default (HOLD, escalate)

Daily cost report:
```
COST SUMMARY (2024-03-13)
  Haiku:   47 × avg $0.0003 = $0.014
  Sonnet:  19 × avg $0.0021 = $0.040
  Opus:     2 × avg $0.0180 = $0.036
  Cached:  67%
  Total:   $0.090
```

### 10.3 Output channels

| Channel | Purpose | Audience | Format |
|---|---|---|---|
| Slack | Real-time operational — HITL approvals, intra-day alerts, EOD summary | Trader / on-call | Threaded messages with signed approval buttons |
| Repo-committed bundle at `reports/YYYY-MM-DD/` | Durable, versioned, auditable | Auditor / reviewer | Markdown narrative + XLSX positions/P&L + JSONL decisions + JSONL trace |

**Justification (different time horizons, different audiences, no new operational surface):** Slack is real-time and reuses HITL infrastructure; repo bundle is durable and reuses the eval-report pattern. Email and web dashboard rejected as adding new ops surfaces without orthogonal value.

---

## 11. Bonuses: AWS Bedrock AgentCore + IaC/CI-CD

Three other bonuses are landed in earlier sections:
- Advanced RAG → §6
- Cost-aware model routing → §10.2
- Documented prompt-injection defenses → §8

### 11.1 AWS Bedrock AgentCore

`docs/agentcore_mapping.md` documents the architecture mapping:

| Our component | AgentCore primitive | Migration |
|---|---|---|
| LangGraph orchestrator | AgentCore Runtime | Adapter, ~50 LOC |
| MCP servers | AgentCore Gateway (MCP-native) | Direct |
| DeskState / TradeJournal | AgentCore Memory | Schema mapping, ~50 LOC |
| OpenTelemetry traces | AgentCore Observability | Exporter swap |
| HITL signed approvals | AgentCore Identity | API binding, ~30 LOC |
| SqliteSaver | AgentCore Runtime checkpointer | Adapter |

**Concrete deliverable:** Reporter agent runs on AgentCore's local runtime (simplest agent, no broker, no state mutation). Other agents remain LangGraph for the demo but are AgentCore-ready by interface.

### 11.2 Terraform (IaC)

`infra/terraform/`:
```
main.tf, variables.tf
modules/
  network/   (VPC, subnets, SGs)
  compute/   (ECS Fargate task + service)
  storage/   (RDS Postgres + S3 reports/traces)
  secrets/   (Secrets Manager)
  bedrock/   (AgentCore runtime config)
  observability/ (CloudWatch + OTLP collector)
envs/
  dev.tfvars, prod.tfvars
```

Validated and plan-clean. **Not applied** — `terraform plan` output committed at `infra/terraform/PLAN.md`. `make deploy-dev` available for reviewer.

### 11.3 GitHub Actions (CI/CD)

`.github/workflows/`:

| Workflow | Trigger | Steps |
|---|---|---|
| `pr.yml` | every PR | ruff, mypy, unit tests, integration tests, deterministic eval (1 regime), `terraform validate`, docker build (no push), `git diff --exit-code reports/` |
| `main.yml` | merge to main | all of pr.yml + 3-regime eval + docker build & push to GHCR + terraform plan |
| `release.yml` | tag `v*` | all of main.yml + release artifact + eval report for release notes |

README displays last green build badges. No auto-deploy step — `make deploy-dev` is human-gated.

---

## 12. Repo Layout

```
ai-investment-firm/
├── firm/                       # source
│   ├── orchestrator/           # LangGraph workflow
│   ├── agents/                 # research, pm, risk, execution, reporter
│   ├── mcp_servers/            # broker, market_data, research, slack, fundamentals, risk_metrics
│   ├── memory/                 # desk_state, trade_journal, playbook
│   ├── rag/                    # ingest, retrieval, ranking
│   ├── grounding/              # claim schema, citations adapter
│   ├── policy/                 # hard limits, soft policy
│   └── observability/          # OTel setup
├── data/
│   ├── firm.db                 # SQLite, gitignored
│   ├── qdrant/                 # Qdrant volume, gitignored
│   └── cassettes/              # LLM cassettes, committed
├── config/
│   ├── policy.yaml             # risk limits, HITL thresholds
│   ├── router.yaml             # cost routing matrix
│   └── universe.yaml           # frozen 30-ticker universe as_of
├── tests/
│   ├── unit/
│   ├── integration/            # crash recovery, outbox, HITL
│   ├── eval/                   # replay smoke test
│   ├── red_team/               # 50 injection tests, public corpus
│   └── point_in_time/          # PIT filter checks
├── sample_runs/2024-03-13/     # full sample day artifact
├── reports/                    # eval + daily reports
├── docs/
│   ├── README.md
│   ├── architecture.md
│   ├── threat_model.md
│   ├── eval.md
│   ├── runbook.md
│   ├── path-to-production.md
│   └── agentcore_mapping.md
├── infra/terraform/
├── .github/workflows/
├── Dockerfile
├── docker-compose.yml          # local: firm + qdrant
├── Makefile                    # demo, eval, deploy targets
└── pyproject.toml
```

---

## 13. Deliverables Map (back to brief)

| Brief deliverable | Where |
|---|---|
| A. Git repository, runnable, <10min clone-to-demo | Whole repo + `make demo` |
| B. Architecture diagram (logical + deployment view) | `docs/architecture.md` |
| C. README + technical overview + runbook + eval report | `docs/` |
| D. Sample run committed (1+ historical day) | `sample_runs/2024-03-13/` |
| Output channels (≥2 + justify) | Slack + repo bundle, §10.3 |
| Multi-agent (≥4) | 5 agents + Position Monitor, §3 |
| RAG + citations | §6, §7 |
| HITL on high-risk | §3 + §8.4 + §10.3 |
| Observability | §10.1 |
| Guardrails | §7, §8, §3.5 |
| Eval harness in CI | §9 + §11.3 |
| Bonus: managed agent runtime | AgentCore mapping + Reporter on AgentCore, §11.1 |
| Bonus: IaC + CI/CD | §11.2 + §11.3 |
| Bonus: advanced RAG | §6 |
| Bonus: cost-aware routing | §10.2 |
| Bonus: prompt-injection docs | §8 |

---

## 14. Open Questions and Known Risks

### Open
- **Broker choice:** Alpaca paper trading API (free, market hours simulation) vs custom paper sim (full control, no rate limits). Lean: Alpaca for realism, but verify the API supports the historical replay timestamps we need.
- **Universe construction:** frozen 30 tickers as of 2023-11-01 — list to be defined in `config/universe.yaml` at implementation kickoff.
- **Playbook seed corpus:** 5–10 rules to be hand-written before first eval run.

### Known risks
- Forward references inside otherwise-valid chunks defeat PIT filter (documented limitation)
- N=5 per regime is statistically weak; offset by 3 regimes and process metrics
- AgentCore deployment of a single agent depends on AgentCore SDK stability; fallback is full-local demo with mapping doc only
- Citations API is Anthropic-only — vendor lock contained behind one adapter

---

## 15. References

- Anthropic Contextual Retrieval — https://www.anthropic.com/news/contextual-retrieval
- Anthropic Citations API — https://docs.anthropic.com/en/docs/build-with-claude/citations
- LangGraph PostgresSaver migration — https://blog.langchain.com/langgraph-v0-2/
- LangMem memory taxonomy — https://github.com/langchain-ai/langmem
- FinMem (trading agent memory) — arXiv 2311.13743
- CRAG (Corrective RAG) — arXiv 2401.15884
- Cohere Rerank Best Practices — https://docs.cohere.com/docs/reranking-best-practices
- Inspect AI (UK AISI) — https://inspect.aisi.org.uk/
- SQLite WAL durability — Andrew Ayer, https://www.agwa.name/blog/post/sqlite_durability
- Fly.io Litestream — https://fly.io/blog/introducing-litefs/
- VCR-langchain — https://github.com/amosjyng/vcr-langchain

---

**End of design specification.**
