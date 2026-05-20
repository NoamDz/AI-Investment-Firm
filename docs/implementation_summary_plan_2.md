# Plan 2 — what it is, in plain English

Plan 1 built the skeleton — a paper-trading pipeline with exactly-once order semantics, but the "decide what to buy" logic was a stub that always picked AAPL. **Plan 2 swaps that stub for LLMs grounded in retrieval over real 10-K/10-Q filings**, while keeping every guarantee Plan 1 made (outbox idempotency, audit trail, deterministic replay, HITL gate).

The hard problem isn't "ask Sonnet what to buy." It's: **how do you make sure every number the model emits is grounded in a real document, and how do you replay the entire reasoning chain deterministically tomorrow?** Plan 2's answer has five moving parts.

---

## Idea 1: Citations are a hard contract, not a courtesy

Every `Claim` the research agent produces carries either a `source_chunk_id` (a pointer back into the retrieval results) **or** a `tool_call_id` (a pointer to a deterministic math tool). The Pydantic validator in `firm/core/models.py` enforces this at the type level — a Claim with neither is impossible to construct.

This is enforced **end-to-end**, not after the fact:

1. **The extractor uses the Anthropic Citations API** (`firm/llm/citations.py`). When we send the model the retrieved chunks, each chunk is a `document` content block with `citations: {enabled: true}`. The model's response then comes back as a sequence of `text` blocks, each carrying its own `citations` array pointing to the exact chunk and character span it drew from.
2. **Uncited text is dropped, not flagged-and-kept.** If a `text` block comes back without citations, we discard it and increment `last_uncited_count`. The model never gets a chance to talk freely — every emitted Claim is anchored.
3. **Math is routed to tools, not done by the LLM.** Ratios and risk metrics flow through `FundamentalsTool` / `RiskMetricsTool` (`firm/tools/`), which read from precomputed parquet files. The tool returns a `Decimal`. The Claim that incorporates it carries `tool_call_id`, not `source_chunk_id`. **The LLM never does arithmetic.** This is the spec's "no LLM arithmetic" rule and it's enforced by prompt + tool routing, not post-validation.

The thing the spec is paranoid about is hallucinated financial numbers. The wall is: schema validator → extractor filter → tool routing. Three independent layers that all have to fail for an uncited claim to escape.

---

## Idea 2: Hybrid retrieval with point-in-time filtering

When the research agent asks "what's AAPL doing?", we don't want a single nearest-neighbor lookup. We want:
- **Dense retrieval** (`nomic-embed-text-v1.5`) for semantic similarity — "this chunk is *about* the same thing as the question."
- **Sparse retrieval** (BM25 with a ticker-aware tokenizer) for keyword precision — "the literal word AAPL appears here."
- **Reciprocal Rank Fusion** (RRF, k=60) to combine the two ranked lists.
- **A reranker** (`BAAI/bge-reranker-v2-m3`) over the fused top-50 to keep the best 8.

The fusion + reranker combo matters because Sonnet reads only the top 8 chunks. Dense retrieval alone misses the right document when the question uses a synonym; sparse alone misses semantic paraphrase. RRF picks up the union, the reranker filters out the chaff.

Two non-obvious bits:

**Point-in-time filtering.** The retriever takes `as_of=clock.now()` and passes `published_before=as_of` to Qdrant. This means a backtest of "what would the firm have done on 2024-03-13?" cannot leak information from filings published on 2024-03-14. The retriever raises `ValueError` if `as_of` is naive (no timezone) — there's no ambiguity allowed. There's a dedicated integration test (`tests/integration/test_retrieval_pit.py`) that asserts the filter actually works.

**Contextual augmentation.** Before chunks are embedded, each document is summarized by `claude-haiku-4-5` into a 2-3 sentence doc-level context, and that summary is **prepended** to every chunk from that document at retrieve time. This is Anthropic's "contextual retrieval" trick — small chunks lose context ("the company reported strong Q4..."  — *which* company?), the prepended summary restores it. The summaries are cached in the `llm_cache` table, so re-ingestion costs nothing.

---

## Idea 3: Two-stage grounding — extract, then judge

The research agent doesn't just believe the model's claims. It runs them past a second model.

```
retrieve(question, as_of=now) → 8 chunks
            ↓
extract(query, chunks) via Sonnet + Citations API → list[Claim]
            ↓
judge.assess(question, claims) via Haiku → SufficiencyResult
            ↓
decide: ok → BUY ; partial → ESCALATE ; insufficient → REFUSE
```

The **sufficiency judge** (`firm/grounding/judge.py`) gets the question and every cited Claim and labels each one `SUPPORTED | PARTIAL | UNSUPPORTED`. The aggregate verdict drives the branch in `firm/agents/research.py`:

- All `SUPPORTED` → `BUY` (or `HOLD` if no claims survived).
- Some `PARTIAL` → `ESCALATE` with reason `sufficiency:partial`.
- Mostly `UNSUPPORTED` → `REFUSE` with `failure_mode=INSUFFICIENT_EVIDENCE`.

**Why two models, not one?** Two reasons. First, asking the same model "are you sure?" is theatrical — it's mostly going to agree with itself. Second, by using Haiku (cheaper, smaller) for the gate, we get an independent grader that's less correlated with Sonnet's hallucination modes.

**Failure-mode discipline.** The research agent has five concrete failure modes mapped to `FailureMode` enum values:

| FailureMode | Trigger |
|---|---|
| `INSUFFICIENT_EVIDENCE` | retrieval empty, or judge says nothing is supported |
| `LLM_UNAVAILABLE` | Anthropic API down, transport error, malformed JSON |
| `SCHEMA_VALIDATION_FAILED` | judge response parsed but failed Pydantic validation (T32a) |
| `STALE_DATA` | filing older than `stale_filing_days` (T29a) |
| `UNCITED_CLAIM` | reserved (defensive drop runs at extractor; enum exists for future emit) |

This matters because the operator runbook needs to know **why** a heartbeat refused. "It didn't work" is useless; "judge response failed schema validation at 2026-05-20T14:30Z" tells you where to look.

---

## Idea 4: PM as a vote-of-3 committee, aggregated in Python

The portfolio manager is **not** one LLM call. It's three.

`firm/agents/pm.py` runs three `PmVoter` invocations, one per **lens**:
- **Quality** — does this business have durable fundamentals?
- **Valuation** — is the price reasonable for what you get?
- **Catalyst** — is there a near-term reason the thesis pays off?

Each voter sees the same Claims, gets a lens-specific system prompt, and returns a `PmVote` (BUY/HOLD/SELL, confidence 0-1, rationale, list of cited claim ids). Then `aggregate_votes` — pure deterministic Python — combines them by a 10-case dispatch table:

| Votes | Outcome |
|---|---|
| 3× BUY (unanimous) | BUY, full confidence |
| 2× BUY + 1× HOLD | BUY, confidence × 0.8 (with reservation) |
| 2× BUY + 1× SELL | **ESCALATE** (directional split) |
| 1× BUY + 1× HOLD + 1× SELL | **ESCALATE** (full disagreement) |
| 2× SELL + 1× HOLD | SELL, confidence × 0.8 |
| 2× HOLD + 1× directional | HOLD (majority dominates) |
| ...8 more cases... | ... |

**Why aggregate in Python instead of asking a fourth LLM "what's the consensus?"** Because the tiebreaker has to be auditable and replayable. A Python function on three votes is one diff line away from being explained in court; a fourth LLM call is a black box. The committee LLM-calls are independent (different prompts) and produce evidence; the aggregator picks the answer.

**Chinese wall.** The PM has no access to retrieval or tools. It can only reason over the Claims produced by Research. This is structural — the `make_pm(voter)` factory doesn't take a retriever or a tool list. If a voter tried to look up a number, it couldn't — there's no plumbing.

---

## Idea 5: The LLM cache — what makes the whole thing replayable

Every Anthropic call goes through `CachedAnthropicClient` (`firm/llm/anthropic_client.py`), which keys on `sha256(system + messages + tools)` and writes the full response JSON to the `llm_cache` SQLite table. This sits underneath everything: extractor, judge, voters, the contextual augmenter for ingest.

There are **three modes**, set via `FIRM_LLM_MODE`:

| Mode | Behavior |
|---|---|
| `live` (default) | Real API call; response written to cache. |
| `cached` | Read-only from cache; raises `LlmCacheMissError` on a miss. CI uses this. |
| `record` | Same as live but explicit — used to seed the cache before flipping CI to `cached`. |

This is the trick that makes the demo deterministic. Once you run `make ingest` once and `make demo` once with `FIRM_LLM_MODE=record`, every subsequent `make demo` with `FIRM_LLM_MODE=cached` reads from the same prompt hashes and produces bit-identical output — same Claims, same vote, same trade.

Combined with `ReplayClock(FIRM_REPLAY_AT=...)` from Plan 1, the entire heartbeat — retrieval scores, model outputs, trade ids — is reproducible. **A reviewer can re-run `make demo` next month and get the exact same JSONL report.** That's the property that lets you trust an audit trail.

---

## The supporting cast (briefly)

- **`make ingest`** — One-time pipeline: chunks each FinanceBench doc at 512 tokens / 64 overlap, generates a doc-level summary via Haiku (cached), embeds dense+sparse, upserts into Qdrant collection `firm_chunks`. Idempotent — re-runs skip already-indexed docs. Status persisted in `ingest_runs` SQLite table.

- **`tables_to_prose`** (`firm/rag/preprocess.py`) — HTML tables in 10-Ks are murder for embedders. Each row gets converted to a single English sentence (`"Revenue in fiscal 2023 was $383B; in 2022 it was $394B."`). The original HTML never goes into the embedder.

- **Ticker-aware tokenizer** — BM25's tokenizer treats `AAPL` as a single token, not `a-a-p-l`. Without this, sparse retrieval would miss the most important keyword in every finance query.

- **Boot reconciliation + HITL gate carry over from Plan 1** — Plan 2 didn't touch the outbox, didn't touch reconciliation, didn't touch the LangGraph checkpoint pattern. The grounded research agent **drops in** behind the same `Decision` interface; PM, Risk, HITL, Execution all see the same shape they saw in Plan 1.

- **`escalate_new_ticker: true`** — Flipped back on after T29a. First trade of any ticker the firm hasn't traded before goes to HITL. This is the dogfooding gate: even with perfect retrieval, a brand-new symbol gets a human eye before money moves.

---

## So what is Plan 2, in one paragraph?

Plan 2 turns the dummy research agent into a **two-stage grounded reasoning system** — Sonnet extracts cited Claims over hybrid-retrieved chunks from real 10-K/10-Q filings, Haiku grades each Claim's sufficiency, a vote-of-3 PM committee reasons only over those Claims, and the whole thing is deterministically replayable via a prompt-hash cache. The numbers can't be hallucinated (validator + Citations API + tool routing); the documents can't leak future data (point-in-time filter); the decisions are independently graded (sufficiency judge); the committee can't be a single point of failure (three lenses + deterministic aggregator); and any heartbeat can be re-run tomorrow and get bit-identical output (LLM cache + ReplayClock). Plan 3 adds Slack approvals, real-time replication, and observability; Plan 4 adds the eval harness and CI. The reason this was hard isn't the model calls — it's that **grounding is a property of the pipeline, not the prompt**.
