# Plan 3: HITL Approvals + Reports + Observability + Cost Routing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal.** Close the operator-facing gaps between Plan 2's grounded pipeline and a system a human can actually run, audit, and pay for in production. At the end of Plan 3, a heartbeat decision flows over Slack with signed HMAC approvals (no CLI ack stand-in), every LLM/tool/retrieval call emits an OTel span with token+cost+citation metadata to `traces/YYYY-MM-DD/run-<id>.jsonl`, the cost router picks a model tier per `RouterFeatures` and gracefully falls back on failure, `reports/YYYY-MM-DD/` is committed at EOD with a Markdown narrative + XLSX positions/P&L + decisions JSONL + a `RECONCILIATION` block that asserts books-tie-to-broker, and Litestream is running. Transcripts + news `CorpusSource` adapters land alongside the existing FinanceBench source so the corpus matches spec §6.3.

**Architecture.** Five orthogonal slices, each gated by a CI invariant from Plan 2's failure-mode discipline:

1. **HITL transport switch** (CLI → Slack). Same `hitl_queue` state machine, new ingress: signed Slack interactive button → server-side HMAC verifies `(decision_id, approver_id, ts)` → existing `ack(decision_id, approver_id)` writes to `audit_log` and the LangGraph wakes from its checkpoint. The CLI ack remains as a developer fallback under `--dev-ack`.
2. **OTel JSONL spans.** OpenTelemetry SDK with a local file exporter writes one span per agent node + per LLM/tool/retrieval call to `traces/YYYY-MM-DD/run-<id>.jsonl`. Replaces the ad-hoc Plan 2 logging.
3. **Cost router + fallback ladder.** `RouterFeatures` Pydantic model scores each decision; `config/router.yaml` maps profile → model. Wraps the existing `CachedAnthropicClient` so cache semantics are preserved. On Sonnet failure: truncate chunks → retry; still failing: downgrade to Haiku with reduced scope; on Haiku failure: REFUSE with `LLM_UNAVAILABLE`.
4. **Daily reports + EOD reconcile.** `firm.reports.daily` renders Markdown + XLSX + decisions JSONL into `reports/YYYY-MM-DD/`, including a `RECONCILIATION (EOD)` block that re-runs the boot reconcile and shows the diff against the broker. CI checks golden output and asserts non-empty diff renders red.
5. **Corpus adapters.** Earnings-transcript adapter (parameterised over a local JSONL dump; same `published_at` discipline) + news adapter (Polygon or NewsAPI; degrades to no-op on missing creds). Litestream finally runs as a sibling container.

```
                 Slack (interactive button, signed)
                              │
                              ▼
   HITL gate ── verify HMAC ── ack(decision_id, approver_id)
                              │
                              ▼
   LangGraph resumes from checkpoint ─── Risk-approved Decision ─── Execution

   Every LLM/tool/retrieval call ──► OTel span ──► traces/YYYY-MM-DD/run-<id>.jsonl

   Decision cost ▶ RouterFeatures ▶ config/router.yaml ▶ {Haiku|Sonnet|Opus}
                                                            └─► fallback ladder

   Market close ─► daily_report.md + positions.xlsx + decisions.jsonl + RECONCILIATION block
```

**Tech Stack (additions vs Plan 2).**
- `opentelemetry-sdk>=1.27` + `opentelemetry-api>=1.27` — span instrumentation
- `slack-sdk>=3.31` — Slack Web API + signature verification
- `openpyxl>=3.1` — XLSX writer for daily report
- `jinja2>=3.1` — Markdown template rendering
- `litestream` — sibling container (binary, no Python dep) added to `docker-compose.yml`
- Optional `polygon-api-client>=1.14` OR `newsapi-python>=0.2.7` — news adapter (gated by env vars)

**Out of scope (deferred to Plan 4):** the FinanceBench-Q&A eval harness, red-team corpus, GitHub Actions CI, Terraform/AgentCore deployment, and the final 14/14 FailureMode CI invariant (15 enum values minus the `UNKNOWN` catch-all, all with end-to-end triggering fixtures). None of those depend on this plan's deliverables, and bundling them in here would push the implementation surface beyond what one reviewer can sanity-check in a sitting.

---

## File Structure

New and modified files relative to Plan 2. Unchanged files are not listed.

```
ai-investment-firm/
├── pyproject.toml                          # MODIFIED: add opentelemetry, slack-sdk, openpyxl, jinja2, polygon-api-client
├── docker-compose.yml                      # MODIFIED: add litestream sibling, mount traces volume, SLACK_* + LITESTREAM_* env
├── Makefile                                # MODIFIED: add `report` target
├── .env.example                            # MODIFIED: SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET, OTEL_*
├── config/
│   ├── router.yaml                         # NEW: RouterFeatures → model profile mapping (spec §10.2)
│   ├── policy.yaml                         # MODIFIED: hitl.slack_channel + cost_ledger_enabled
│   └── litestream.yml                      # NEW: backup destination + replication interval
├── firm/
│   ├── core/
│   │   └── models.py                       # MODIFIED: RouterFeatures, CostLedgerRow, ReconcileBlock
│   ├── db/
│   │   └── schema.sql                      # MODIFIED: cost_ledger table; existing hitl_queue.approver_id NOT NULL
│   ├── obs/                                # NEW MODULE
│   │   ├── __init__.py
│   │   ├── tracer.py                       # OTel TracerProvider + JSONL file exporter
│   │   └── spans.py                        # @span decorators + helpers for agent/LLM/tool/retrieval
│   ├── llm/
│   │   ├── router.py                       # NEW: CostRouter wrapping CachedAnthropicClient
│   │   └── anthropic_client.py             # MODIFIED: emit cost + token counts onto active OTel span
│   ├── hitl/
│   │   ├── slack.py                        # NEW: Slack ingress (FastAPI app or webhook), HMAC verify, calls existing ack()
│   │   └── signing.py                      # NEW: sign/verify HMAC over (decision_id, approver_id, ts)
│   ├── reports/                            # NEW MODULE
│   │   ├── __init__.py
│   │   ├── daily.py                        # render_daily_report(date, db, broker, traces_path) entry point
│   │   ├── md_template.j2                  # Markdown narrative template
│   │   └── xlsx.py                         # positions/P&L XLSX writer
│   ├── rag/
│   │   ├── transcripts.py                  # NEW: TranscriptsCorpusSource (parameterised over local JSONL dump)
│   │   └── news.py                         # NEW: NewsCorpusSource (Polygon or NewsAPI; no-op without creds)
│   └── cli.py                              # MODIFIED: `report` subcommand; `run --once` boots tracer; `--dev-ack` flag
└── tests/
    ├── unit/
    │   ├── test_cost_router.py             # NEW: RouterFeatures → profile mapping, fallback ladder
    │   ├── test_otel_spans.py              # NEW: JSONL spans contain all spec §10.1 fields
    │   ├── test_slack_signing.py           # NEW: HMAC sign/verify roundtrip + tampering rejection
    │   ├── test_daily_report.py            # NEW: golden Markdown + XLSX structure + RECONCILIATION block
    │   ├── test_transcripts_source.py      # NEW: published_at discipline, no-NULL invariant
    │   └── test_news_source.py             # NEW: graceful degradation when creds absent
    └── integration/
        ├── test_hitl_slack_e2e.py          # NEW: signed button → ack → LangGraph wake
        ├── test_reconciliation_eod.py      # NEW: matches spec §5.7 EOD block
        ├── test_cost_router_fallback.py    # NEW: Sonnet 503 → Haiku → REFUSE
        └── test_otel_run_e2e.py            # NEW: one heartbeat → one trace.jsonl with N spans
```

---

## Task list

Tasks are ordered for subagent-driven execution. Each task is sized to fit one fresh-subagent invocation (plus two-stage review). Tasks within the same section can run sequentially; tasks across sections are mostly independent and can interleave.

### Section A — Observability spine (§10.1)

This section is first because every downstream task (router, reports, HITL) emits spans, and the test invariants for those tasks assume the span schema.

- [x] **T01: Bootstrap `firm/obs/tracer.py`** — TracerProvider + custom file exporter that writes one JSON line per span to `traces/YYYY-MM-DD/run-<run_id>.jsonl`. `run_id` = `ulid_new()` minted at CLI entry. Honor `OTEL_EXPORTER` env (`file` default; `otlp` for production). Default span processor is `BatchSpanProcessor` so hot-path agent code doesn't block on disk; expose `firm.obs.tracer.use_sync_exporter()` (also via `FIRM_OTEL_SYNC=1` env) that swaps in `SimpleSpanProcessor` for test determinism. Conftest enables sync mode at session scope. Fields per span match spec §10.1: `trace_id`, `span_id`, `parent_span_id`, `agent`, `operation`, `decision_id`, `duration_ms`, `model`, `input_tokens`, `output_tokens`, `cached_tokens`, `cost_usd`, `citations`, `failure_mode`, `status`. Test in `tests/unit/test_otel_spans.py` writes a known span and asserts every field present + JSON-parsable.

- [x] **T02: `firm/obs/spans.py` decorators** — `@agent_span("research")`, `@llm_span("anthropic", model)`, `@tool_span("fundamentals.get_ratio")`, `@retrieval_span("hybrid|rerank|pit")`. Each is a context manager that opens a span, sets `agent`/`operation` attrs, swallows-and-rethrows exceptions (setting `status="error"` + the exception class name on `failure_mode`). Test verifies a nested span sequence yields correct `parent_span_id` chaining.

- [x] **T03: Wire spans into existing agents** — Replace ad-hoc logging in `firm/agents/research.py`, `firm/agents/pm.py`, `firm/agents/risk.py`, `firm/orchestrator/graph.py` with the decorators from T02. The Risk node sets `failure_mode` from the Decision when present. The Reporter writes `decisions.jsonl` *and* the trace pointer onto each row. Spec-compliance test: `tests/integration/test_otel_run_e2e.py` runs one heartbeat and asserts ≥ 1 span per agent + 1 span per LLM call + 1 trace per retrieval stage.

- [x] **T04: CachedAnthropicClient onto-span instrumentation** — in `firm/llm/anthropic_client.py`, when a cached hit is served, record `cached_tokens` and `cost_usd=0.0` onto the *currently active* span. On a real call, record `input_tokens`, `output_tokens`, computed `cost_usd` from a per-model rate card in `config/router.yaml`. Test confirms cached vs live paths leave distinguishable spans.

### Section B — Cost router + fallback ladder (§10.2)

- [x] **T05: `RouterFeatures` model** — `firm/core/models.py`, fields `risk_weight: float`, `novelty: float`, `complexity: float`, `time_pressure: float`. Add a `score(weights) -> profile_name` static method. Test exhaustively covers low/standard/high-risk boundary cases.

- [x] **T06: `config/router.yaml`** — Three profiles `{haiku, sonnet, opus}`. Each carries `model_id`, `max_tokens`, `temperature`, `input_per_mtok_usd`, `output_per_mtok_usd`. Top-level `weights` for the scoring function and explicit `fallback_chain: [sonnet, haiku]`. Load via `firm.core.config.load_router_config()` mirroring `load_policy()`. Test asserts every profile resolves to a real model_id from `config/llm.yaml`.

- [x] **T07: `firm/llm/router.py`** — `CostRouter(features, anthropic_client) -> ProfileChoice`. Public API: `route_for_decision(features) -> profile` and `call_with_fallback(profile, system, messages, tools) -> AnthropicResponse`. The fallback ladder retries Sonnet with truncated chunks once, then downgrades to Haiku with `max_tokens *= 0.5`, then raises `LLMUnavailableError`. Test the ladder by injecting a stub client whose first two calls 503.

- [x] **T08: Wire router into Research + PM** — Research uses Sonnet by default; passes `RouterFeatures(novelty=high, complexity=high)` for first-of-kind tickers (the same signal `escalate_new_ticker` watches). PM voters use Sonnet; only escalate to Opus when sufficiency judge returned PARTIAL but the human ack overrode. On `LLMUnavailableError`: REFUSE with `failure_mode=LLM_UNAVAILABLE`, conservative payload "all-models-exhausted". Spec-compliance test: simulate Sonnet down → assert REFUSE path + cost ledger row written.

- [x] **T09: `cost_ledger` table** — Add to `firm/db/schema.sql`: `(decision_id, agent, model, input_tokens, output_tokens, cached_tokens, cost_usd, created_at)`. Append-only. The router writes one row per LLM call (cached or live). Test asserts schema indices on `decision_id` and `created_at`.

### Section C — HITL via signed Slack (§8.4)

- [x] **T10: `firm/hitl/signing.py`** — `sign(decision_id, approver_id, ts, secret) -> str`, `verify(payload, signature, secret) -> bool`. HMAC-SHA256 over a canonical `f"{decision_id}|{approver_id}|{ts}"`. Reject signatures older than 5 minutes (replay defense). Test the tampering paths: wrong secret, wrong approver, swapped decision_id, expired timestamp.

- [x] **T11: `firm/hitl/slack.py`** — Slim FastAPI app exposing `POST /slack/interactive`. Verifies Slack's request-signing header (per Slack docs) **and** our internal HMAC on the button payload. On verified ack, calls the existing `hitl.ack(decision_id, approver_id)` which already writes to `audit_log` and unblocks the LangGraph node. Returns 200 with an updated ephemeral message. Test with stubbed Slack signing + golden payload fixture.

- [x] **T12: Slack notifier on HITL entry** — When the LangGraph reaches an ESCALATE node, push a threaded message to `policy.hitl.slack_channel` containing the Decision summary + a `Approve`/`Reject` button pair, each carrying a signed payload (T10). Reuse the existing `slack-sdk` `WebClient` so we have a single dependency. Test: ESCALATE Decision in unit fixture → asserts one `chat.postMessage` mock call with two buttons.

- [x] **T13: `--dev-ack` CLI fallback** — Keep `firm.cli.ack <decision_id>` working unchanged but require `--dev-ack` flag in non-test environments (otherwise emit a Slack reminder and exit 1). Ensures we don't accidentally ship a backdoor. Test: missing flag → exits 1; with flag → original behavior.

- [x] **T13a: Dual-key Slack secret rotation** — Extend `firm/hitl/signing.py:verify()` to accept either `FIRM_SLACK_SECRET` or optional `FIRM_SLACK_SECRET_PREVIOUS`, valid only while `FIRM_SLACK_SECRET_ROTATED_AT` is within a configurable grace window (default 24h). Logs which key matched so audit can trace rotation events. Runbook section in T29 documents the procedure: set previous → set new → wait window → unset previous. Test: valid sig under previous key during window accepted; same sig after window rejected; tampered sig rejected under both keys.

- [x] **T14: `hitl_queue.approver_id` NOT NULL** — `firm/db/schema.sql` migration: the approver_id column is currently nullable from Plan 1's stub. Tighten to NOT NULL since Slack always supplies it. Test: insertion without approver_id raises `IntegrityError`.

### Section D — Daily reports + EOD reconcile (§5.7, §10.3)

- [x] **T15: `firm.reports.xlsx` writer** — Writes `positions.xlsx` with two sheets (`Positions`, `P&L`). Sources data from the broker (positions/cash at close) and `decisions` table (P&L attribution). Golden-file test compares cell-by-cell.

- [x] **T16: `firm.reports.daily` Markdown** — `render_daily_report(date, db, broker, traces_path) -> Path`. Uses Jinja2 template `md_template.j2`. Sections: (1) Decision summary (count, BUY/SELL/HOLD/REFUSE/ESCALATE histogram with failure_mode breakdown), (2) Cost summary (group cost_ledger by model — output matches spec §10.2 example exactly), (3) `RECONCILIATION (EOD)` block (T17). Golden file in `tests/fixtures/reports/2024-03-13/daily_report.md`.

- [x] **T17: EOD reconcile block** — Re-runs `reconcile_on_boot()` after market close, formats output per spec §5.7. Non-empty diff renders the section in red (Markdown footnote convention) and links to `audit_log` entry. Test the three drift scenarios (clean / position-drift / cash-drift) against golden output.

- [x] **T18: `make report` target** — `firm.cli report --date YYYY-MM-DD` writes the bundle into `reports/YYYY-MM-DD/`. Idempotent: re-running overwrites. CI committed-bundle invariant: the sample run committed under `sample_runs/2024-03-13/` includes the daily report bundle.

### Section E — Corpus adapters (§6.3)

- [x] **T19: `firm.rag.transcripts.TranscriptsCorpusSource`** — Local JSONL adapter (path from `config/rag.yaml:corpus.transcripts_path`). Each line: `{ticker, quarter, fiscal_year, published_at, body}`. `published_at` required; missing rejected per spec §6.3 discipline. Reuses the same chunker + contextual augmentation as FinanceBench. Test the no-NULL invariant.

- [x] **T20: `firm.rag.news.NewsCorpusSource`** — Polygon or NewsAPI client gated by `POLYGON_API_KEY` / `NEWSAPI_KEY` env. Without creds it's a no-op (logs once, returns empty iter). With creds it polls rolling 12 months for the 30-ticker universe. Test both paths.

- [x] **T20a: News rate limiting + opt-in flag** — Wrap the Polygon/NewsAPI client in `firm/rag/_rate_limit.py:TokenBucket` (4 req/min ceiling — one below Polygon free tier's 5/min cap) with exponential backoff on 429. Cache headlines per `(ticker, YYYY-MM-DD)` in SQLite so a heartbeat retry doesn't double-spend the quota. Gate the entire adapter behind `FIRM_NEWS_ENABLED` (default `false`); production env opts in explicitly. Test: 6 rapid calls → only 4 hit the wire, 2 wait; 429 response → 1 backoff before retry; disabled flag → no network call.

- [x] **T21: Multi-source `make ingest`** — Update `firm.cli ingest` to take `--source {financebench,transcripts,news,all}` (default `all`). Each source contributes chunks to the same `firm_chunks` Qdrant collection; the chunk payload carries `source: str` so retrieval can be filtered if needed. Test ingestion is order-independent and idempotent (uses the now-non-destructive `create_collection` from Plan 2's hardening).

### Section F — Litestream live

- [x] **T22: Litestream container** — Add a `litestream` service to `docker-compose.yml` running `litestream replicate` against `data/firm.db` to a `data/litestream/` directory (file destination; S3 is optional via env). Healthcheck: file destination growing. Config `config/litestream.yml` sets `max-wal-size: 16MB` so a stuck checkpointer surfaces as a replication error rather than unbounded growth; `PRAGMA wal_autocheckpoint=1000` in `firm/db/__init__.py` complements this on the SQLite side. Test: integration `docker compose up firm litestream && stop firm && verify litestream caught up`.

- [x] **T23: PIT restore drill** — `docs/runbook.md` gets a "Restore from Litestream" section with exact commands. CI invariant: `make litestream-drill` (a target that restores into a temp DB and asserts row counts match) runs green. After the drill, an additional assertion: `os.path.getsize('data/firm.db-wal') < 16 * 1024 * 1024` — catches a paused-but-undetected replicator before it eats the disk.

- [x] **T23a: `firm doctor` health command** — New `firm.cli doctor` subcommand prints WAL size, last-checkpoint age, last-replication timestamp, Qdrant `points_count`, and cost-ledger row count for today. One line per check, `OK` / `WARN` / `FAIL` prefix. Ops wires this to monitoring (cron + alert on any non-OK line). Test: snapshot the output format against a golden fixture.

### Section G — Hardening pickups from Plan 2 audit

These are the loose ends the Plan 2 audit surfaced (the four 🔴/🟡 fixes already landed inline). They are listed here so they aren't lost; each is a 1-task subagent invocation.

- [x] **T24: Parallel PM voting** — Plan 2 ran the three voters sequentially. Plan 3's OTel + cost router make parallel execution observable and cheap. Convert `make_pm.invoke` to use `asyncio.gather` over the three voters; cap concurrency at 3. Save the latency delta into the OTel parent span.

- [x] **T24a: Cached-client thread-safety** — Prerequisite for T24. The current `CachedAnthropicClient` uses a single `sqlite3.connect(...)` from one thread. Convert to a `threading.local()` connection factory (each thread opens its own connection lazily) with a module-level `threading.Lock` only around writes — reads stay lock-free. Document the concurrency model in a `CachedAnthropicClient` class docstring. Stress test: 50 parallel `extract()` calls across 10 threads, assert no `sqlite3.ProgrammingError` and cache-hit rate matches sequential baseline.

- [x] **T25: Full FailureMode CI invariant (partial)** — Plan 2 brought coverage to 9 modes (gate UNCITED_CLAIM was deferred enum-only by design). Plan 3 adds 3 more triggering fixtures (`LLM_UNAVAILABLE`, `RECONCILIATION_DRIFT`, `SIGNED_APPROVAL_INVALID`). Leaves the last gap (UNCITED_CLAIM end-to-end) for Plan 4 because it ties into the red-team corpus.

- [x] **T26: Cost ledger smoke** — `make demo` writes ≥ 1 row to `cost_ledger`. Reporter prints "Cost so far today: $0.0XX" at heartbeat end. Test asserts the print line matches `Cost so far today: \$\d+\.\d{3}`.

- [x] **T27: Hooks for the Plan 2 test-runner audit** — Findings from the post-Plan-2 background test run (250/252 collectible tests pass; verdict 🟡):

  - **T27a: Raise `test_cli_run_produces_decision` subprocess timeout.** `tests/integration/test_cli.py:108` uses `timeout=120`. On cold-machine runs the BM25 pre-pass + `NomicEmbedder._lazy_load` (`sentence_transformers` + torch warmup) + Qdrant init regularly exceeds 120 s. Bump to `timeout=300` AND split the embedder warmup out into a session-scoped `pytest` fixture so subsequent CLI tests don't re-pay the warmup cost. Add a runtime assertion at heartbeat end: "warmup completed in N s" so regressions are visible.

  - **T27b: Reconcile spec §9.8 `runs/<ts>/` vs. impl `data/reports/<date>/`.** The spec sample-run artifact path is `runs/<ts>/`; the implementation writes to `data/reports/<date>/decisions.jsonl`. Pick one and fix the other. Plan 3's sample-run task (T30) is the natural forcing function — its target directory should match whichever path the spec section ends up with after this reconciliation. Recommend updating the spec to match `data/reports/<date>/` (calendar-date keying is what `Reporter` actually emits and aligns with the daily-report bundle).

  - **T27c: Python 3.11 pin enforcement.** README says "verified end-to-end on … Python 3.11". `uv run …` on a machine with 3.13 cached resolves a torch wheel missing `torch.SymInt`, breaking `sentence_transformers` import → breaks every test that exercises the embedder. Document the explicit recreation incantation: `uv venv --python 3.11 && uv pip install -e ".[dev]"`. Add a `firm/__init__.py` guard that raises a clear error if `sys.version_info < (3, 11) or >= (3, 13)`, so the failure mode surfaces in seconds rather than after a 4-minute test run.

  - **T27d: Document the `requires_models` marker.** The test-runner audit found that `@pytest.mark.requires_models` runs by default (no `-m` filter in pyproject.toml). Three integration files and two unit files depend on local model files (bge reranker + nomic embedder weights). Add a top-of-CONTRIBUTING note: "to skip model-loading tests, run `pytest -m 'not requires_models'`". Wire the CI matrix in Plan 4 to run the marker subset separately so model-availability flakes don't block the unit job.

  - **T27e: CRLF/LF policy.** The audit noted that ~15 tracked files picked up CRLF/LF churn mid-session under autocrlf. Add a `.gitattributes` with `* text=auto eol=lf` so tracked files stay LF regardless of operator OS. This is a one-liner with no functional impact; deferring beyond Plan 3 risks a noisy diff every time a Windows operator touches the repo.

### Section H — Documentation + sample run

- [x] **T28: Update `README.md`** — Status table marks Plan 3 done. Quickstart section adds the Slack `.env` block and a one-liner for `make report`. Mention Litestream in "Restore from backup" instead of the volume-tar approach.

- [x] **T29: Update `docs/runbook.md`** — New sections: (1) Slack approval flow (signature verification, replay window, dev fallback), (2) Trace inspection (`jq` recipes against `run-<id>.jsonl`), (3) Cost ledger inspection (SQL queries against `cost_ledger`), (4) Litestream restore drill.

- [x] **T30: Sample run** — Commit `sample_runs/2024-03-13/` with `daily_report.md`, `positions.xlsx`, `decisions.jsonl`, `trace.jsonl`. This is the reviewer's end-to-end replay artifact (spec §10.1 final line).

---

## CI invariants delivered by this plan

| Invariant | Test | Lives in |
|---|---|---|
| Every LLM/tool/retrieval call emits a span | `test_otel_run_e2e` | `tests/integration/` |
| Span schema matches spec §10.1 | `test_otel_spans` | `tests/unit/` |
| Cost ledger row per LLM call | `test_cost_router.test_ledger_write` | `tests/unit/` |
| Sonnet 503 → Haiku → REFUSE | `test_cost_router_fallback` | `tests/integration/` |
| Slack signature tampering rejected | `test_slack_signing` | `tests/unit/` |
| Slack ack → LangGraph wake | `test_hitl_slack_e2e` | `tests/integration/` |
| EOD reconcile block in golden report | `test_daily_report.test_reconciliation_block` | `tests/unit/` |
| Non-empty diff renders red | `test_daily_report.test_drift_renders_red` | `tests/unit/` |
| `published_at` NOT NULL across all corpus sources | `test_transcripts_source` + `test_news_source` | `tests/unit/` |
| Litestream catches up after firm restart | `make litestream-drill` | `Makefile` |
| FailureMode coverage — 7 fixtures + 7 documented `ALLOWED_GAPS` (full 15-value enumeration) | extends Plan 2's `test_failure_mode_coverage` | `tests/integration/test_failure_mode_coverage.py` |

---

## Risks / notes

Each risk below has a planned mitigation; the right column names the task that delivers it.

- **Parallel PM under cached client → T24a.** SQLite raises `ProgrammingError` if a connection is reused across threads. T24a swaps the single connection for a `threading.local()` factory with a write-only lock; T24's 50-call stress test is the regression gate.
- **Slack signing secret rotation → T13a.** Naïve rotation invalidates every pending HITL approval in flight. T13a accepts both current and previous secret during a configurable grace window, and the T29 runbook documents the drain/swap procedure.
- **OTel exporter latency → T01.** Synchronous file export blocks the hot path on every flush. T01 ships `BatchSpanProcessor` by default and exposes a `FIRM_OTEL_SYNC=1` test override so determinism stays available without paying for it in production.
- **News API rate limits → T20a.** Polygon free tier caps at 5 req/min with hard burst bans. T20a adds a 4 req/min token bucket, per-day SQLite cache, exponential backoff on 429, and an opt-in `FIRM_NEWS_ENABLED` flag (default off).
- **Litestream WAL growth → T22 + T23 + T23a.** Strict mode pauses replication during a failed checkpoint, growing the WAL unboundedly. T22 caps WAL at 16 MB via `max-wal-size` + `PRAGMA wal_autocheckpoint=1000`; T23 asserts the cap in the restore drill; T23a's `firm doctor` surfaces WAL size + last-checkpoint age for monitoring.

**Explicitly deferred to Plan 4:**
- FinanceBench-Q&A eval harness with three regime windows (spec §9.1–§9.7).
- Red-team corpus 50 cases across 10 injection classes (spec §8.5).
- GitHub Actions CI workflows incl. golden-file + Litestream drill (spec §11.3).
- Terraform/AgentCore deployment artefacts (spec §11.1, §11.2).
- Final 14/14 FailureMode end-to-end fixtures (15 enum values minus `UNKNOWN` catch-all) including UNCITED_CLAIM — promotes the 7 modes currently in `ALLOWED_GAPS` to first-class triggering fixtures.

**Carried forward documented limitations:** PIT forward-reference leakage (spec §6.4, runbook §"Known Limitations" added in Plan 2).

---

**End of Plan 3.**
