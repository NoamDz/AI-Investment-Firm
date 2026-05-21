# Operator Runbook — AI Investment Firm (Plan 2)

Terse reference for day-to-day operations.  For architecture see
`docs/superpowers/specs/2026-05-18-ai-investment-firm-design.md`.

---

## `make ingest`

### What it does

1. Starts a BM25 vocabulary pre-pass over every FinanceBench document.
2. For each document: preprocesses HTML tables to prose, chunks at
   `target_tokens=512` with `overlap_tokens=64` (per `config/rag.yaml`),
   generates a contextual summary via `claude-haiku-4-5` (cached in `llm_cache`),
   embeds with `nomic-ai/nomic-embed-text-v1.5` (dense) and BM25 (sparse).
3. Upserts all chunk vectors into the `firm_chunks` Qdrant collection.
4. Records job status (start time, doc count, chunk count, status) in the
   `ingest_runs` SQLite table.

The LLM summaries are written to `llm_cache` so a re-run is cheap: only
documents not yet summarized make real API calls.

Command:

```bash
docker compose up -d qdrant
make ingest            # uses FIRM_LLM_MODE=cached by default (see below)
```

Override mode to force new API calls:

```bash
FIRM_LLM_MODE=record make ingest
```

### When to re-run

Re-run ingest if any of the following change:

- The FinanceBench corpus (new documents or updated splits).
- `config/rag.yaml` chunking params (`target_tokens`, `overlap_tokens`).
- The dense embedding model (`embedding.dense_model`).
- The contextual summary prompt (in `firm/llm/prompts.py`).

If only the reranker or retrieval `top_k` changes, no re-ingest is needed.

---

## `FIRM_LLM_MODE` semantics

Set via environment variable before `firm run` or `firm ingest`.

| Mode | Behaviour |
|------|-----------|
| `live` | Real Anthropic API calls; responses written to `llm_cache`. Default if unset. |
| `cached` | Read-only from `llm_cache`; raises `LlmCacheMissError` on a cache miss. Use in CI and for deterministic replay. |
| `record` | Same as `live` but explicitly logs that new entries are being written. Use to seed fixtures before flipping CI to `cached`. |

Examples:

```bash
# Production (default)
firm run --once

# CI — fail loudly if any prompt is not pre-cached
FIRM_LLM_MODE=cached firm run --once

# Seed the cache from a fresh API run
FIRM_LLM_MODE=record firm run --once
```

Unknown values fall back silently to `cached`.

---

## `FIRM_VCR_MODE` semantics

Set via environment variable alongside `FIRM_LLM_MODE`. Controls the cassette
layer that sits between the SQLite cache and the live Anthropic SDK.

| Mode | Behaviour |
|------|-----------|
| `live` | Cassette layer is a pass-through; calls the real Anthropic SDK. Default if unset. |
| `record` | Pass-through to the real SDK + writes `.yaml` cassettes under `FIRM_CASSETTE_DIR` (default `tests/eval/cassettes/`). Use to seed cassette files before flipping to `replay`. |
| `replay` | Reads cassettes only; raises `CassetteMissError` on miss. Used by tests and `make eval`. |

**`FIRM_CASSETTE_DIR`** — path to the cassette directory (default:
`tests/eval/cassettes/` relative to the repo root). Override when running
record-mode tests against a scratch directory.

**Interaction with `FIRM_LLM_MODE`:** the cassette layer only engages when
`FIRM_LLM_MODE` is `live` or `record`.  In `cached` mode the transport is
never reached, so `FIRM_VCR_MODE` has no effect.

Examples:

```bash
# Re-record cassettes after a prompt change
FIRM_LLM_MODE=record FIRM_VCR_MODE=record make eval

# Replay cassettes in CI (no network calls)
FIRM_LLM_MODE=live FIRM_VCR_MODE=replay make eval
```

---

## `llm_cache` table

Schema (from `firm/db/schema.sql`):

| Column | Type | Notes |
|--------|------|-------|
| `prompt_hash` | TEXT | SHA-256 of `(system, messages, tools)` |
| `model` | TEXT | e.g. `claude-sonnet-4-6` |
| `response_json` | TEXT | Full Anthropic response JSON |
| `input_tokens` | INTEGER | Token count from response usage |
| `output_tokens` | INTEGER | Token count from response usage |
| `created_at` | TEXT | ISO-8601 UTC timestamp |

Primary key: `(prompt_hash, model)`.

### Inspect

```bash
sqlite3 data/firm.db "SELECT prompt_hash, model, length(response_json) FROM llm_cache LIMIT 5"
```

### Clear individual entries

```bash
sqlite3 data/firm.db "DELETE FROM llm_cache WHERE model='claude-haiku-4-5'"
```

### Clear everything

```bash
sqlite3 data/firm.db "DELETE FROM llm_cache"
# or wipe the entire DB (also clears decisions, outbox, positions):
make clean
```

After clearing, set `FIRM_LLM_MODE=record` and run `make ingest` + `make demo`
to repopulate the cache before switching back to `cached` mode.

---

## Qdrant volume backup

The Qdrant data is stored in the named Docker volume `qdrant_data`
(declared in `docker-compose.yml`).  The volume lives at the Docker
engine's volume root (typically `/var/lib/docker/volumes/qdrant_data/`
on Linux, or the Docker Desktop VM equivalent on macOS/Windows).

### Before re-indexing

```bash
# Dump the volume contents to a tar archive via a temporary container:
docker run --rm \
  -v qdrant_data:/qdrant/storage:ro \
  -v "$(pwd)":/backup \
  busybox \
  tar czf /backup/qdrant_storage_backup_$(date +%Y%m%d).tar.gz /qdrant/storage
```

### Restore

```bash
docker compose down
docker volume rm qdrant_data
docker volume create qdrant_data
docker run --rm \
  -v qdrant_data:/qdrant/storage \
  -v "$(pwd)":/backup \
  busybox \
  tar xzf /backup/qdrant_storage_backup_YYYYMMDD.tar.gz -C /
docker compose up -d qdrant
```

The `firm_chunks` collection will be available immediately after Qdrant starts.
No re-ingest needed after a restore from a clean backup.

### Verify collection after restore

```bash
curl -s http://localhost:6333/collections/firm_chunks | python -m json.tool | grep vectors_count
```

---

## Restore from Litestream (SQLite)

Litestream continuously replicates `data/firm.db` (and its WAL) to
`data/litestream/firm/` via the `litestream` service in `docker-compose.yml`.
This is the path of choice for SQLite recovery; the Qdrant volume backup
above is a separate procedure for the vector store.

### Listing available generations

```bash
docker compose run --rm litestream snapshots /data/firm.db
```

Each row is a generation + snapshot pair you can restore from.  The
most recent generation appears at the top.

### Restoring the latest snapshot

```bash
docker compose stop firm                  # avoid concurrent writers
docker compose run --rm litestream \
  restore -o /data/firm.restored.db /data/firm.db
mv data/firm.restored.db data/firm.db     # promote into place
docker compose start firm
```

After restart, `firm` opens the restored DB on its next connection.  The
WAL/SHM files are recreated automatically.

### Restoring to a specific point in time

```bash
docker compose run --rm litestream \
  restore -o /data/firm.pit.db \
  -timestamp 2024-03-13T14:30:00Z \
  /data/firm.db
```

The replicator keeps 72 h of history (see `config/litestream.yml`).

### Drill — verify replication is healthy

```bash
make litestream-drill
```

The drill (script in `scripts/litestream_drill.py`):

1. Asserts `data/firm.db-wal` is under the 16 MB ceiling from
   `config/litestream.yml` (catches a paused replicator before it
   eats the disk).
2. If a working litestream binary or docker is available, also runs
   a synthetic replicate-then-restore cycle and asserts row counts
   match.  Otherwise prints a `SKIPPED:` notice and exits 0.

Run this in CI and on every operator on-call rotation.

### When the drill fails

| Failure              | Likely cause                               | Fix |
|----------------------|--------------------------------------------|-----|
| WAL oversized        | Replicator paused / crashed                | `docker compose restart litestream` |
| Restore row mismatch | Replica corrupt or replication interrupted | Investigate via `docker compose logs litestream`; consider re-snapshotting from a clean state |
| `SKIPPED:` in CI     | Docker daemon not in CI runner             | Acceptable for now; add docker-in-docker for stronger guarantee |

---

## Slack approval flow

### Required configuration

| Source | Key | Purpose |
|--------|-----|---------|
| `.env` | `FIRM_SLACK_BOT_TOKEN` | OAuth bot token for outbound `chat.postMessage` notifications |
| `config/policy.yaml` | `hitl.slack_channel` | Channel ID where approval messages are posted |
| `config/policy.yaml` | `hitl.slack_approver_id` | Slack user ID who must click Approve/Reject |

The `POST /slack/interactive` endpoint is served by the FastAPI app in `firm/hitl/slack.py`.

### Signature verification

Every inbound request is verified at two levels:

1. **Slack outer HMAC** (`X-Slack-Signature` header): `v0=HMAC-SHA256(slack_signing_secret, "v0:{X-Slack-Request-Timestamp}:{raw_body}")`. Requests older than 300 seconds are rejected (replay-window protection).

2. **Internal button HMAC** (`sig` field inside the button `value` JSON): proves our notifier constructed the button payload. Uses `firm.hitl.signing.sign/verify` over `"{decision_id}|{approver_id}|{ts}"`.

### Audit log entries

On **successful approval**: `hitl_queue.status` flips to `approved`; no explicit audit_log entry — the status column is the record.

On **signature failure**: `audit_log.event = 'hitl.signature_rejected'` with `detail.failure_mode = 'signed_approval_invalid'`.

Query recent rejections:

```bash
sqlite3 data/firm.db "SELECT ts, detail FROM audit_log WHERE event='hitl.signature_rejected' ORDER BY ts DESC LIMIT 10"
```

### Dev fallback

When Slack is unavailable or during development, approve via CLI:

```bash
python -m firm.cli ack <DECISION_ID> --dev-ack
```

The `--dev-ack` flag is required outside a pytest session; without it the CLI exits 1 with a reminder to use the Slack workflow. `reject` accepts the same flag.

---

## Trace inspection (`jq` recipes)

Traces are written to `traces/<YYYY-MM-DD>/run-<run_id>.jsonl` (one JSON object per line, one line per completed span). The path root is controlled by `FIRM_TRACES_ROOT` (default: `traces`).

Span schema fields: `trace_id`, `span_id`, `parent_span_id`, `agent`, `operation`, `decision_id`, `duration_ms`, `model`, `input_tokens`, `output_tokens`, `cached_tokens`, `cost_usd`, `citations`, `failure_mode`, `status`.

### All spans for a specific decision

```bash
jq 'select(.decision_id == "dec-abc123")' traces/2024-03-13/run-<run_id>.jsonl
```

### All `agent.research` spans (longest first)

The exporter does not write a `start_time` field, and `span_id` is a random
OTel-assigned 64-bit hex (not monotonic), so neither can be used to recover
chronological order. JSONL line order reflects span-end order; for ranking
by weight use `duration_ms`:

```bash
jq -s '[.[] | select(.operation == "agent.research")] | sort_by(-.duration_ms)' \
  traces/2024-03-13/run-<run_id>.jsonl
```

### Total cost across all LLM spans in a run

```bash
jq -s '[.[] | select(.cost_usd > 0) | .cost_usd] | add // 0' \
  traces/2024-03-13/run-<run_id>.jsonl
```

---

## Cost ledger inspection

The `cost_ledger` table records one row per LLM call (cached or live).

Schema:

| Column | Type | Notes |
|--------|------|-------|
| `id` | INTEGER | Auto-increment monotonic key |
| `decision_id` | TEXT | Enclosing Decision (may be `""` for pre-decision calls) |
| `agent` | TEXT | e.g. `research`, `pm`, `risk` |
| `model` | TEXT | Anthropic model id |
| `input_tokens` | INTEGER | NULL for cached rows |
| `output_tokens` | INTEGER | NULL for cached rows |
| `cached_tokens` | INTEGER | NULL for live rows |
| `cost_usd` | REAL | 0.0 for cached rows |
| `created_at` | TEXT | ISO-8601 UTC |

### Total spend today

```bash
sqlite3 data/firm.db "SELECT ROUND(SUM(cost_usd), 4) FROM cost_ledger WHERE date(created_at) = date('now')"
```

### Per-agent, per-model breakdown

```bash
sqlite3 data/firm.db "SELECT agent, model, ROUND(SUM(cost_usd),4) as total_usd FROM cost_ledger GROUP BY agent, model ORDER BY total_usd DESC"
```

### Cache-hit ratio

```bash
sqlite3 data/firm.db "SELECT
  COUNT(CASE WHEN cached_tokens IS NOT NULL THEN 1 END) AS cached_calls,
  COUNT(CASE WHEN input_tokens IS NOT NULL THEN 1 END) AS live_calls,
  COUNT(*) AS total
FROM cost_ledger WHERE date(created_at) = date('now')"
```

### Most expensive individual decisions

```bash
sqlite3 data/firm.db "SELECT decision_id, ROUND(SUM(cost_usd),4) as total FROM cost_ledger GROUP BY decision_id ORDER BY total DESC LIMIT 10"
```

---

## Known Limitations

### Forward-reference leakage in PIT-filtered RAG (spec §6.4)

Point-in-time retrieval filters chunks by their **document-level** `published_at`
timestamp against the run's `as_of` cursor. Forward references *inside* an
otherwise-valid chunk — phrases like *"as we'll see in Q4…"* or *"refer to our
upcoming guidance…"* — cannot be detected or stripped automatically, because the
chunk itself is dated before `as_of` and the reference target may never materialize
in the corpus.

**Operator impact.** Backtests may incorporate subtly leaked forward information.
In production, this means historical filings can mention future events that the
agent then treats as known. Spot-check citations on flagged decisions; treat any
research claim whose `source_quote` references a future quarter relative to
`as_of` as suspect.

**Mitigation path.** Detection would require either claim-level published_at
tagging (deferred to Plan 3) or post-hoc forward-reference NER over `cited_text`.
Neither is implemented in Plan 2.

