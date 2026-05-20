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
