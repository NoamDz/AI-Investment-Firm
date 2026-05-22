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

## `make eval` operational guide

### What it does

`make eval` runs a 3-regime replay-mode smoke sweep over pre-recorded cassettes and
price fixtures.  No network calls are made; all LLM responses come from committed YAML
cassettes and prices from committed Parquet files.  The three regimes each exercise a
different market scenario (bull / bear / choppy — defined in `firm/eval/regimes.py`).

Outputs written to disk:

| Path | Contents |
|------|----------|
| `reports/eval/r1/regime.md` | Per-decision audit for regime r1 (bull) |
| `reports/eval/r2/regime.md` | Per-decision audit for regime r2 (bear) |
| `reports/eval/r3/regime.md` | Per-decision audit for regime r3 (choppy) |
| `reports/eval/summary.md` | Cross-regime aggregate: pass/fail counts, citation rate, risk gate hits |

### Basic invocation

```bash
# Full sweep — all three regimes (default; used in CI):
make eval

# Single regime — faster, useful during PR development:
make eval REGIME=r1
```

The `REGIME=` variable is passed as `--regime $(REGIME)` to `firm.cli eval`.  Passing
`REGIME=` (empty) or omitting it both run the full sweep.  The underlying env vars
`FIRM_LLM_MODE=cached`, `FIRM_VCR_MODE=replay`, `FIRM_PRICES_MODE=replay`, and
`FIRM_RANDOM_SEED=42` are set automatically by the Makefile recipe — do **not** set
them to `live`/`record` unless you are re-recording cassettes (see below).

### Expected runtime

REPLAY mode never touches the network.  Expect the full sweep to complete in under
30 seconds on a developer laptop.  If it takes longer, suspect a cassette miss
(`CassetteMissError`) causing a retry loop rather than a genuine slowdown.

### Determinism gate failure

CI runs `bash scripts/check_reports_clean.sh` after `make eval`.  The script runs
`make eval` **twice** and diffs `reports/eval/`.  A non-empty diff exits 1 and prints
the first 50 lines of the diff.

**Diagnostic procedure:**

1. Run `make eval` locally to reproduce.
2. Inspect `git diff reports/eval/` — compare what changed.
3. **Timestamp/clock drift** — the diff shows date strings that change between
   runs.  The eval path is mis-using `datetime.now()` somewhere it shouldn't.
   Every eval code path must use the seeded clock in `firm/core/random.py` (which
   respects `FIRM_RANDOM_SEED`).  Find the raw `datetime.now()` call and replace
   it with the injected clock.
4. **Content drift** — the diff shows substantive text changes (agent decisions,
   numbers, phrasing).  Two sub-causes:
   a. **Stale cassette** — a prompt or model changed since the cassette was
      recorded; the cassette key no longer matches, so the adapter is falling
      back to a different (or missing) cassette.  Symptom: `CassetteMissError`
      in logs, or the output looks like a different model version answered.
      Fix: re-record via the procedure in the next section.
   b. **Non-deterministic code path** — `set` iteration order, `dict` ordering
      (Python 3.7+ is insertion-ordered, but explicit sorting may be absent
      in some path), or random sampling that bypasses `firm.core.random`.
      Fix: diff and trace the offending call site; add explicit sorting or
      route the RNG through `firm.core.random.get_rng()`.
5. **Stale cassette** — re-record via the procedure in `## Re-recording eval
   cassettes` below.
6. **Other drift** — run `git diff reports/eval/` between the two successive
   runs, narrow to the smallest differing block, and trace the code path from
   the eval harness call site to the non-deterministic output.

See ["## Re-recording eval cassettes"](#re-recording-eval-cassettes) below for how
to refresh cassettes.  See `docs/eval.md` for the harness design.

---

## Re-recording eval cassettes

`make eval` runs in REPLAY mode (no network) and depends on two committed
artifacts: YAML cassettes under `tests/eval/cassettes/<regime>/` and price
parquets under `data/prices_eval/`. Both are captured by a one-time
operator-run script — `scripts/eval_capture.py` — which is the ONLY
sanctioned way to (re)populate them.

### When to re-record

* After any prompt change that flows through the eval harness (system
  prompts, voter rubrics, citation enforcement instructions).
* After a model upgrade (e.g., Sonnet 4.6 → 4.7) — the cassette key is
  model-aware, so a stale cassette manifests as a `CassetteMissError`.
* When adding a new regime to `firm/eval/regimes.py`.
* When the SPY / basket benchmark window in `firm/eval/regimes.py` changes.

### Prerequisites

* `ANTHROPIC_API_KEY` exported in the environment.
* Working network access to `api.anthropic.com` + Yahoo Finance.
* Estimated cost: ~$1-3 per regime (3 regimes total). Use `--dry-run` to
  preview before spending budget.

### Command

```bash
# Preview the plan first (no API calls, no key required):
python scripts/eval_capture.py --regime all --dry-run

# Then capture for real (will prompt for cost confirmation):
python scripts/eval_capture.py --regime all

# Or capture a single regime + skip the prompt for non-interactive runs:
python scripts/eval_capture.py --regime r1 --yes
```

Each regime is launched in its own subprocess with `FIRM_LLM_MODE=record`,
`FIRM_VCR_MODE=record`, and `FIRM_PRICES_MODE=record`. The subprocess
boundary keeps env mutations from leaking across regimes.

### What gets written

| Path | Committed? | Notes |
|------|-----------|-------|
| `tests/eval/cassettes/<regime_id>/*.yaml` | YES | One YAML per unique `(model, system, messages, tools)` tuple. |
| `data/prices_eval/<TICKER>.parquet` | YES | Adjusted closes from yfinance; one parquet per ticker (idempotent). |
| `data/captured/<regime_id>/` | NO | Throwaway eval reports; safe to delete after verification. |

### Verify

After committing the cassettes + parquets, confirm replay-mode `make eval`
succeeds without network:

```bash
# Optionally drop network access (Linux): unshare -n make eval
make eval
```

Eval reports should land under `reports/eval/` with non-empty regime files
and a populated `summary.md`.

### DO NOT

* **DO NOT** run `scripts/eval_capture.py` in CI. It would burn API budget
  on every push and isn't deterministic across recordings.
* **DO NOT** add a `make eval-capture` target. Capture is operator-triggered
  by design — Makefile targets invite muscle-memory re-runs.
* **DO NOT** commit `data/captured/`. Only the cassettes + parquets are
  reproducibility-critical; the captured reports are byproducts.

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

## Deploying to dev

### When to run

Run `make deploy-dev` **only** for the one-time bring-up of a dev environment that does not already exist. It is not a replacement for the iterative Terraform workflow: day-to-day infrastructure changes should be reviewed with `terraform plan` first, then applied manually with `terraform -chdir=infra/terraform apply -var-file=envs/dev.tfvars` after inspecting the plan output.

### Cost warning

`make deploy-dev` provisions real AWS resources that accrue charges immediately:

| Resource | Approximate cost |
|---|---|
| ECS Fargate cluster | ~$15/mo |
| RDS Postgres `db.t4g.micro` | ~$15/mo |
| NAT Gateway | ~$32/mo + per-GB data-processing fee |
| S3 buckets, CloudWatch logs, Secrets Manager | Negligible at idle |

**When you are done**, run the destroy command manually to avoid ongoing charges:

```bash
terraform -chdir=infra/terraform destroy -var-file=envs/dev.tfvars
```

`make destroy-dev` does **not** exist — this is intentional. Destroys are higher-risk than applies (they delete data), so operators must invoke the destroy command directly after confirming the blast radius.

### Prerequisites

Before running `make deploy-dev`:

1. **`AWS_PROFILE` configured** — ensure `~/.aws/credentials` or `~/.aws/config` contains a profile with sufficient IAM permissions (ECS, RDS, VPC, S3, Secrets Manager, CloudWatch Logs, IAM roles).
2. **Terraform >= 1.6.0 installed** — verify with `terraform version`.
3. **`envs/dev.tfvars` reviewed** — open `infra/terraform/envs/dev.tfvars` and confirm the VPC CIDR, instance sizes, region, and any secret ARN references are correct for your account before continuing.

### Running the target

```bash
make deploy-dev
```

The target prints a WARNING block listing the resources and estimated costs, then prompts:

```
Type 'DEPLOY' to continue:
```

Type exactly `DEPLOY` (all uppercase) and press Enter. Any other input — including lowercase `deploy`, a typo, or an empty string — aborts with a non-zero exit code and no Terraform invocation.

### How to abort at the confirmation prompt

If you triggered `make deploy-dev` accidentally:

- **Before pressing Enter**: press `Ctrl+C`. The shell process is interrupted; no Terraform command runs.
- **After pressing Enter with `DEPLOY`**: the `terraform apply -auto-approve` has already started. You cannot abort cleanly mid-apply without risking a partially-provisioned environment. Let the apply complete, verify the outputs, then run the manual destroy command above.

### Undoing a dev deployment

There is no `make destroy-dev` target (intentional — see above). To tear down the environment:

```bash
terraform -chdir=infra/terraform destroy -var-file=envs/dev.tfvars
```

Review the plan Terraform prints before typing `yes`. Confirm you are targeting the correct AWS account and region before proceeding.

### CI never runs `deploy-dev`

The `make deploy-dev` target is explicitly absent from all CI workflows. The `.github/workflows/main.yml` workflow runs only `terraform plan` (in read-only mode) to catch configuration drift. `terraform apply` and `terraform destroy` are operator-only operations and must be run locally with valid AWS credentials.

---

## Reading the Terraform plan

### The committed snapshot

`infra/terraform/PLAN.md` is the authoritative pre-apply dry-run snapshot.  It was
captured via `scripts/sanitise_plan.sh` (T38), which runs a `terraform show -no-color`
on a binary plan file and strips account IDs / region noise before committing.  The
CI workflow checks that this file exists and is non-empty but does not re-run
`terraform plan` (that requires live AWS credentials).

### How to regenerate locally

Requires: AWS credentials (`AWS_PROFILE` set or `~/.aws/credentials` populated),
`terraform >= 1.6.0`, and `infra/terraform/envs/dev.tfvars` reviewed.

```bash
# Step 1 — produce the binary plan
terraform -chdir=infra/terraform plan \
  -var-file=envs/dev.tfvars \
  -out=tfplan.bin

# Step 2 — render to human-readable text (update PLAN.md)
terraform -chdir=infra/terraform show -no-color tfplan.bin > infra/terraform/PLAN.md

# Step 3 — strip sensitive values before committing (optional but recommended)
bash scripts/sanitise_plan.sh infra/terraform/PLAN.md
```

The binary `tfplan.bin` is `.gitignore`d; only `PLAN.md` is committed.

### How to read PLAN.md

The file is the raw output of `terraform show -no-color tfplan.bin`.  Key landmarks:

- **`Plan:` summary line** — appears near the top; e.g.
  `Plan: 42 to add, 0 to change, 0 to destroy.`  This is the first thing to check
  when reviewing a plan before apply.
- **Resource blocks** — each resource appears as
  `# module.<module>.<resource_type>.<resource_name> will be created` followed by the
  attribute diff.  Lines with `+` are additions, `~` are in-place updates, `-` are
  deletions.
- **`(known after apply)`** — placeholder for values computed by AWS on create
  (e.g. ARNs, IDs).  Normal; not a problem.

### Per-module ownership

| Module | Owns |
|--------|------|
| `network` | VPC (`/16`), 2 public + 2 private subnets across 2 AZs, Internet Gateway, NAT Gateway (single AZ for dev), public + private route tables, 3 security groups (ECS task egress-only, RDS 5432 from ECS, OTLP 4317 from ECS) |
| `compute` | ECS Fargate cluster (Container Insights on), ECS task execution + task IAM roles, task definition (`FARGATE`/`awsvpc`, `:8080`), ECS service (desired 1, autoscaling 1–3 @ CPU 70 %) |
| `storage` | 3 S3 buckets (`reports`, `traces`, `cassettes`) — versioned, AES-256 SSE, public-access blocked; RDS Postgres 15 (`db.t4g.micro`) in private subnets, master credentials auto-rotated via Secrets Manager |
| `secrets` | Customer-managed KMS key (annual rotation), KMS alias, 6 Secrets Manager entries (`firm/anthropic_api_key`, `firm/slack_signing_secret`, `firm/slack_bot_token`, `firm/firm_hmac_secret`, `firm/firm_hmac_secret_prev`, `firm/firm_hmac_rotated_at`) |
| `bedrock` | IAM role for AgentCore Runtime (trusted by `bedrock-agentcore.amazonaws.com`), inline policy granting HMAC secret reads + KMS decrypt + CW log writes; CloudWatch log group for AgentCore reporter (`/aws/bedrock-agentcore/<project>-<env>-reporter`, 90-day retention) |
| `observability` | CloudWatch log groups (`/firm/<env>` for telemetry, `/ecs/<project>-<env>-otelcol` for collector stdout), otelcol-contrib Fargate service (4317/4318), 4-widget CloudWatch dashboard |

---

## AgentCore Reporter deployment

### What it is

The AgentCore Reporter adapter wraps the existing `firm.agents.reporter.make_reporter`
closure as a `@agent`-decorated function so it can be served by AWS Bedrock AgentCore
Runtime.  The adapter is a thin marshalling shim: it deserialises an
`InvocationRequest.payload` (JSON matching `WorkingState`) into the closure and
serialises the `{"report_path": str}` result back into an `InvocationResponse`.

- **Source**: `firm/agentcore/reporter_adapter.py`
- **Background**: `docs/agentcore_mapping.md` (migration table from LangGraph → AgentCore)
- **Agent name** (Terraform contract): `"firm-reporter"` — matches `locals.agentcore_runtime_name` in `infra/terraform/modules/bedrock/main.tf`

### Local runtime (developer machine)

The `bedrock-agentcore-sdk` is gated under the optional `[agentcore]` extra (T41).
The core LangGraph path never imports this module, so the extra is not required for
normal development.

**Step 1 — install the extra:**

```bash
pip install -e ".[agentcore]"
```

**Step 2 — set env vars:**

```bash
export FIRM_REPORTS_ROOT=/tmp/reports
export FIRM_DB_PATH=/tmp/firm.db   # omit to skip SQLite persistence
```

These are consumed at module import time — changing them after `import
firm.agentcore.reporter_adapter` has no effect without `importlib.reload`.

**Step 3 — invoke via the local runtime:**

```python
import json
from bedrock_agentcore_sdk import InvocationRequest
from firm.agentcore.reporter_adapter import reporter

# Construct a minimal WorkingState payload
payload = json.dumps({
    "decisions": [],
    "as_of": "2024-03-13T14:30:00+00:00",
    # ... other WorkingState fields
})
request = InvocationRequest(payload=payload)
response = reporter(request)
result = json.loads(response.body)  # {"report_path": "/tmp/reports/2024-03-13/decisions.jsonl"}
assert "report_path" in result
```

The local runtime does not require AWS credentials — it runs the `@agent` function
in-process.

### Deployed Runtime (AWS)

**Prerequisites:**

1. `terraform apply` has completed successfully (the `bedrock` module provisions the
   IAM role; the AgentCore Runtime entity itself is created via the CLI — see below).
2. The Docker image includes the `[agentcore]` extra.  **This is NOT included in the
   default production build** — the Dockerfile must be updated to install
   `pip install -e ".[agentcore]"` in addition to the base dependencies before the
   first deployed-runtime invocation.

**Deployment steps (after `terraform apply`):**

```bash
# TODO: confirm exact CLI shape after first prod deploy
aws bedrock-agentcore create-runtime \
  --name firm-reporter \
  --role-arn <output from terraform: module.bedrock.agentcore_runtime_role_arn> \
  --execution-role-arn <agentcore_runtime_role_arn>
```

**Production env vars** must be set in the ECS task definition or injected via
Secrets Manager before the container starts:

| Variable | Dev value | Production value |
|----------|-----------|-----------------|
| `FIRM_REPORTS_ROOT` | `/tmp/reports` | S3 mount path or fixed EFS path |
| `FIRM_DB_PATH` | `/tmp/firm.db` | RDS Postgres connection string or omit for JSONL-only |

**Verification (post-deploy):**

```bash
# TODO: confirm exact CLI shape after first prod deploy
aws bedrock-agent invoke-agent \
  --agent-id <firm-reporter-agent-id> \
  --agent-alias-id <alias-id> \
  --session-id test-session-001 \
  --input-text '{"decisions": [], "as_of": "2024-03-13T14:30:00+00:00"}'
```

See `docs/agentcore_mapping.md` for the migration table.
See `firm/agentcore/reporter_adapter.py` for the entry point.

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

