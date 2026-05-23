# Operator Runbook

Day-to-day operations for `firm`. Architecture lives in [`technical-overview.md`](technical-overview.md).

---

## LLM / cassette modes

Two env vars control determinism. Set them before `firm run`, `firm ingest`, or `firm eval`.

| Var | Values | Behaviour |
|---|---|---|
| `FIRM_LLM_MODE` | `cached` *(default)* | Reads from SQLite `llm_cache`; raises `LlmCacheMissError` on miss. CI / replay. |
| | `live` | Real Anthropic API calls; writes responses to `llm_cache`. |
| | `record` | Same as `live` but logs that new entries were written. Use to seed CI. |
| `FIRM_VCR_MODE` | `live` *(default)* | No cassette layer. |
| | `record` | Pass-through to SDK + writes `.yaml` cassettes under `FIRM_CASSETTE_DIR` (`tests/eval/cassettes/`). |
| | `replay` | Reads cassettes only; raises `CassetteMissError` on miss. Used by `make eval`. |

`FIRM_VCR_MODE` only engages when `FIRM_LLM_MODE` is `live` or `record`; in `cached` mode the transport is never reached.

---

## `make ingest` — refresh the RAG corpus

Re-run when any of the following change: the FinanceBench corpus, `config/rag.yaml` chunking params, the dense embedding model, or the contextual summary prompt in `firm/llm/prompts.py`. Reranker / top-k changes do **not** require re-ingest.

```bash
docker compose up -d qdrant
make ingest                       # cached mode; only new docs hit the API
FIRM_LLM_MODE=record make ingest  # force fresh summaries
```

---

## `make eval` & cassette re-recording

`make eval` runs all three regimes from committed YAML cassettes — no network, ~30s on a laptop. CI then re-runs it via `firm/ops/check_reports_clean.sh` and diffs the two outputs; non-empty diff exits 1.

Re-record when prompts change, after a model upgrade, or when a regime / benchmark window changes. Cost is ~$1–3 per regime.

```bash
# Preview (no API calls, no key required)
python firm/ops/eval_capture.py --regime all --dry-run

# Capture (prompts for cost confirmation)
python firm/ops/eval_capture.py --regime all
python firm/ops/eval_capture.py --regime r1 --yes   # single regime, non-interactive
```

Each regime runs in its own subprocess with `FIRM_LLM_MODE=record`, `FIRM_VCR_MODE=record`, `FIRM_PRICES_MODE=record` to keep env mutations from leaking. Cassettes land in `tests/eval/cassettes/<regime>/` and prices in `data/prices_eval/`; both are committed. `data/captured/` is throwaway — don't commit it.

**Determinism drift diagnostics** when `check_reports_clean.sh` fails:

1. `make eval` locally and `git diff reports/eval/` — find the smallest differing block.
2. **Timestamp drift** → some code path uses `datetime.now()` instead of the injected `Clock` from `firm/core/clock.py`. Find and replace.
3. **Content drift** → either stale cassette (re-record) or a non-deterministic Python path (`set` iteration, missing sort, RNG bypass). Route RNG through `firm.core.random.get_rng()`.

> Do **not** add a `make eval-capture` target — capture is operator-triggered by design.

---

## Restore from Litestream (SQLite)

Litestream continuously replicates `data/firm.db` (+ WAL) to `data/litestream/firm/` via the `litestream` service in `docker-compose.yml`. Retains 72h of history (`config/litestream.yml`).

```bash
# List available snapshots
docker compose run --rm litestream snapshots /data/firm.db

# Restore latest
docker compose stop firm
docker compose run --rm litestream restore -o /data/firm.restored.db /data/firm.db
mv data/firm.restored.db data/firm.db
docker compose start firm

# Point-in-time
docker compose run --rm litestream \
  restore -o /data/firm.pit.db -timestamp 2024-03-13T14:30:00Z /data/firm.db
```

Drill — run in CI + on-call rotation:

```bash
make litestream-drill   # asserts WAL < 16MB, optionally replicate-then-restore cycle
```

| Failure | Likely cause | Fix |
|---|---|---|
| WAL oversized | Replicator paused/crashed | `docker compose restart litestream` |
| Restore row mismatch | Replica corrupt | Check `docker compose logs litestream`; re-snapshot |
| `SKIPPED:` in CI | No Docker in runner | Acceptable; add docker-in-docker for stronger guarantee |

---

## HITL — Slack approval + dev fallback

**Config** (`.env` + `config/policy.yaml`):

| Source | Key | Purpose |
|---|---|---|
| `.env` | `FIRM_SLACK_BOT_TOKEN` | OAuth bot token for outbound `chat.postMessage` |
| `policy.yaml` | `hitl.slack_channel` | Channel ID for approval messages |
| `policy.yaml` | `hitl.slack_approver_id` | Slack user who must click Approve/Reject |

Endpoint: `POST /slack/interactive` served by `firm/hitl/slack.py`. Inbound requests verified at two levels:

1. **Slack outer HMAC** — `X-Slack-Signature` against `slack_signing_secret`; timestamps older than 300s rejected (replay window).
2. **Internal button HMAC** — `sig` field inside the button `value` JSON; signed by our notifier over `"{decision_id}|{approver_id}|{ts}"` via `firm.hitl.signing.sign/verify`.

On approval, `hitl_queue.status` flips to `approved` (the column is the audit record). On signature failure, `audit_log.event='hitl.signature_rejected'` with `detail.failure_mode='signed_approval_invalid'`.

```bash
# Query recent signature rejections
sqlite3 data/firm.db "SELECT ts, detail FROM audit_log \
  WHERE event='hitl.signature_rejected' ORDER BY ts DESC LIMIT 10"
```

**Dev fallback** when Slack is unavailable (or for the demo):

```bash
python -m firm.cli ack <DECISION_ID> --dev-ack       # approve
python -m firm.cli reject <DECISION_ID> --dev-ack    # reject
```

`--dev-ack` is required outside pytest; without it the CLI exits 1 and points you at Slack.

---

## Cost ledger inspection

`cost_ledger` records one row per LLM call. Schema in `firm/db/schema.sql`. Useful queries:

```bash
# Total spend today
sqlite3 data/firm.db "SELECT ROUND(SUM(cost_usd),4) FROM cost_ledger \
  WHERE date(created_at)=date('now')"

# Per-agent, per-model breakdown
sqlite3 data/firm.db "SELECT agent, model, ROUND(SUM(cost_usd),4) AS total_usd \
  FROM cost_ledger GROUP BY agent, model ORDER BY total_usd DESC"

# Cache-hit ratio today
sqlite3 data/firm.db "SELECT
  COUNT(CASE WHEN cached_tokens IS NOT NULL THEN 1 END) AS cached,
  COUNT(CASE WHEN input_tokens  IS NOT NULL THEN 1 END) AS live,
  COUNT(*) AS total FROM cost_ledger WHERE date(created_at)=date('now')"

# Most expensive decisions
sqlite3 data/firm.db "SELECT decision_id, ROUND(SUM(cost_usd),4) AS total \
  FROM cost_ledger GROUP BY decision_id ORDER BY total DESC LIMIT 10"
```

---

## Trace inspection (`jq`)

Traces land in `traces/<YYYY-MM-DD>/run-<run_id>.jsonl` (one span per line). Root is `FIRM_TRACES_ROOT` (default `traces/`). Span fields: `trace_id`, `span_id`, `parent_span_id`, `agent`, `operation`, `decision_id`, `duration_ms`, `model`, `input_tokens`, `output_tokens`, `cached_tokens`, `cost_usd`, `failure_mode`, `status`.

```bash
# All spans for a decision
jq 'select(.decision_id=="dec-abc123")' traces/2024-03-13/run-*.jsonl

# Slowest research spans
jq -s '[.[]|select(.operation=="agent.research")]|sort_by(-.duration_ms)' \
  traces/2024-03-13/run-*.jsonl

# Total LLM cost in a run
jq -s '[.[]|select(.cost_usd>0)|.cost_usd]|add // 0' \
  traces/2024-03-13/run-*.jsonl
```

---

## AWS deploy (dev)

```bash
make deploy-dev   # WARNING block + prompts for literal 'DEPLOY' before apply
```

Costs immediately: ECS Fargate ~$15/mo, RDS `db.t4g.micro` ~$15/mo, NAT Gateway ~$32/mo + egress. **There is no `make destroy-dev` by design** — destroys are higher blast-radius than applies. Tear down manually:

```bash
terraform -chdir=infra/terraform destroy -var-file=envs/dev.tfvars
```

Module ownership + regenerating `PLAN.md` lives in [`infra/README.md`](../infra/README.md). CI runs `terraform plan` in read-only mode only — apply / destroy are operator-only.

---

## AgentCore Reporter

The Reporter agent is also wrapped as a Bedrock AgentCore `@agent` adapter at `firm/agentcore/reporter_adapter.py`. The SDK lives behind the optional `[agentcore]` extra so the core LangGraph path doesn't import it.

```bash
pip install -e ".[agentcore]"
export FIRM_REPORTS_ROOT=/tmp/reports
export FIRM_DB_PATH=/tmp/firm.db
```

Env vars are consumed at import time — change requires `importlib.reload`. Migration table + invocation contract: [`agentcore_mapping.md`](agentcore_mapping.md). The Terraform `bedrock` module provisions the IAM role; the Runtime entity itself is created via `aws bedrock-agentcore create-runtime` after `terraform apply`.

---

## Known limitations

**Forward-reference leakage in PIT-filtered RAG** (spec §6.4). PIT filtering is at the **document** level (chunk's `published_at` vs `as_of`). Forward references *inside* an otherwise-valid chunk — *"as we'll see in Q4…"* — cannot be detected automatically. Backtests may incorporate subtly leaked forward info; spot-check citations on flagged decisions. Mitigation requires claim-level published_at tagging or post-hoc forward-reference NER over `cited_text`.
