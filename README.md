# AI Investment Firm

Multi-agent paper-trading firm. Take-home for Cato Networks — Agentic AI Engineer.

[![PR CI](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/pr.yml/badge.svg)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/pr.yml)
[![Main CI](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/main.yml)
[![Release](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/release.yml/badge.svg)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/release.yml)

## Quickstart (hybrid: GPU ingest on host → container runtime)

Embedding the FinanceBench corpus is the only heavy step. We do it on the host so it
can use a local GPU; everything else runs in Docker. Verified end-to-end on Windows 11
with Python 3.11 + CUDA 12.4 (RTX 4060) + Docker Desktop.

### 1. One-time setup (host)

```powershell
# Python venv with GPU torch + project deps (skip if you already have a CUDA Python env)
# IMPORTANT: Python 3.11.x specifically — 3.13 wheels are missing torch.SymInt and
# break sentence_transformers. The package raises a clear error on 3.12+/3.10-.
python -m venv .venv
.\.venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch    # or whl/cpu
pip install -e ".[dev]"

# If `uv` is the cached resolver and you hit a "torch.SymInt missing" import error,
# the cached venv is on the wrong Python. Recreate explicitly:
#   uv venv --python 3.11 && uv pip install -e ".[dev]"

# Anthropic key
copy .env.example .env                  # then edit .env, set ANTHROPIC_API_KEY
```

### 2. Start Qdrant (Docker)

```powershell
docker compose up -d qdrant
```

Verify it's healthy:

```powershell
docker inspect --format "{{.State.Health.Status}}" plan2-rag-grounding-qdrant-1
# expect: healthy
```

### 3. Ingest the corpus (host, GPU)

```powershell
$env:ANTHROPIC_API_KEY = (Select-String '^ANTHROPIC_API_KEY=' .env).Line.Split('=',2)[1]
$env:QDRANT_URL = "http://localhost:6333"
$env:FIRM_INGEST_MAX_DOCS = "20"        # 20 docs ≈ 1-2 min on GPU; unset for full 84 docs (~5-10 min)
python -m firm.cli ingest
```

Expected last line:

```
ingest completed: corpus=financebench docs_completed=20/20 chunks_written=84
```

Verify the corpus is populated:

```powershell
curl http://localhost:6333/collections/firm_chunks
# expect: "points_count": > 0, "status": "green"

python -c "import sqlite3; c=sqlite3.connect('data/firm.db'); print(c.execute('SELECT docs_indexed, chunks_indexed, status FROM ingest_runs ORDER BY rowid DESC LIMIT 1').fetchone())"
# expect: (20, 84, 'completed')
```

### 4. Run the firm (Docker)

```powershell
docker compose up firm
```

Expected last container line:

```
Heartbeat complete. Report: /data/reports/2024-03-13/decisions.jsonl
```

Verify outputs:

```powershell
# Heartbeat report on the host (data is bind-mounted)
type data\reports\2024-03-13\decisions.jsonl | python -m json.tool | Select-Object -First 30

# Positions and cash
python -c "import sqlite3; c=sqlite3.connect('data/firm.db'); print('positions:', list(c.execute('SELECT * FROM positions'))); print('cash:', c.execute('SELECT * FROM cash').fetchone())"
```

#### What to expect from the demo

Research always investigates `universe.tickers[0]` (currently `AAPL` — see
`config/universe.yaml`). With the default 20-doc corpus the alphabetically-first
FinanceBench docs are 3M, AES, AMD, Activision, Adobe, Amazon, Amcor, AMEX, American
Water Works — **no Apple** — so retrieval returns no chunks and the heartbeat
**REFUSES** with `failure_mode: insufficient_evidence`. The full
research → PM → risk → execution pipeline still runs end-to-end, terminating in a
signed REFUSE Decision and writing the report.

To see a non-refuse path, ingest the full 84-doc corpus (which includes Apple) and
re-run:

```powershell
Remove-Item Env:FIRM_INGEST_MAX_DOCS
python -m firm.cli ingest                    # ~5-10 min on GPU
docker compose up firm
```

**Re-run instantly (deterministic replay from LLM cache)**

```powershell
$env:FIRM_LLM_MODE = "cached"
docker compose up firm
```

## See the HITL flow

> Requires the **full 84-doc corpus** so research produces a BUY rather than REFUSE on AAPL.

The default demo pre-seeds `FIRM_INITIAL_POSITIONS={"AAPL":"10"}` (in `docker-compose.yml`)
so the trade clears risk without human approval. To exercise the HITL path, wipe initial
positions:

```powershell
docker compose run --rm -e FIRM_INITIAL_POSITIONS= firm

# Find the pending decision id
docker compose run --rm firm sqlite3 /data/firm.db `
  "SELECT decision_id FROM hitl_queue WHERE status='pending' ORDER BY created_at DESC LIMIT 1"

# Approve and resume the graph
docker compose run --rm firm python -m firm.cli ack <DECISION_ID>
docker compose run --rm firm
```

### Slack integration (.env additions)

To enable inbound Slack approval callbacks, add to `.env`:

```
FIRM_SLACK_BOT_TOKEN=xoxb-...        # OAuth bot token for outbound notifications
```

The `POST /slack/interactive` endpoint verifies every inbound request with Slack's v0
signing scheme (HMAC-SHA256 over `v0:{X-Slack-Request-Timestamp}:{raw_body}`) and rejects
requests older than 5 minutes (replay-window protection). The `slack_channel` and
`slack_approver_id` used for outbound notifications come from `config/policy.yaml`, not
from environment variables.

**Dev fallback** — in non-production environments (or with Docker) use the CLI directly:

```powershell
docker compose run --rm firm python -m firm.cli ack <DECISION_ID> --dev-ack
```

`--dev-ack` is required outside a pytest session; without it the CLI exits with a
reminder to use the Slack workflow.

See `docs/runbook.md#slack-approval-flow` for the full operator procedure.

### Generate a daily report

```powershell
make report DATE=2024-03-13
```

Writes the bundle to `data/reports/2024-03-13/`:
- `daily_report.md` — decision histogram, cost summary, EOD reconcile block
- `positions.xlsx` — Positions and P&L sheets

Equivalent without `make`: `python -m firm.cli report --date 2024-03-13`

## Eval harness

`make eval` runs a 3-regime smoke sweep (trending / sideways / volatile) and
re-renders `reports/eval/summary.md`. The Main CI badge above tracks this on
every push to main.

```powershell
make eval
# Equivalent without make:
python -m firm.cli eval
```

Output paths written on each run:

- `reports/eval/r1/regime.md` — trending regime scorecard
- `reports/eval/r2/regime.md` — sideways regime scorecard
- `reports/eval/r3/regime.md` — volatile regime scorecard
- `reports/eval/summary.md` — cross-regime aggregate

**Determinism gate** — cassettes pin every LLM call and the RNG is seeded, so
eval output is byte-stable across re-runs. CI asserts `git diff --exit-code
reports/` after eval; any drift fails the build.

See `docs/eval.md` for the full design (metrics, regimes, "not measured" list).

## Deployment

AWS infrastructure is sketched in Terraform (`infra/terraform/`). The Reporter
agent also runs on AWS Bedrock AgentCore's local runtime via an adapter in
`firm/agentcore/reporter_adapter.py`.

```powershell
# Terraform plan — writes infra/terraform/PLAN.md (dry-run, no real AWS calls)
make deploy-dev
```

To run the AgentCore Reporter locally:

```powershell
pip install -e ".[agentcore]"
# Then invoke via the local runtime entry point:
python firm/agentcore/reporter_adapter.py
```

See `docs/path-to-production.md` for the take-home → prod delta map, and
`docs/agentcore_mapping.md` for the firm-to-AgentCore migration table.

### Restore from backup (SQLite)

SQLite (`data/firm.db`) is continuously replicated by the `litestream` service.
To restore, see `docs/runbook.md#restore-from-litestream-sqlite` for the exact
commands. The Qdrant vector store uses a separate Docker-volume tar approach
documented in `docs/runbook.md#qdrant-volume-backup`.

## Running `firm run` natively (no Docker)

The `docker compose up firm` flow above sets every required environment variable
for you. If you bypass Docker and invoke `python -m firm.cli run …` directly
from the host, you must additionally export:

```powershell
# 32-byte hex string used as HMAC nonce secret for signed decisions.
$env:FIRM_HMAC_SECRET = "$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

Without it the CLI exits with `FIRM_HMAC_SECRET is required for the grounded
research path.` A non-hex value also fails fast with a clear error.

## Real paper trading (Alpaca)

Set in `.env`:

```
FIRM_BROKER=ALPACA
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

Then `docker compose up firm`.

## Architecture

- Plan 1 summary: `docs/implementation_summary_plan_1.md`
- Plan 2 summary: `docs/implementation_summary_plan_2.md`
- Full design spec: `docs/superpowers/specs/2026-05-18-ai-investment-firm-design.md`
- Operator runbook: `docs/runbook.md`
- AgentCore migration map: `docs/agentcore_mapping.md`
- Eval harness design: `docs/eval.md`
- Threat model: `docs/threat_model.md`
- Path to production: `docs/path-to-production.md`

## Status

- [x] Plan 1: Foundation + Walking Skeleton
- [x] Plan 2: RAG + Citations + Grounding
- [x] Plan 3: HITL + Daily Reports + Observability
- [x] Plan 4: Eval Harness + Red Team + CI/CD + Bonuses
