# Quickstart (hybrid: GPU ingest on host → container runtime)

Embedding the FinanceBench corpus is the only heavy step; we do it on the host
so it can use a local GPU, and the firm itself runs in Docker. Verified on
Windows 11 with Python 3.11 + CUDA 12.4 (RTX 4060) + Docker Desktop.

For the terse 3-command flow, see the root [README](../README.md#quickstart).

## 1. One-time setup (host)

```powershell
# Python 3.11.x specifically — 3.13 wheels are missing torch.SymInt and
# break sentence_transformers. firm/__init__.py raises a clear error on 3.12+/3.10-.
python -m venv .venv
.\.venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch    # or whl/cpu
pip install -e ".[dev]"

# If `uv` is the cached resolver and you hit a "torch.SymInt missing" import error,
# the cached venv is on the wrong Python. Recreate explicitly:
#   uv venv --python 3.11 && uv pip install -e ".[dev]"

copy .env.example .env                  # then edit .env, set ANTHROPIC_API_KEY
```

## 2. Start Qdrant (Docker)

```powershell
docker compose up -d qdrant
docker inspect --format "{{.State.Health.Status}}" plan2-rag-grounding-qdrant-1
# expect: healthy
```

## 3. Ingest the corpus (host, GPU)

```powershell
$env:ANTHROPIC_API_KEY = (Select-String '^ANTHROPIC_API_KEY=' .env).Line.Split('=',2)[1]
$env:QDRANT_URL = "http://localhost:6333"
$env:FIRM_INGEST_MAX_DOCS = "20"        # ~1-2 min on GPU; unset for full 84 docs (~5-10 min)
python -m firm.cli ingest
```

Expected: `ingest completed: corpus=financebench docs_completed=20/20 chunks_written=84`.

Verify:

```powershell
curl http://localhost:6333/collections/firm_chunks
# expect: "points_count": > 0, "status": "green"

python -c "import sqlite3; c=sqlite3.connect('data/firm.db'); print(c.execute('SELECT docs_indexed, chunks_indexed, status FROM ingest_runs ORDER BY rowid DESC LIMIT 1').fetchone())"
# expect: (20, 84, 'completed')
```

## 4. Run the firm (Docker)

```powershell
docker compose up firm
```

Expected last line: `Heartbeat complete. Report: /data/reports/2024-03-13/decisions.jsonl`.

Verify outputs:

```powershell
type data\reports\2024-03-13\decisions.jsonl | python -m json.tool | Select-Object -First 30
python -c "import sqlite3; c=sqlite3.connect('data/firm.db'); print('positions:', list(c.execute('SELECT * FROM positions'))); print('cash:', c.execute('SELECT * FROM cash').fetchone())"
```

### What to expect from the demo

Research always investigates `universe.tickers[0]` (currently `AAPL` — see
`config/universe.yaml`). With the default 20-doc corpus the alphabetically-first
FinanceBench docs are 3M, AES, AMD, Activision, Adobe, Amazon, Amcor, AMEX,
American Water Works — **no Apple** — so retrieval returns no chunks and the
heartbeat **REFUSES** with `failure_mode: insufficient_evidence`. The full
research → PM → risk → execution pipeline still runs end-to-end, terminating
in a signed REFUSE Decision and writing the report.

To see a non-refuse path, ingest the full 84-doc corpus (includes Apple):

```powershell
Remove-Item Env:FIRM_INGEST_MAX_DOCS
python -m firm.cli ingest                    # ~5-10 min on GPU
docker compose up firm
```

### Re-run instantly (deterministic replay from LLM cache)

```powershell
$env:FIRM_LLM_MODE = "cached"
docker compose up firm
```

## HITL flow

> Requires the **full 84-doc corpus** so research produces a BUY rather than REFUSE on AAPL.

The default demo pre-seeds `FIRM_INITIAL_POSITIONS={"AAPL":"10"}` (in
`docker-compose.yml`) so the trade clears risk without human approval. To
exercise the HITL path, wipe initial positions:

```powershell
docker compose run --rm -e FIRM_INITIAL_POSITIONS= firm

docker compose run --rm firm sqlite3 /data/firm.db `
  "SELECT decision_id FROM hitl_queue WHERE status='pending' ORDER BY created_at DESC LIMIT 1"

docker compose run --rm firm python -m firm.cli ack <DECISION_ID>
docker compose run --rm firm
```

### Slack integration

Add to `.env`:

```
FIRM_SLACK_BOT_TOKEN=xoxb-...
```

The `POST /slack/interactive` endpoint verifies every inbound request with
Slack's v0 signing scheme (HMAC-SHA256 over `v0:{timestamp}:{raw_body}`) and
rejects requests older than 5 minutes. The `slack_channel` and
`slack_approver_id` for outbound notifications come from `config/policy.yaml`.

**Dev fallback** (no Slack workspace handy):

```powershell
docker compose run --rm firm python -m firm.cli ack <DECISION_ID> --dev-ack
```

See [runbook §Slack approval flow](runbook.md#slack-approval-flow) for the full operator procedure.

## Generate a daily report

```powershell
make report DATE=2024-03-13
# Equivalent: python -m firm.cli report --date 2024-03-13
```

Writes three artifacts to `data/reports/2024-03-13/`:

- `daily_report.html` — self-contained file, no JS, no external CSS. Open in your browser; print to PDF with Ctrl-P.
- `daily_report.md` — same decision histogram + cost summary + EOD reconcile as before (legacy plain-text).
- `positions.xlsx` — Positions / P&L / Decisions sheets.

## Running `firm run` natively (no Docker)

`docker compose up firm` sets every required env var for you. If you invoke
`python -m firm.cli run …` directly from the host you must additionally export
the HMAC nonce secret:

```powershell
$env:FIRM_HMAC_SECRET = "$(python -c 'import secrets; print(secrets.token_hex(32))')"
```

Without it the CLI exits with `FIRM_HMAC_SECRET is required for the grounded
research path.` A non-hex value also fails fast.

## Real paper trading (Alpaca)

```
FIRM_BROKER=ALPACA
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
```

Then `docker compose up firm`.
