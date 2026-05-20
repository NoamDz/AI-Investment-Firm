# AI Investment Firm

Multi-agent paper-trading firm. Take-home for Cato Networks — Agentic AI Engineer.

## Quickstart (Plan 2 — RAG-grounded demo)

```bash
docker compose up -d qdrant
pip install -e ".[dev]"
make ingest    # one-time, populates Qdrant from FinanceBench
make demo      # heartbeat with grounded research and confirmed paper trade
```

Output: one heartbeat through the 5-agent workflow, a JSONL report under `data/reports/<date>/` containing at least one citation, and one `confirmed` row in the `outbox` table.

## Docker demo

```bash
docker compose up --build
```

## Real paper trading (Alpaca)

```bash
export FIRM_BROKER=ALPACA
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
make demo
```

## Architecture

See `docs/superpowers/specs/2026-05-18-ai-investment-firm-design.md` for the full design.

## Status

- [x] **Plan 1 (this branch):** Foundation + Walking Skeleton — 5 agents stubbed, outbox-protected trade, boot reconciliation.
- [x] Plan 2: RAG + Citations + Grounding
- [ ] Plan 3: HITL + Daily Reports + Observability
- [ ] Plan 4: Eval Harness + Red Team + CI/CD + Bonuses
