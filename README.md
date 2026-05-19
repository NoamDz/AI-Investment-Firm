# AI Investment Firm

Multi-agent paper-trading firm. Take-home for Cato Networks — Agentic AI Engineer.

## Quickstart (clone-to-demo in <10 min)

```bash
git clone <repo>
cd ai-investment-firm
pip install -e ".[dev]"
make demo
```

Output: one heartbeat through the 5-agent workflow, one paper trade via FakeBroker, and a JSONL report in `data/reports/2024-03-13/`.

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
- [ ] Plan 2: RAG + Citations + Grounding
- [ ] Plan 3: HITL + Daily Reports + Observability
- [ ] Plan 4: Eval Harness + Red Team + CI/CD + Bonuses
