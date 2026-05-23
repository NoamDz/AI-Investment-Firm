# AI Investment Firm

Multi-agent paper-trading firm. Take-home for Cato Networks — Agentic AI Engineer.

[![PR CI](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/pr.yml/badge.svg)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/pr.yml)
[![Main CI](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/main.yml)
[![Release](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/release.yml/badge.svg)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/release.yml)

A LangGraph orchestrator drives `research → PM → risk → (HITL) → execution → reporter`
agents per heartbeat. Research is grounded via the Anthropic Citations API; a Haiku
sufficiency judge gates whether evidence supports the claim; three PM lenses
(quality / valuation / catalyst) vote in parallel; the risk node escalates to a
human-in-the-loop queue when policy thresholds trip; the reporter writes a daily
markdown + xlsx bundle and reconciles broker books.

## Prerequisites

- Python 3.11.x (3.13 ships without `torch.SymInt`; 3.10 lacks newer typing)
- Docker Desktop
- Anthropic API key (`ANTHROPIC_API_KEY`)
- *Optional:* CUDA GPU for faster corpus ingest

## Quickstart

```powershell
copy .env.example .env                   # then set ANTHROPIC_API_KEY
docker compose up -d qdrant              # vector store
python -m firm.cli ingest                # one-time corpus embed (~2 min, 20 docs)
docker compose up firm                   # one heartbeat → REFUSE or BUY → daily report
```

Full step-by-step (host venv setup, GPU notes, HITL exercise, Alpaca, native run):
[`docs/quickstart.md`](docs/quickstart.md).

## Daily report

```powershell
make report DATE=2024-03-13
```

Writes `data/reports/2024-03-13/daily_report.md` (decision histogram, cost
summary, EOD reconciliation) and `positions.xlsx`.

## Eval & red-team

```powershell
make eval                                # 3-regime deterministic sweep, ~5 min from cassettes
python -m firm.cli red-team              # 51-case adversarial corpus, 10 invariant suites
```

Both run from cassettes — no API key needed. See [`docs/eval.md`](docs/eval.md)
for metrics + regime design.

## Deployment

AWS infrastructure is sketched in `infra/terraform/` (6 modules; sanitised
`PLAN.md` is committed). The Reporter agent also runs on Bedrock AgentCore's
local runtime via `firm/agentcore/reporter_adapter.py`.

```powershell
# HUMAN-GATED — creates real AWS resources (~$60/mo idle). Prompts for 'DEPLOY'.
make deploy-dev
```

See [`docs/path-to-production.md`](docs/path-to-production.md) for the
take-home → prod delta and [`docs/agentcore_mapping.md`](docs/agentcore_mapping.md)
for the AgentCore migration table.

## Where to look next

| Doc | What's in it |
|-----|--------------|
| [`docs/quickstart.md`](docs/quickstart.md) | Full host+Docker setup, HITL flow, Slack, Alpaca, native run |
| [`docs/runbook.md`](docs/runbook.md) | Operator procedures: Slack approval, restore-from-Litestream, Qdrant backup |
| [`docs/eval.md`](docs/eval.md) | Eval harness design — metrics, regimes, "not measured" list |
| [`docs/threat_model.md`](docs/threat_model.md) | STRIDE-style threat model + red-team coverage |
| [`docs/path-to-production.md`](docs/path-to-production.md) | Take-home → prod delta map |
| [`docs/agentcore_mapping.md`](docs/agentcore_mapping.md) | Firm-to-AgentCore migration table |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | Test layout, `requires_models` marker, re-recording cassettes |
| [`docs/implementation_summary_plan_1.md`](docs/implementation_summary_plan_1.md), [`_2`](docs/implementation_summary_plan_2.md) | Per-plan implementation notes |

## Status

- [x] Plan 1: Foundation + Walking Skeleton
- [x] Plan 2: RAG + Citations + Grounding
- [x] Plan 3: HITL + Daily Reports + Observability
- [x] Plan 4: Eval Harness + Red Team + CI/CD + Bonuses
