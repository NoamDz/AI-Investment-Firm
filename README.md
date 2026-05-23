# AI Investment Firm

Multi-agent paper-trading firm. Take-home for Cato Networks — Agentic AI Engineer.

[![PR CI](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/pr.yml/badge.svg)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/pr.yml)
[![Main CI](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/main.yml)
[![Release](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/release.yml/badge.svg)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/release.yml)

## Overview

A **LangGraph** state machine drives one heartbeat through seven nodes:
`monitor → research → PM → risk → (HITL) → execution → reporter`. The graph
is persisted with a SQLite checkpointer, so an interrupt (HITL waiting on
approval) resumes from the same checkpoint on the next tick instead of
restarting. `--once` runs one heartbeat; `--loop --interval-seconds N` runs
continuously until SIGINT.

The design is opinionated about three failure modes that ungrounded LLM
trading systems hit:

- **Hallucinated evidence.** Research uses the **Anthropic Citations API**
  against a Qdrant index of FinanceBench 10-Ks (BM25 + dense Nomic + BGE-v2-m3
  rerank). A **Haiku sufficiency judge** then re-reads the same retrieved
  passages and rejects any claim the evidence does not actually support —
  yielding a typed `INSUFFICIENT_EVIDENCE` / `UNCITED_CLAIM` REFUSE rather
  than a confident wrong answer.
- **Single-LLM groupthink.** The PM node fans out **three lenses in parallel**
  via a LangGraph `Send` — *quality*, *valuation*, *catalyst* — and requires
  a majority vote to advance.
- **Silent risk-policy bypass.** The risk node runs deterministic gates
  (max gross / net exposure, per-name and per-sector limits, drawdown,
  quote-age). ESCALATE triggers `interrupt_before=["hitl"]`; a human approves
  in Slack or via `make dev-ack`; a reaper closes stale items as
  `UNAPPROVED_HIGH_RISK` after the timeout.

Every LLM call goes through a **cost router** (`config/router.yaml`,
fallback `sonnet → haiku`) and a **content-addressed SQLite cache**, so under
`FIRM_LLM_MODE=cached` the entire pipeline is byte-deterministic and the
eval / red-team suites run offline. Failures are typed (`FailureMode`
enum, 15 values with `UNKNOWN` catch-all); every value has a triggering
fixture in `tests/eval/failure_modes/`. Execution is idempotent — fills are
keyed by HMAC nonce and reconciled against an audit log with chained
parent IDs.

**Output channels (deliverable §6):** a **Streamlit web dashboard**
(`firm/dashboard.py`) is the primary live view — positions, recent
decisions, HITL queue, cost ledger, reconciliation status, auto-refresh.
A daily **`positions.xlsx`** sheet is the second channel for operators who
pivot in Excel. The two are complementary: the dashboard is a human-read
real-time view; the xlsx is a portable snapshot for downstream tooling.
Both are produced from the same `firm.db` source of truth, so they cannot
drift.

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

Continuous demo (the way a reviewer would run it live) — two terminals:

```powershell
# Terminal 1 — the firm loop
python -m firm.cli run --loop --interval-seconds 60     # Ctrl-C to stop

# Terminal 2 — the live dashboard
pip install -e ".[dashboard]"
streamlit run firm/dashboard.py                          # http://localhost:8501
```

Full step-by-step (host venv setup, GPU notes, HITL exercise, Alpaca, native run):
[`docs/quickstart.md`](docs/quickstart.md).

## Daily report

```powershell
make report DATE=2024-03-13
```

Writes `data/reports/2024-03-13/positions.xlsx` (channel #2) and the
backing `daily_report.md` (decision histogram, cost summary, EOD
reconciliation). The dashboard reads the same `firm.db` continuously.

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
| [`docs/architecture.md`](docs/architecture.md) | Logical agent flow + deployment view (Mermaid) |
| [`docs/technical-overview.md`](docs/technical-overview.md) | Agent contracts, state flow, partial-failure model |
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
