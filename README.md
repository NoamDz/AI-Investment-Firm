# AI Investment Firm

Multi-agent paper-trading firm. Take-home for Cato Networks — Agentic AI Engineer.

[![PR CI](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/pr.yml/badge.svg)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/pr.yml)
[![Main CI](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/main.yml/badge.svg?branch=main)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/main.yml)
[![Release](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/release.yml/badge.svg)](https://github.com/NoamDz/AI-Investment-Firm/actions/workflows/release.yml)

## Overview

This repo is a small AI-run trading desk that you can spin up with one
`docker compose up`. Seven agents take turns: read the market, pick a
trade, debate it, check the rules, ask a human if the trade is large,
place the order with a paper broker, and write the day's report. The
whole desk runs on a one-minute "heartbeat" loop, all the state lives in
a single SQLite file, and every prompt is cached so you never pay for the
same call twice.

The interesting part is *how* the agents collaborate — not just a chain
of LLM calls, but a back-and-forth where each step can stop, refuse, or
escalate the trade. A trade only reaches the broker after research has
grounded every claim against a real SEC filing, a sufficiency judge has
agreed the quotes actually support the claims, three independent
"portfolio manager" voters have agreed it's worth doing, a deterministic
risk check has cleared every limit, and (for large trades) a human has
approved it in Slack.

The pieces that come with the system:

- **7 agents** in a fixed pipeline, orchestrated by LangGraph
- A **grounded research path** built on the Anthropic Citations API and a
  Qdrant index of SEC 10-Ks (hybrid retrieval: keyword + vector + reranker)
- The **3-voter PM** described below
- Risk limits enforced by plain Python (an LLM cannot argue its way past them)
- A **real human-in-the-loop pause** that survives a process restart
- A **reproducible eval** over three historical regimes — runs offline
  from cassettes, byte-identical run-to-run
- A **51-case red-team** suite covering 10 attack families
- **Streamlit dashboard** + **daily Excel sheet** as the two report channels
- Full **observability**: every agent call, every LLM call, every fill
  leaves an OpenTelemetry span with the decision ID attached

## How the system works

### The seven agents

Each agent does one thing and hands off to the next:

1. **monitor** — reads the clock and the list of tickers the firm is
   allowed to trade.
2. **research** — picks one candidate trade and writes its thesis as a
   list of *claims*. Every claim must quote a passage from a real filing
   (Qdrant retrieves the candidate passages; the Anthropic Citations API
   returns the exact verbatim quote, which is stored on the claim — never
   paraphrased).
3. **PM (the interesting one)** — not a single model, but three voters
   running in parallel:
   - a **quality voter** asks "is this a good business?"
   - a **valuation voter** asks "is the price reasonable?"
   - a **catalyst voter** asks "is there a near-term reason to act?"

   Each one votes BUY / HOLD / VETO independently. A simple majority is
   required to advance the trade. The point is to stop a single LLM's
   bad day — or a single prompt-injection attempt — from carrying a
   trade to the floor.
4. **risk** — runs the rule book in plain Python (not an LLM). Max 10%
   in one name, 30% per sector, gross book capped at 100% of capital,
   max daily loss 3%, quote can't be older than 60 seconds, etc. Trades
   that bust a hard limit are refused outright. Trades that are
   "approvable but big" are routed to the human instead of straight to
   the broker.
5. **HITL** — pauses the workflow and posts a signed Slack message
   ("approve / edit / reject?"). The pause is *real* — the LangGraph
   checkpoint is persisted to disk, so you can stop the process, walk
   away, come back tomorrow, and the trade is still waiting. If nobody
   answers in 30 minutes, a background reaper auto-refuses it.
6. **execution** — places the order with the simulated broker (5 bps
   slippage + per-share commission, so a buy at $100 actually settles at
   $100.05 plus fees). Fills are idempotent — the same nonce never
   settles twice, so a network retry is safe.
7. **reporter** — writes the day's Markdown report and refreshes the
   dashboard's view.

### The back-and-forth

A heartbeat isn't just a one-way march through the seven nodes — agents
push back on each other. Two loops are worth knowing:

- **Research ↔ sufficiency judge.** After research extracts its claims
  with citations, a smaller, cheaper model (Haiku) re-reads the exact
  same passages and labels each claim *ok*, *partial*, or *insufficient*.
  Too many *insufficient* labels and the whole proposal is killed before
  the PM ever sees it. This is the system's main defence against
  hallucinated facts — every claim is verified by an independent reader
  against the source it cites.

- **Risk ↔ HITL ↔ execution.** Risk picks one of three outcomes:
  *approve* (→ execution), *refuse* (heartbeat ends), or *escalate*
  (→ HITL). When the human approves, the graph resumes from the same
  checkpoint and the trade flows to execution. If the human *edits* the
  size, the new size goes through the same risk check again on the next
  tick — the human can't shortcut the rules, only the threshold for
  escalation.

State flows through one shared `firm.db`: positions, cash, every
decision, the cost ledger, the HITL queue, and the LangGraph checkpoint
all share the same SQLite file. A crash mid-trade resumes mid-trade from
the same source of truth that the broker reconciliation reads at boot.

### How the system handles things going wrong

Every failure has a name. There are 15 of them in a `FailureMode` enum,
with a catch-all `UNKNOWN` so nothing slips through silently:

- Broker is unreachable → `BROKER_UNAVAILABLE`
- Anthropic API errors → the cost router falls through to a smaller
  model; if every hop fails → `LLM_UNAVAILABLE`
- Last price quote is too old → `STALE_DATA`
- Retrieved text contains a `<system>` tag → `PROMPT_INJECTION_DETECTED`
- LLM cites a claim ID it never extracted → `UNCITED_CLAIM`
- Sufficiency judge rejects too many quotes → `INSUFFICIENT_EVIDENCE`
- Human ignores Slack for 30 min → `UNAPPROVED_HIGH_RISK`
- Risk gate refuses a trade → `RISK_LIMIT_BREACHED`
- End-of-day books don't tie to the broker → `RECONCILIATION_DRIFT`

Each of the 15 has a coverage test in `tests/integration/`. The red-team
suite on top of that (51 adversarial cases — citation forgery, role
hijack, confused-deputy, unicode homoglyphs, spoofed approvals,
multi-step chains) proves each guardrail fires when it should and stays
quiet when it shouldn't.

### Cost, caching, and reproducibility

Every LLM call goes through a **cost router** that picks the cheapest
model that can do the job (Haiku first, fall through to Sonnet on
overload or schema errors) and a **prompt cache** keyed by a content
hash of the system prompt + messages + tools. The same prompt is never
billed twice. A `cost_ledger` row records the model, tokens, and USD
spent per decision, so you can answer "what did this trade cost?" for
any decision in the log.

In `FIRM_LLM_MODE=cached` the entire pipeline is byte-deterministic:
`make eval` replays three historical regimes from recorded prices and
recorded LLM responses, no API key needed. A `check_reports_clean.sh`
helper runs eval twice and `diff`s the output — if any source of
randomness leaks in, CI catches it.

### Observability — replay one trade from the trace

Every agent call, every LLM call, every tool call, and every retrieval
emits an OpenTelemetry span tagged with the decision ID, the parent
decision it came from, the failure mode (if any), the model used, the
tokens spent, and the USD cost. Spans are written one-per-line to
`data/traces/<date>.jsonl`. A reviewer can `grep` one decision ID and
see the whole heartbeat — research → vote → risk check → fill — without
standing up any infrastructure. In production the same tracer ships the
same spans to a real OTLP backend.

### Two report channels

Both read the same `firm.db`, so they cannot disagree:

- **Streamlit dashboard** — live view of positions, recent decisions, the
  HITL queue, today's cost, reconciliation status. Auto-refreshes while
  the firm runs.
- **Daily `positions.xlsx`** — a spreadsheet for operators who pivot in
  Excel, written by the reporter at end-of-day.

## Prerequisites

- Python 3.11.x (3.13 ships without `torch.SymInt`; 3.10 lacks newer typing)
- Docker Desktop
- Anthropic API key (`ANTHROPIC_API_KEY`)
- *Optional:* CUDA GPU for faster corpus ingest

## Quickstart

```powershell
copy .env.example .env                   # then set ANTHROPIC_API_KEY
docker compose up -d qdrant              # vector store
python -m firm.cli ingest                # one-time corpus embed (~2 min)
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

Full step-by-step (host venv setup, GPU notes, HITL exercise, Alpaca,
native run): [`docs/quickstart.md`](docs/quickstart.md).

## Status

- [x] Plan 1: Foundation + Walking Skeleton
- [x] Plan 2: RAG + Citations + Grounding
- [x] Plan 3: HITL + Daily Reports + Observability
- [x] Plan 4: Eval Harness + Red Team + CI/CD + Bonuses
