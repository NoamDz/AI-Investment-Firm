# Sample run - 2024-08-07

## What this day demonstrates

- **Regime:** vol_spike
- **Setup:** Opens flat; vol_spike regime - no immediate trade.
- **What to look for:** ESCALATE row triggered by `risk_limit_breached` - see decision `dec-escalate-1`.

## Decisions

| ts | action | ticker | shares | conf | citations | failure_mode | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-08-07T13:45:00+00:00 | SELL | NVDA | 40 | 0.78 | 5 |  | De-risking semis into sell-off; momentum break confirmed |
| 2024-08-07T15:20:00+00:00 | ESCALATE |  |  | 0.55 | 4 | risk_limit_breached | Risk gate flagged size; routing to HITL |

## Walking one trade

The most-cited decision this day is `dec-sell-1`. Reproduce its full chain with:

```bash
grep '"decision_id":"dec-sell-1"' sample_runs/2024-08-07/trace.jsonl | jq .
```

```jsonl
{"trace_id": "20240807000000000000000000000001", "span_id": "c3d4e5f607080001", "parent_span_id": null, "agent": "research", "operation": "agent.research", "decision_id": "dec-sell-1", "duration_ms": 1956.4, "model": "", "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "citations": 5, "failure_mode": "", "status": "ok"}
{"trace_id": "20240807000000000000000000000001", "span_id": "c3d4e5f607080002", "parent_span_id": "c3d4e5f607080001", "agent": "", "operation": "retrieval.hybrid", "decision_id": "dec-sell-1", "duration_ms": 41.2, "model": "", "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "20240807000000000000000000000001", "span_id": "c3d4e5f607080003", "parent_span_id": "c3d4e5f607080002", "agent": "", "operation": "retrieval.rerank", "decision_id": "dec-sell-1", "duration_ms": 24.8, "model": "bge-reranker-v2-m3", "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "20240807000000000000000000000001", "span_id": "c3d4e5f607080004", "parent_span_id": "c3d4e5f607080001", "agent": "", "operation": "llm.call", "decision_id": "dec-sell-1", "duration_ms": 1789.3, "model": "claude-sonnet-4-6", "input_tokens": 3402, "output_tokens": 521, "cached_tokens": 0, "cost_usd": 0.01807, "citations": 5, "failure_mode": "", "status": "ok"}
{"trace_id": "20240807000000000000000000000001", "span_id": "c3d4e5f607080010", "parent_span_id": null, "agent": "pm", "operation": "agent.pm", "decision_id": "dec-sell-1", "duration_ms": 1402.1, "model": "", "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "20240807000000000000000000000001", "span_id": "c3d4e5f607080011", "parent_span_id": "c3d4e5f607080010", "agent": "", "operation": "llm.call", "decision_id": "dec-sell-1", "duration_ms": 1332.6, "model": "claude-sonnet-4-6", "input_tokens": 1942, "output_tokens": 328, "cached_tokens": 0, "cost_usd": 0.01088, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "20240807000000000000000000000001", "span_id": "c3d4e5f607080012", "parent_span_id": "c3d4e5f607080010", "agent": "", "operation": "llm.call", "decision_id": "dec-sell-1", "duration_ms": 1308.7, "model": "claude-sonnet-4-6", "input_tokens": 1944, "output_tokens": 311, "cached_tokens": 0, "cost_usd": 0.01060, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "20240807000000000000000000000001", "span_id": "c3d4e5f607080013", "parent_span_id": "c3d4e5f607080010", "agent": "", "operation": "llm.call", "decision_id": "dec-sell-1", "duration_ms": 1351.4, "model": "claude-sonnet-4-6", "input_tokens": 1943, "output_tokens": 319, "cached_tokens": 0, "cost_usd": 0.01076, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "20240807000000000000000000000001", "span_id": "c3d4e5f607080020", "parent_span_id": null, "agent": "risk", "operation": "agent.risk", "decision_id": "dec-sell-1", "duration_ms": 438.9, "model": "", "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "20240807000000000000000000000001", "span_id": "c3d4e5f607080021", "parent_span_id": "c3d4e5f607080020", "agent": "", "operation": "llm.call", "decision_id": "dec-sell-1", "duration_ms": 411.5, "model": "claude-haiku-4-5-20251001", "input_tokens": 1020, "output_tokens": 138, "cached_tokens": 0, "cost_usd": 0.000932, "citations": 0, "failure_mode": "", "status": "ok"}
```

## Bundle

- [`daily_report.md`](daily_report.md) - legacy plain-text summary
- [`daily_report.html`](daily_report.html) - rendered report (open in browser)
- [`positions.xlsx`](positions.xlsx) - Positions / P&L / Decisions sheets
- [`decisions.jsonl`](decisions.jsonl) - raw decisions
- [`trace.jsonl`](trace.jsonl) - raw spans
- [`dashboard.png`](dashboard.png) - Tab 1 (Today's Report) screenshot
