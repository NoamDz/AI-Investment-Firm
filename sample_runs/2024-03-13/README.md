# Sample run - 2024-03-13

## What this day demonstrates

- **Regime:** earnings_heavy
- **Setup:** Opens with BUY AAPL x 100; earnings_heavy regime.
- **What to look for:** Highest-confidence BUY is `dec-buy-1` (0.85).

## Decisions

| ts | action | ticker | shares | conf | citations | failure_mode | rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-03-13T14:30:00+00:00 | BUY | AAPL | 100 | 0.85 | 4 |  | Strong earnings momentum |
| 2024-03-13T16:00:00+00:00 | HOLD |  |  | 0.60 | 3 |  | No clear edge |

## Walking one trade

The most-cited decision this day is `dec-buy-1`. Reproduce its full chain with:

```bash
grep '"decision_id":"dec-buy-1"' sample_runs/2024-03-13/trace.jsonl | jq .
```

```jsonl
{"trace_id": "00000000000000000000000000000001", "span_id": "a1b2c3d4e5f60001", "parent_span_id": null, "agent": "research", "operation": "agent.research", "decision_id": "dec-buy-1", "duration_ms": 1842.31, "model": "", "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "citations": 4, "failure_mode": "", "status": "ok"}
{"trace_id": "00000000000000000000000000000001", "span_id": "a1b2c3d4e5f60002", "parent_span_id": "a1b2c3d4e5f60001", "agent": "", "operation": "retrieval.hybrid", "decision_id": "dec-buy-1", "duration_ms": 38.7, "model": "", "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "00000000000000000000000000000001", "span_id": "a1b2c3d4e5f60003", "parent_span_id": "a1b2c3d4e5f60002", "agent": "", "operation": "retrieval.rerank", "decision_id": "dec-buy-1", "duration_ms": 22.4, "model": "bge-reranker-v2-m3", "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "00000000000000000000000000000001", "span_id": "a1b2c3d4e5f60004", "parent_span_id": "a1b2c3d4e5f60001", "agent": "", "operation": "llm.call", "decision_id": "dec-buy-1", "duration_ms": 1672.5, "model": "claude-sonnet-4-6", "input_tokens": 3120, "output_tokens": 482, "cached_tokens": 0, "cost_usd": 0.01658, "citations": 4, "failure_mode": "", "status": "ok"}
{"trace_id": "00000000000000000000000000000001", "span_id": "a1b2c3d4e5f60010", "parent_span_id": null, "agent": "pm", "operation": "agent.pm", "decision_id": "dec-buy-1", "duration_ms": 1289.6, "model": "", "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "00000000000000000000000000000001", "span_id": "a1b2c3d4e5f60011", "parent_span_id": "a1b2c3d4e5f60010", "agent": "", "operation": "llm.call", "decision_id": "dec-buy-1", "duration_ms": 1240.1, "model": "claude-sonnet-4-6", "input_tokens": 1860, "output_tokens": 312, "cached_tokens": 0, "cost_usd": 0.01026, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "00000000000000000000000000000001", "span_id": "a1b2c3d4e5f60012", "parent_span_id": "a1b2c3d4e5f60010", "agent": "", "operation": "llm.call", "decision_id": "dec-buy-1", "duration_ms": 1198.4, "model": "claude-sonnet-4-6", "input_tokens": 1862, "output_tokens": 298, "cached_tokens": 0, "cost_usd": 0.01005, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "00000000000000000000000000000001", "span_id": "a1b2c3d4e5f60013", "parent_span_id": "a1b2c3d4e5f60010", "agent": "", "operation": "llm.call", "decision_id": "dec-buy-1", "duration_ms": 1265.0, "model": "claude-sonnet-4-6", "input_tokens": 1861, "output_tokens": 305, "cached_tokens": 0, "cost_usd": 0.01015, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "00000000000000000000000000000001", "span_id": "a1b2c3d4e5f60020", "parent_span_id": null, "agent": "risk", "operation": "agent.risk", "decision_id": "dec-buy-1", "duration_ms": 412.8, "model": "", "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "cost_usd": 0.0, "citations": 0, "failure_mode": "", "status": "ok"}
{"trace_id": "00000000000000000000000000000001", "span_id": "a1b2c3d4e5f60021", "parent_span_id": "a1b2c3d4e5f60020", "agent": "", "operation": "llm.call", "decision_id": "dec-buy-1", "duration_ms": 386.2, "model": "claude-haiku-4-5-20251001", "input_tokens": 980, "output_tokens": 124, "cached_tokens": 0, "cost_usd": 0.000893, "citations": 0, "failure_mode": "", "status": "ok"}
```

## Bundle

- [`daily_report.md`](daily_report.md) - legacy plain-text summary
- [`daily_report.html`](daily_report.html) - rendered report (open in browser)
- [`positions.xlsx`](positions.xlsx) - Positions / P&L / Decisions sheets
- [`decisions.jsonl`](decisions.jsonl) - raw decisions
- [`trace.jsonl`](trace.jsonl) - raw spans
- [`dashboard.png`](dashboard.png) - Tab 1 (Today's Report) screenshot
