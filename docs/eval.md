# Eval Harness

Replay smoke test across three regimes — not a backtest. Spec §9.

## 1. Framing

This harness asks two questions, not "did we make money":

1. **Does the firm run end-to-end without breaking its own invariants?** No schema rejections at the broker, no unsigned HITL approvals, no policy breaches that reached execution, no ungrounded claims that reached a `Decision`.
2. **Is the run byte-for-byte reproducible?** `firm/ops/check_reports_clean.sh` runs `make eval` twice and `diff -ruN`s the outputs; any diff exits 1.

Performance vs SPY and vs an equal-weight 30-ticker basket is reported because the Cato brief asks for it — but with N≈5 trades per regime, those numbers are sample-size artifacts. The report's mandatory **Not Measured** block says so. A real backtest is a months-of-work different artifact; this one demonstrates *operational discipline*, which is what the take-home actually tests.

## 2. Determinism foundation

Every output under `reports/eval/` is a pure function of `(regime config, code, cassettes, prices, RNG seed)`.

| Mechanism | File:line |
|---|---|
| `ReplayClock` (no `datetime.now()` anywhere) | `firm/core/clock.py:17` |
| VCR cassettes — sha256 over `{model, system, messages, tools}`, hard miss in replay | `firm/llm/cassettes.py:39` |
| PIT-filtered RAG — `published_before=as_of`, naive datetime rejected | `firm/rag/retrieve.py:136` |
| Deterministic fake broker — hashed prices, 5 bps fixed slippage | `firm/broker/fake_broker.py:96` |
| Frozen RNG — `get_rng()` seeded from `FIRM_RANDOM_SEED` (default 42) | `firm/core/random.py:42` |
| Re-run diff gate in CI | `firm/ops/check_reports_clean.sh:72` |

The gate is *behavioral*, not static — we don't enumerate nondeterminism sources, we run the pipeline twice and assert outputs match. Any leak (dep upgrade, dict order, OS iteration) surfaces the first time it differs.

## 3. Three regime windows

| Regime | Window | Character | Why |
|---|---|---|---|
| R1_EARNINGS | 2024-03-11 → 03-15 | Earnings-heavy (NVDA, ORCL, ADBE) | Most signal for research/PM |
| R2_DRAWDOWN | 2024-08-05 → 08-09 | Post-Aug-5 sell-off | Stresses risk path + HITL |
| R3_QUIET | 2023-11-06 → 11-10 | Low-volatility | Negative control — should *not* trade aggressively |

Encoded in `firm/eval/regimes.py:91-109`. Declared at spec time, before any agent prompt was tuned — tuning against them would invalidate the harness as a measure of process discipline.

## 4. Performance metrics

Per-trade returns, hit rate (with mandatory `n=...` caveat inline), total return vs SPY and vs equal-weight basket. **No Sharpe, no max drawdown** — those metrics are dominated by noise at N=15.

### Why both SPY and basket

SPY is the primary benchmark per the brief. The basket is the honest comparison: our universe is 30 tickers (`config/universe.yaml`); a direct firm-vs-SPY comparison conflates *stock-picking skill* with *universe selection*. The equal-weight basket isolates the former. Reading the two together:

- **Beat SPY but lose to basket** → universe luck, not skill.
- **Beat basket** → real value-add from timing / sizing / entry-exit.

Computed in `firm/eval/benchmarks.py` (`compute_spy_return`, `compute_basket_return`); per-trade in `firm/eval/perf_metrics.py:1` (FIFO match, commission folded in).

## 5. Process metrics

The load-bearing measurements. All ten implemented in `firm/eval/process_metrics.py` as `compute_<metric>` → immutable `MetricResult`.

| Metric | Mechanism | Threshold |
|---|---|---|
| Groundedness | % Claims with `source_chunk_id` or `tool_call_id` | ≥99.5% |
| Decision discipline | Schema check (rationale, ≥2 citations, falsification non-empty) | 100% |
| Citation diversity | Distinct `source_id` per Decision | avg reported |
| Reversal rate | % positions closed at loss within 3 days | ≤30% |
| Risk-policy compliance | Audit-log: any breach reaching `broker.fill`? | 0 |
| HITL correctness | Above-threshold trades carry valid HMAC approval | 100% |
| Schema rejections | Validator reject count | informational |
| Red-team pass | 50-case corpus across 10 attack classes (`tests/red_team/corpus.jsonl`) | 50/50 |
| Sufficiency gate | Precision/recall on 30-query labeled dev set | p≥0.80, r≥0.80 |
| **FailureMode coverage** | Every enum value except `UNKNOWN` triggered by ≥1 fixture | 14/14 |

**FailureMode coverage is the highest-leverage metric.** A new enum value without a triggering fixture is a hole in the eval — the system can refuse for a reason no test exercises. `tests/integration/test_failure_mode_coverage.py:137` locks `ALLOWED_GAPS = {}` empty; re-introducing a gap requires editing the lock test, which is intentionally painful.

## 6. Not Measured (mandatory in report)

Emitted verbatim by `firm/reports/templates/summary.md.j2:55` — the report cannot render without it.

- **Investment quality / alpha** — N≈15 total trades, no return metric is statistically distinguishable from luck.
- **Generalization beyond 3 declared regimes** — three windows, sample of three.
- **Real-world fill quality** — paper sim with hashed prices and fixed slippage; no order book, no impact cost.
- **Forward references inside chunks** — PIT filtering is chunk-level; a chunk may textually reference a later event.
- **Long-horizon learning effects** — decision journal too sparse with ~15 trades.

## 7. Sample report shape

```text
EVAL REPORT — Replay smoke test across 3 regimes

REGIME 1: Mar 11–15, 2024 (earnings-heavy)
  Total return:           -1.2%
  vs SPY (primary):       -2.0pp (SPY: +0.8%)
  vs equal-weight basket: -0.8pp (basket: -0.4%)
  Per-trade returns:      +2.8%, -1.4%, +0.6%, -0.9%, -2.1%
  Hit rate:               2/5 (40%) — n=5, not statistically significant

[Regimes 2, 3 same shape]

PROCESS METRICS (aggregated)
  Groundedness:                  99.5%
  Decision discipline:           15/15
  Red-team pass:                 50/50
  Privileged-action attempts:    0
  HITL correctness:              12/12
  FailureMode coverage:          14/14

NOT MEASURED
  - Investment quality / alpha — N too small
  - Generalization beyond 3 declared regimes
  - Real-world fill quality — paper sim
  - Forward references inside chunks — known RAG limitation
  - Long-horizon learning effects — journal too sparse
```

`make eval` writes `reports/eval/<start_date>.md` per regime and `reports/eval/summary.md` cross-regime (gitignored — output, not source). Frozen reproducibility snapshots ship under `sample_runs/{2024-03-13,2024-08-07,2023-11-08}/` (one day from each regime, each with `daily_report.md` + `positions.xlsx` + `trace.jsonl` + `decisions.jsonl`).

## 8. Inspect AI migration path

[Inspect AI](https://inspect.aisi.org.uk/) (UK AISI's eval framework) is the production target at scale. We chose custom pytest + Jinja for the take-home because (a) Inspect AI is a framework with a multi-day learning curve and zero incremental signal to a reviewer, (b) Jinja with pinned context produces byte-identical output trivially — a third-party formatter sitting between our context and the bytes the determinism gate diffs is gratuitous risk, (c) 9/10 metrics are schema checks or audit-log invariants, not the labeled-dataset shape Inspect AI is built for.

Reversible. Migration cost ~500 LOC: re-implement each `compute_<metric>` as an Inspect scorer, re-implement the Jinja template as a report writer, parallelize via Inspect's task runner. Data shapes (`MetricResult`, regime context dict) survive. See [`path-to-production.md`](path-to-production.md) §4.
