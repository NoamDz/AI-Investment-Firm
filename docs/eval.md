# Eval Harness

A reproducible replay of the firm against three historical 5-day windows. **Not a backtest** — see "What this measures" below.

## How to run

```bash
make eval                                                       # all 3 regimes, cached LLM, replayed prices
python -m firm.cli eval --regime r1_earnings --mode cached      # one regime
```

What appears on disk:

- `reports/eval/<regime_id>/summary.md` — per-regime numbers + the mandatory "Not Measured" block
- `reports/eval/<regime_id>/<date>/daily_report.md` + `daily_report.html` + `positions.xlsx` + `decisions.jsonl` + `trace.jsonl` — one bundle per replayed trading day

One day from each regime is checked in under `sample_runs/` so a reviewer can read a real bundle without running anything.

## What this measures

Two questions, in order of importance:

1. **Does the firm run end-to-end without breaking its own invariants?** No schema rejections at the broker, no unsigned human-approvals, no policy breaches that reached execution, no ungrounded claims that reached a Decision.
2. **Is the run byte-for-byte reproducible?** `firm/ops/check_reports_clean.sh` runs `make eval` twice and `diff -ruN`s the outputs. Any diff exits 1, which fails CI.

Performance vs SPY *is* reported (the brief asks for it), but with ~15 trades total the numbers are sample-size artifacts. The harness measures **operational discipline**, not stock-picking skill. The report says so in a mandatory "Not Measured" block — the Jinja template literally won't render without it.

## How a single regime runs

For each regime the harness does this, in order:

1. **Load the regime config** — start/end dates, universe of tickers, RNG seed. `firm/eval/regimes.py`.
2. **Freeze the clock.** Every `datetime.now()` in the firm goes through `ReplayClock`, so the firm believes it is the regime's date and the same day produces the same output. `firm/core/clock.py:17`.
3. **Replay prices.** When the firm asks for a quote, the parquet file at `data/eval/prices/<TICKER>.parquet` returns the adjusted close for the frozen date. In `replay` mode the network is never touched; a missing parquet is a hard error. `firm/eval/benchmarks.py`.
4. **Replay LLM responses.** Every Anthropic call hits the VCR layer; the cassette is looked up by `SHA256(model, system, messages, tools)`. A miss is a hard error in cached mode. `firm/llm/cassettes.py:39`.
5. **Run the heartbeats.** `firm/eval/runner.py` invokes the same LangGraph the live firm uses. The risk gate, the sufficiency judge, the human-approval pause, the reporter — every production path runs.
6. **Compute metrics.** Ten process metrics + portfolio benchmarks run against the database. `firm/eval/process_metrics.py`, `firm/eval/perf_metrics.py`.
7. **Render the report.** `firm/reports/templates/summary.md.j2` fills in the numbers and adds the "Not Measured" block.

Everything under `reports/eval/` is a pure function of `(regime config, code, cassettes, prices, RNG seed)`. Change any of those five inputs and the output is allowed to change; touch nothing and it must not.

## The three regimes

Three windows declared in `firm/eval/regimes.py:91-116`. They were named *before* any agent prompt was tuned — tuning against them would invalidate the harness as a measure of process discipline.

| ID | Window | Character | Why included |
|---|---|---|---|
| `r1_earnings` | 2024-03-11 → 03-15 | NVDA, ORCL, ADBE all report | Most signal for the research and PM agents |
| `r2_drawdown` | 2024-08-05 → 08-09 | Post-Aug-5 sell-off | Stresses the risk gate and human-approval path |
| `r3_quiet` | 2023-11-06 → 11-10 | Low-volatility quiet | Negative control — the firm should *not* trade aggressively |

## What gets measured

### Portfolio performance (informational)

| Metric | Source |
|---|---|
| Per-trade return, FIFO-matched, commission folded in | `firm/eval/perf_metrics.py` |
| Hit rate (with `n=...` caveat printed inline) | same |
| Total return vs SPY | `firm/eval/benchmarks.py:compute_spy_return` |
| Total return vs equal-weight basket of the universe | `firm/eval/benchmarks.py:compute_basket_return` |

No Sharpe, no max-drawdown — those need `N >> 15` to mean anything. **Why both SPY and basket:** SPY is the brief's benchmark. The basket isolates stock-picking from universe selection — beat SPY but lose to basket → you got lucky in your tickers; beat basket → you actually picked well.

### Process quality — the load-bearing measurements

Ten metrics in `firm/eval/process_metrics.py`, each returning an immutable `MetricResult`:

| Metric | What it asks | Threshold |
|---|---|---|
| **Groundedness** | What % of claims carry a `source_chunk_id` or `tool_call_id`? | ≥99.5% |
| **Decision discipline** | Does every Decision have a rationale, ≥2 citations, and a non-empty falsification clause? | 100% |
| **Citation diversity** | How many distinct sources per Decision? | Average reported |
| **Reversal rate** | What % of positions are closed at a loss within 3 days of opening? | ≤30% |
| **Risk-policy compliance** | Did any policy breach reach the broker? | 0 |
| **HITL correctness** | Do all above-threshold trades carry a valid HMAC approval? | 100% |
| **Schema rejections** | How many agent outputs failed Pydantic validation? | Informational |
| **Red-team pass** | Does the 51-case adversarial corpus refuse / escalate as expected? | 51/51 |
| **Sufficiency gate** | Precision/recall on a 30-query labeled dev set | p≥0.80, r≥0.80 |
| **FailureMode coverage** | Has every FailureMode enum value (except UNKNOWN) been triggered by ≥1 fixture? | 14/14 |

The last one is the highest-leverage. A new enum value added without a triggering fixture is a hole — the system could refuse for a reason no test exercises. `tests/integration/test_failure_mode_coverage.py:137` locks the gap set empty; widening it requires editing the lock test (intentionally painful).

## The determinism gate

`firm/ops/check_reports_clean.sh` runs `make eval` twice and `diff -ruN`s the outputs. Any diff exits 1.

The gate is **behavioral**, not static — we don't try to enumerate sources of nondeterminism; we run the pipeline twice and watch for the first time output differs. A dependency upgrade that subtly reorders a dict, an OS-level iteration change, a clock leak — they all surface on the next CI run.

What the gate stands on:

| Mechanism | File |
|---|---|
| `ReplayClock` — no `datetime.now()` anywhere in the firm | `firm/core/clock.py:17` |
| VCR cassettes hashed by `(model, system, messages, tools)` | `firm/llm/cassettes.py:39` |
| Point-in-time RAG — `published_before=as_of`, naive datetime rejected | `firm/rag/retrieve.py:136` |
| Deterministic fake broker — `SHA256(ticker, timestamp)` → price | `firm/broker/fake_broker.py:96` |
| Frozen RNG — `get_rng()` seeded from `FIRM_RANDOM_SEED` (default 42) | `firm/core/random.py:42` |

## What is deliberately not measured

The eval report includes these lines verbatim (the template won't compile without them):

- **Investment quality / alpha** — N≈15 total trades; no return metric is statistically distinguishable from luck.
- **Generalization beyond 3 declared regimes** — three windows, sample of three.
- **Real-world fill quality** — paper sim with hashed prices and fixed slippage; no order book, no impact cost.
- **Forward references inside chunks** — point-in-time filtering is chunk-level; a chunk may textually reference a later event.
- **Long-horizon learning effects** — decision journal too sparse with ~15 trades.

## Sample report shape

```text
EVAL REPORT — R1_EARNINGS (2024-03-11 → 03-15)

Total return:           -1.2%
vs SPY (primary):       -2.0pp (SPY: +0.8%)
vs equal-weight basket: -0.8pp (basket: -0.4%)
Per-trade returns:      +2.8%, -1.4%, +0.6%, -0.9%, -2.1%
Hit rate:               2/5 (40%) — n=5, not statistically significant

PROCESS METRICS
  Groundedness:                  99.5%
  Decision discipline:           15/15
  Red-team pass:                 51/51
  HITL correctness:              12/12
  FailureMode coverage:          14/14

NOT MEASURED
  - Investment quality / alpha — N too small
  - Generalization beyond 3 declared regimes
  - Real-world fill quality — paper sim
  - Forward references inside chunks — known RAG limitation
  - Long-horizon learning effects — journal too sparse
```

## Inspect AI — the production target

[Inspect AI](https://inspect.aisi.org.uk/) (UK AISI's eval framework) is the natural next step at scale. We chose custom pytest + Jinja for the take-home because:

- Inspect AI is a framework with a multi-day learning curve and zero incremental signal for a reviewer of *this* artifact.
- Jinja with pinned context produces byte-identical output trivially. A third-party formatter sitting between our context and the bytes the determinism gate diffs is gratuitous risk.
- 9 of the 10 metrics are schema checks or audit-log invariants, not the labeled-dataset shape Inspect AI is built for.

Migration is reversible (~500 LOC): re-implement each `compute_<metric>` as an Inspect scorer, swap the Jinja template for a report writer, parallelize via Inspect's task runner. The data shapes (`MetricResult`, regime context dict) survive. See [`path-to-production.md`](path-to-production.md).
