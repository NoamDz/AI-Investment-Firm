# Plan 4: Eval Harness + Red-Team Corpus + GitHub Actions CI + Terraform/AgentCore Deployment

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal.** Close every deferred item from Plan 3 and ship the final reviewer-facing artefacts. At the end of Plan 4: `make eval` produces a reproducible 3-regime smoke report (per-trade returns + SPY/equal-weight basket benchmarks + 10 process metrics + "Not Measured") with a `git diff --exit-code reports/` determinism gate; `tests/red_team/` contains 50 architectural-invariant assertions across 10 injection classes; three GitHub Actions workflows (`pr.yml`, `main.yml`, `release.yml`) gate every PR; `infra/terraform/` plans clean to a multi-module AWS stack with a committed `PLAN.md`; the Reporter agent runs on AWS Bedrock AgentCore's local runtime; and `tests/integration/test_failure_mode_coverage.py` asserts **14/14** end-to-end triggering fixtures (15 enum values minus the `UNKNOWN` catch-all) — `ALLOWED_GAPS` is empty.

**Architecture.** Six orthogonal slices, each gated by a CI invariant. None of the slices depend on each other for landing, but the eval harness is sequenced first because the determinism gate (VCR cassettes + frozen seed + `git diff --exit-code reports/`) is the load-bearing primitive every other slice tests against.

1. **Determinism foundation.** Wrap `CachedAnthropicClient` with `vcrpy` cassettes (live / record / replay modes via `FIRM_VCR_MODE`); frozen RNG seed at process boot (`FIRM_RANDOM_SEED`, default 42, threaded through PM tie-breaks and any retrieval shuffles); CI helper `scripts/check_reports_clean.sh` that runs `make eval` twice and asserts `git diff --exit-code reports/`. This is the bedrock — without it nothing else in Plan 4 can claim reproducibility.
2. **Red-team corpus.** 50 cases × 10 injection classes (spec §8.5) loaded as a JSONL corpus; each test asserts an *architectural invariant* — no privileged broker action, no schema bypass, no unapproved trade — not pattern detection. Promotes `UNCITED_CLAIM` from `ALLOWED_GAPS` to a first-class end-to-end fixture (citation-forgery class).
3. **Eval harness (3-regime smoke).** Replay across the three regimes already declared in spec §9.3 (2024-03-11→15 earnings-heavy, 2024-08-05→09 drawdown, 2023-11-06→10 quiet); aggregate per-trade returns, hit rate, total return vs SPY (primary) + equal-weight basket (secondary); compute the 10 process metrics from spec §9.5; emit `reports/eval/<regime>.md` plus an aggregate `reports/eval/summary.md`. `git diff --exit-code reports/eval/` is the determinism gate.
4. **Final 14/14 `FailureMode` coverage.** Promote each entry currently in `ALLOWED_GAPS` (7 modes: `UNCITED_CLAIM`, `INSUFFICIENT_EVIDENCE`, `PROMPT_INJECTION_DETECTED`, `SCHEMA_VALIDATION_FAILED`, `STALE_DATA`, `UNGROUNDED_CLAIM`, `TOOL_PERMISSION_DENIED`, `UNAPPROVED_HIGH_RISK`, `BROKER_UNAVAILABLE` — net 7 after Plan 3 already promoted 3) to a triggering integration fixture. `ALLOWED_GAPS` becomes empty; the meta-test in `test_failure_mode_coverage.py` flips its assertion from "every value is fixture-or-gap" to "every value (except `UNKNOWN`) has a fixture".
5. **GitHub Actions CI.** Three workflows mapped per spec §11.3: `pr.yml` runs lint + type + unit + integration + 1-regime eval + determinism gate + `terraform validate` + docker build (no push); `main.yml` adds the full 3-regime eval + docker push to GHCR + `terraform plan`; `release.yml` tags + builds the release artefact and attaches the eval report to the release notes.
6. **Terraform IaC + AgentCore bonus.** `infra/terraform/` with six modules (`network`, `compute`, `storage`, `secrets`, `bedrock`, `observability`) and two env tfvars files; `terraform plan` output committed to `infra/terraform/PLAN.md` per spec §11.2. `docs/agentcore_mapping.md` captures the spec §11.1 migration table, and `firm/agentcore/reporter_adapter.py` runs the Reporter agent on AgentCore's local runtime (the simplest agent — no broker, no state mutation).

```
                                make eval (3 regimes)
                                       │
   FIRM_VCR_MODE=replay ──►  ReplayClock + cassettes + frozen RNG  ──►  reports/eval/
                                       │                                  │
                                       ▼                                  ▼
                          process + perf metrics                  git diff --exit-code

   tests/red_team/  ──► 50 cases × 10 classes ──► architectural invariants (no broker, no schema bypass)
                                       │
                                       ▼
                         FailureMode coverage 14/14 (UNCITED_CLAIM e2e)

   .github/workflows/{pr,main,release}.yml ──► gate everything above

   infra/terraform/  ──► plan-clean ──► PLAN.md
                                       │
   docs/agentcore_mapping.md + firm/agentcore/  ──► Reporter on AgentCore local runtime
```

**Tech Stack (additions vs Plan 3).**
- `vcrpy>=6.0` — LLM cassette record/replay (test-only dep)
- `numpy>=1.26` + `pandas>=2.1` — eval metric aggregation (already transitive but pinned here)
- `yfinance>=0.2.40` — historical SPY + universe prices for backtest benchmark (replay mode reads cached frames; live mode hits the API once at record-time only)
- `bedrock-agentcore-sdk>=0.1` — AWS Bedrock AgentCore local runtime (test-only, pinned to a public release)
- `terraform>=1.7` — IaC binary (no Python dep; install instructions in runbook)
- GitHub Actions Ubuntu runners (no new repo dep)

**Out of scope (post-Plan 4 / production hardening):**
- Live cloud deployment (`terraform apply` is human-gated and not in CI).
- Multi-region failover for the AgentCore runtime.
- Inspect-AI integration (referenced in spec §9.9 as future direction; out-of-scope per spec).
- Promoting other LangGraph agents (Research, PM, Risk, Execution) to AgentCore — only Reporter ships, by spec §11.1.

---

## File Structure

New and modified files relative to Plan 3. Unchanged files are not listed.

```
ai-investment-firm/
├── pyproject.toml                          # MODIFIED: add vcrpy, yfinance, bedrock-agentcore-sdk, pandas pin
├── Makefile                                # MODIFIED: add `eval`, `red-team`, `deploy-dev` targets
├── README.md                               # MODIFIED: status table marks Plan 4 done; eval badge + AgentCore note
├── .github/
│   └── workflows/
│       ├── pr.yml                          # NEW: lint + type + unit + integration + 1-regime eval + tf validate + docker build
│       ├── main.yml                        # NEW: PR + 3-regime eval + docker push + tf plan
│       └── release.yml                     # NEW: tagged build + release artefact + eval report attach
├── infra/
│   └── terraform/
│       ├── main.tf
│       ├── variables.tf
│       ├── providers.tf
│       ├── PLAN.md                         # NEW: committed `terraform plan` output (spec §11.2)
│       ├── modules/
│       │   ├── network/                    # VPC, subnets, SGs
│       │   ├── compute/                    # ECS Fargate task + service
│       │   ├── storage/                    # RDS Postgres + S3 reports/traces
│       │   ├── secrets/                    # Secrets Manager
│       │   ├── bedrock/                    # AgentCore runtime config
│       │   └── observability/              # CloudWatch + OTLP collector
│       └── envs/
│           ├── dev.tfvars
│           └── prod.tfvars
├── firm/
│   ├── llm/
│   │   ├── cassettes.py                    # NEW: vcrpy wrapper around CachedAnthropicClient, modes live/record/replay
│   │   └── anthropic_client.py             # MODIFIED: honor FIRM_VCR_MODE env; pass through to cassettes layer
│   ├── core/
│   │   └── random.py                       # NEW: seeded RNG facade (FIRM_RANDOM_SEED env, default 42)
│   ├── eval/                               # NEW MODULE
│   │   ├── __init__.py
│   │   ├── regimes.py                      # Three regime configs per spec §9.3
│   │   ├── benchmarks.py                   # SPY + equal-weight basket return calc
│   │   ├── perf_metrics.py                 # per-trade returns, hit rate, total return
│   │   ├── process_metrics.py              # 10 process metrics per spec §9.5
│   │   ├── runner.py                       # orchestrates 1 regime end-to-end → report
│   │   └── templates/
│   │       ├── regime.md.j2
│   │       └── summary.md.j2
│   ├── agentcore/                          # NEW MODULE (Plan 4 bonus)
│   │   ├── __init__.py
│   │   └── reporter_adapter.py             # Reporter on AgentCore local runtime
│   └── cli.py                              # MODIFIED: add `eval` + `red-team` subcommands
├── tests/
│   ├── eval/                               # NEW
│   │   ├── conftest.py                     # ReplayClock + cassette + frozen seed wiring
│   │   ├── test_regime_smoke.py            # parametrized over 3 regimes
│   │   ├── test_benchmarks.py              # SPY + basket math
│   │   ├── test_process_metrics.py         # 10 metrics × invariant checks
│   │   └── test_determinism_gate.py        # runs eval twice, asserts byte-identical reports
│   ├── red_team/                           # NEW
│   │   ├── conftest.py
│   │   ├── corpus.jsonl                    # 50 cases × 10 classes
│   │   ├── test_direct_override.py
│   │   ├── test_role_hijack.py
│   │   ├── test_delimiter_break.py
│   │   ├── test_unicode_homoglyph.py
│   │   ├── test_encoded.py
│   │   ├── test_indirect_tool_output.py
│   │   ├── test_multistep_chain.py
│   │   ├── test_citation_forgery.py        # ← also triggers UNCITED_CLAIM end-to-end
│   │   ├── test_spoofed_approval.py
│   │   └── test_confused_deputy.py
│   ├── integration/
│   │   ├── test_failure_mode_coverage.py   # MODIFIED: 14/14 fixtures, ALLOWED_GAPS empty
│   │   ├── test_failuremode_insufficient_evidence.py  # NEW
│   │   ├── test_failuremode_prompt_injection.py       # NEW
│   │   ├── test_failuremode_schema_validation.py      # NEW
│   │   ├── test_failuremode_stale_data.py             # NEW
│   │   ├── test_failuremode_ungrounded_claim.py       # NEW
│   │   ├── test_failuremode_tool_permission.py        # NEW
│   │   ├── test_failuremode_unapproved_high_risk.py   # NEW
│   │   └── test_failuremode_broker_unavailable.py     # NEW
│   └── unit/
│       ├── test_random_facade.py           # NEW: frozen seed reproducibility
│       └── test_vcr_cassettes.py           # NEW: record/replay/live mode switching
├── scripts/
│   ├── check_reports_clean.sh              # NEW: runs eval twice, asserts git diff --exit-code reports/eval/
│   └── eval_capture.py                     # NEW: one-time record-mode cassette capture
├── sample_runs/                            # MODIFIED
│   ├── 2024-03-13/                         # existing
│   ├── 2024-08-07/                         # NEW: regime-2 sample
│   └── 2023-11-08/                         # NEW: regime-3 sample
├── docs/
│   ├── eval.md                             # NEW: spec §9 report shape + Inspect AI reference
│   ├── threat_model.md                     # NEW: spec §8.6 — architecture-as-defense rationale
│   ├── agentcore_mapping.md                # NEW: spec §11.1 migration table
│   ├── path-to-production.md               # NEW: prod deltas (Postgres swap, Qdrant cluster, etc.)
│   ├── runbook.md                          # MODIFIED: add eval + tf validate + AgentCore sections
│   └── architecture.md                     # MODIFIED: add deployment view (Terraform module → AWS resource)
└── reports/
    └── eval/                               # NEW (committed sample outputs)
        ├── 2024-03-13.md
        ├── 2024-08-07.md
        ├── 2023-11-08.md
        └── summary.md
```

---

## Task list

Tasks are ordered for subagent-driven execution. Each task is sized to fit one fresh-subagent invocation (plus two-stage review). Sections A and D are sequenced first because every downstream section depends on the determinism foundation (A) and the empty `ALLOWED_GAPS` invariant (D). Sections B, C, E, F, G are mostly parallel-safe once A lands.

### Section A — Determinism foundation (spec §9.2)

This section is first because every downstream invariant in this plan tests against it.

- [x] **T01: `firm/llm/cassettes.py`** — `vcrpy` wrapper. New `CassetteClient(real_client, mode, cassette_dir)`. Modes: `live` (pass-through, no recording), `record` (pass-through + write `.yaml` cassettes keyed by request hash), `replay` (raise if no cassette match). Cassette key derives from `(model, messages_hash, tools_hash, system_hash)` so prompt drift surfaces as a hard miss. Wrap inside `CachedAnthropicClient` so the call order is: cache → cassette → live. Reason: cache-misses-but-cassette-hits stay deterministic; cassette-miss-but-cache-hit never happens because cache writes are gated on real responses only. Test in `tests/unit/test_vcr_cassettes.py`: round-trip record→replay produces identical responses; replay-miss raises `CassetteMissError`.

- [x] **T02: `FIRM_VCR_MODE` env wiring** — `firm/llm/anthropic_client.py` reads `FIRM_VCR_MODE` (default `live` in production, conftest pins `replay` for tests). Conftest fixture exposes a temp cassette dir for record-mode tests. Document in `CONTRIBUTING.md` + runbook: how to re-record after a deliberate prompt change. Test: `pytest -m vcr_replay` flag selects replay-mode test variants of existing integration tests.

- [x] **T03: `firm/core/random.py` seeded RNG facade** — Single `get_rng() -> random.Random` that always seeds from `FIRM_RANDOM_SEED` env (default `42`). Replace every callsite of `random.random()` / `random.shuffle()` / `random.choice()` across `firm/` (PM tie-breaks, any retrieval shuffles in `firm/rag/`) with `get_rng().*`. Test: two processes with the same seed produce identical PM voter ordering on a deliberately-tied vote.

- [x] **T04: `scripts/check_reports_clean.sh`** — Shell script: runs `make eval` twice into temp dirs, diffs them. If non-empty diff → exit 1 with a unified-diff snippet. Wire into `Makefile:check-determinism` target. CI uses this directly (Section E). Test: deliberately mutate one cassette → script exits non-zero with a sensible diff.

### Section B — Red-team corpus (spec §8.5)

Each test asserts an *architectural invariant* — no privileged action reaches the broker, no schema validation is bypassed, no approval is forged. The LLM is allowed to respond however it wants; we only assert that the *system* ignored the malicious instruction. Five cases per class × 10 classes = 50 tests.

- [x] **T05: `tests/red_team/corpus.jsonl` (50 cases × 10 classes)** — JSONL with fields `{case_id, injection_class, payload_text, target_agent, invariant_id}`. Five cases per class: (1) direct override, (2) role hijack, (3) delimiter break, (4) unicode/homoglyph, (5) encoded (base64, rot13), (6) indirect via tool output, (7) multi-step chain, (8) citation forgery, (9) spoofed approval, (10) confused deputy. Each case names which invariant the assertion checks: `INV_NO_PRIVILEGED_ACTION`, `INV_NO_SCHEMA_BYPASS`, `INV_NO_UNAPPROVED_TRADE`, `INV_NO_FORGED_CITATION`, `INV_NO_FORGED_APPROVAL`. Schema-validated at corpus-load time (rejects malformed entries). Test: `pytest tests/red_team/conftest.py::test_corpus_schema` collects + validates.

- [x] **T06: `tests/red_team/conftest.py` invariant fixture** — Loads the corpus, parameterises a fixture by case_id, provides a `red_team_run(case)` helper that pipes `payload_text` into the named `target_agent`, returns the resulting `Decision` + broker call log + audit log. The five invariants are encoded as assertion helpers (`assert_no_privileged_action`, `assert_no_schema_bypass`, `assert_no_unapproved_trade`, `assert_no_forged_citation`, `assert_no_forged_approval`). Each helper consults the audit log + outbox + broker mock to verify nothing crossed the barrier.

- [x] **T07: Ten test modules, one per injection class** — `tests/red_team/test_<class>.py`. Each parameterises over the 5 cases for that class and asserts the named invariants. Citation forgery (T07.h) is special: also asserts the resulting Decision carries `failure_mode=UNCITED_CLAIM` — this is the end-to-end UNCITED_CLAIM fixture that closes the `ALLOWED_GAPS` entry from Plan 3 (so this task plus T11 jointly satisfy Section D's UNCITED_CLAIM requirement).

- [x] **T08: `make red-team` target + `firm.cli red-team`** — Runs only the `tests/red_team/` suite, prints `X/50 passed` summary line, exits non-zero on any failure. Suite must run in <60s in replay mode (CI budget). Wire into `pr.yml` (Section E) as a dedicated step.

### Section C — Eval harness (spec §9)

Replay smoke test, not a backtest. Outputs are committed under `reports/eval/` and gated by `check_reports_clean.sh`.

- [x] **T09: `firm/eval/regimes.py`** — Three `RegimeConfig` dataclasses matching spec §9.3 exactly: (`r1_earnings`, 2024-03-11→15, "earnings-heavy"); (`r2_drawdown`, 2024-08-05→09, "post-Aug-5 sell-off"); (`r3_quiet`, 2023-11-06→10, "low-volatility quiet"). Each carries `start_date`, `end_date`, `universe` (the frozen 30-ticker `config/universe.yaml`), `seed_overrides`. Test: deepcopy equality + serialization round-trip.

- [x] **T10: `firm/eval/benchmarks.py`** — `compute_spy_return(start, end) -> float` and `compute_basket_return(tickers, start, end) -> float`. Uses `yfinance` in record mode to fetch + cache historical adjusted closes into `data/eval/prices/<ticker>.parquet`; replay mode reads from the parquet only (no network). Equal-weight basket = arithmetic mean of per-ticker total returns over the window. Tests: golden numbers for one regime computed offline and pinned.

- [x] **T11: `firm/eval/perf_metrics.py`** — From the heartbeat decisions and broker fills, compute: per-trade return %, hit rate (winners/total), total return (cash + positions delta). Round to 1 decimal place for stability. Output `dict[str, float | str]` matching the spec §9.7 sample shape. Test against a fixture portfolio with hand-computed expected values.

- [x] **T12: `firm/eval/process_metrics.py`** — Compute all 10 process metrics from spec §9.5: groundedness, decision discipline, citation diversity, reversal rate, risk-policy compliance, HITL correctness, schema rejections, red-team pass (calls into Section B suite + reads the X/50), sufficiency-gate precision/recall (against `tests/fixtures/sufficiency_dev_set.jsonl` — 30 labeled queries), FailureMode coverage (reads from `test_failure_mode_coverage.py` results). Each metric returns `MetricResult(name, value, threshold, status)`. Test asserts every metric exists + threshold check fires.

- [x] **T13: `firm/eval/runner.py`** — `run_regime(config) -> RegimeReport`. Spins up a fresh sqlite DB + Qdrant collection in a temp dir, seeds frozen, ReplayClock pinned to `start_date`, runs the daily heartbeat for each calendar day in the window, collects decisions/spans/cost ledger/broker fills, calls T10 + T11 + T12, writes `reports/eval/<start_date>.md` via the Jinja template. Test: one regime produces a non-empty report that includes every spec §9.7 section header.

- [x] **T14: Jinja templates `regime.md.j2` + `summary.md.j2`** — Exactly the shape from spec §9.7 (per-regime + aggregated process metrics + "Not Measured" block). The "Not Measured" section is hard-coded (alpha, generalization, real-world fill, forward-references, long-horizon learning effects — spec §9.6). Golden-file test compares one regime output byte-for-byte.

- [x] **T15: `firm/cli.py eval` subcommand + `make eval` target** — `firm eval --regime {r1,r2,r3,all}` (default `all`). Calls T13 once per regime, then aggregates into `reports/eval/summary.md`. Idempotent: deleting `reports/eval/` and re-running produces byte-identical files. Wire into `make eval`. Test: `make eval && make eval && git diff --exit-code reports/eval/` exits 0.

- [x] **T16: One-time cassette capture script** — `scripts/eval_capture.py` runs each regime in record mode against the real Anthropic API + yfinance, writes cassettes under `tests/eval/cassettes/<regime>/` + price parquet files. Committed once; never re-run in CI. Documented in runbook §"Re-recording eval cassettes". Test: replay-mode `make eval` after committing the cassettes succeeds without network.

- [x] **T17: Sample-run extension to all 3 regimes** — Add `sample_runs/2024-08-07/` + `sample_runs/2023-11-08/` (regime midpoints) alongside the existing `sample_runs/2024-03-13/`. Each contains `daily_report.md`, `positions.xlsx`, `decisions.jsonl`, `trace.jsonl`. Generated by the same `firm eval` run that produces the per-regime report. Test: schema validation + trace_id ↔ decisions.jsonl linkage on each new sample (extends the existing `test_sample_run_bundle_committed`).

### Section D — Final 14/14 `FailureMode` coverage (spec §3.5, §8.5)

Each task here is a single integration test that triggers exactly one previously-deferred `FailureMode` end-to-end. The last task flips the meta-assertion.

- [x] **T18: `INSUFFICIENT_EVIDENCE` fixture** — Stub Research to return zero retrieval results for a known ticker. Heartbeat → Decision with `action=REFUSE`, `failure_mode=INSUFFICIENT_EVIDENCE`. New file `tests/integration/test_failuremode_insufficient_evidence.py`.

- [x] **T19: `PROMPT_INJECTION_DETECTED` fixture** — Inject a delimiter-break payload through the news adapter (`firm/rag/news.py` test override). PM voter's sanitiser flags it; Decision carries `failure_mode=PROMPT_INJECTION_DETECTED`. New file `tests/integration/test_failuremode_prompt_injection.py`. Cross-references Section B test_delimiter_break case `c3` for the payload.

- [x] **T20: `SCHEMA_VALIDATION_FAILED` fixture** — Force the PM agent to emit a Decision with a missing required field (e.g. via a corrupted cassette). Schema validator at the graph boundary catches it → `failure_mode=SCHEMA_VALIDATION_FAILED`, audit row written, no broker call. New file.

- [x] **T21: `STALE_DATA` fixture** — Pin ReplayClock to T+0 and inject a market-data quote with `as_of_ts = T - 90s` (over the 60s staleness cap). Risk node refuses → `failure_mode=STALE_DATA`. New file.

- [x] **T22: `UNGROUNDED_CLAIM` fixture** — PM emits a Decision whose rationale references a `source_chunk_id` that doesn't exist in the chunks table. The grounding validator catches it (extension of Plan 2 invariant) → `failure_mode=UNGROUNDED_CLAIM`. New file.

- [x] **T23: `TOOL_PERMISSION_DENIED` fixture** — Research attempts to call a broker.place_order tool from inside Research (which has no broker access). MCP capability check rejects → `failure_mode=TOOL_PERMISSION_DENIED`. New file. Also serves as the Section B `confused_deputy` invariant in one of its cases.

- [x] **T24: `UNAPPROVED_HIGH_RISK` fixture** — Heartbeat produces a trade > 3% NAV; HITL queue gets a pending row; the test does NOT post an approval; HITL timeout fires (mock the timer); Decision carries `failure_mode=UNAPPROVED_HIGH_RISK` instead of `HITL_TIMEOUT` because the conservative-default policy is "treat as unapproved high risk, not just timeout". New file. Distinguishes from `HITL_TIMEOUT` (already covered) by post-timeout disposition.

- [x] **T25: `BROKER_UNAVAILABLE` fixture** — Mock the broker MCP to return 503 on `place_order`. Execution agent's outbox retries N times then surfaces `failure_mode=BROKER_UNAVAILABLE`. New file. Asserts the outbox row stays in `pending` state for the next heartbeat (not silently lost).

- [x] **T26: Flip `ALLOWED_GAPS` to empty + meta-assertion** — `tests/integration/test_failure_mode_coverage.py`: remove every entry from `ALLOWED_GAPS`. The four meta-tests already in the file (every-enum-mapped, paths-resolve, no-overlap, no-stale-gaps) flip to a strict "every enum value (except `UNKNOWN`) has a triggering fixture". Update `FAILURE_MODE_FIXTURES` registry to include the 7 new modules from T18–T25 + UCITED_CLAIM from Section B T07.h. Final count: **14 fixtures + 1 catch-all UNKNOWN = 15 enum values**. Spec §9.5 + §9.7 strings restated as `14/14`.

### Section E — GitHub Actions CI/CD (spec §11.3)

- [x] **T27: `.github/workflows/pr.yml`** — Triggers on every PR. Steps in order: checkout → setup Python 3.11 → `pip install -e ".[dev]"` → `ruff check` → `mypy firm/` → `pytest -m "not requires_models" -q` → `pytest -m "requires_models" -q` (separate job, allowed to time out) → `make eval REGIME=r1` (just one regime for PR-speed) → `scripts/check_reports_clean.sh` → `make red-team` → `terraform -chdir=infra/terraform validate` → `docker build .` (no push). Total budget: ≤15 minutes. Use the GitHub Actions matrix to parallelise unit + integration + eval.

- [x] **T28: `.github/workflows/main.yml`** — Triggers on merges to main. Inherits every step from pr.yml, then: `make eval REGIME=all` (all 3 regimes), `terraform -chdir=infra/terraform plan -var-file=envs/dev.tfvars -out=plan.bin` (output committed as artifact, not applied), `docker build -t ghcr.io/<owner>/ai-investment-firm:$GITHUB_SHA . && docker push ghcr.io/<owner>/ai-investment-firm:$GITHUB_SHA`. The push step requires `GITHUB_TOKEN` with `packages: write` scope; documented in the workflow file.

- [x] **T29: `.github/workflows/release.yml`** — Triggers on `v*` tags. Inherits main.yml, then: builds the release zip (`firm-<tag>.tar.gz` containing the source + `infra/terraform/PLAN.md` + `reports/eval/`), creates a GitHub Release with the eval-summary excerpt as the body, attaches the zip + the latest `reports/eval/summary.md`. Asserts the tag matches `v\d+\.\d+\.\d+` (semver).

- [x] **T30: README CI badges** — Add three `[![pr](url)](url)` badges under the project title for the three workflows. Document workflow status semantics in the README (green = ship-it; red = no merge). Test: regex-check the README contains all three badge URLs after the change.

### Section F — Terraform IaC (spec §11.2)

Six modules. Terraform validates and plans clean against AWS; not applied. `PLAN.md` is the artefact reviewers see.

- [x] **T31: `infra/terraform/{main,variables,providers}.tf`** — Top-level orchestrator. AWS provider region from `var.region`, default `us-east-1`. Composes all six modules (T32–T37). State backend uses S3 + DynamoDB lock (configured but commented out for take-home; uncomment instructions in `path-to-production.md`). `terraform fmt -check` clean.

- [x] **T32: `modules/network`** — VPC (10.0.0.0/16), 2 public + 2 private subnets across 2 AZs, IGW, NAT, route tables, SGs for: ECS task (egress only), RDS (5432 from ECS only), OTLP collector (4317 from ECS only). Outputs: VPC ID, subnet IDs, SG IDs. No tests (Terraform plans against AWS schema validate it).

- [x] **T33: `modules/compute`** — ECS Fargate cluster + service + task definition for the `firm` container. CPU/mem from variables (default 1 vCPU / 2 GB). IAM role with policies for: Secrets Manager read, Bedrock InvokeAgent, S3 reports/ write, CloudWatch Logs write. Service-level autoscaling: 1–3 tasks on CPU > 70%.

- [x] **T34: `modules/storage`** — RDS Postgres 15 (db.t4g.micro for dev, db.r6g.large for prod, gated by `var.env`), S3 buckets for `reports/`, `traces/`, `cassettes/` (each with versioning enabled + lifecycle rule expiring traces after 90 days). RDS subnet group + parameter group with `max_connections=200`.

- [x] **T35: `modules/secrets`** — AWS Secrets Manager entries (placeholders only; actual values rotated out-of-band): `firm/anthropic_api_key`, `firm/slack_signing_secret`, `firm/slack_bot_token`, `firm/firm_hmac_secret`, `firm/firm_hmac_secret_prev`, `firm/firm_hmac_rotated_at`. KMS key for encryption-at-rest. Output: secret ARNs (consumed by `modules/compute` IAM policy).

- [x] **T36: `modules/bedrock`** — AgentCore Runtime config (one runtime: `firm-reporter`), AgentCore Memory namespace `firm-desk-state`, AgentCore Identity provider linked to the HMAC signing secret from T35. Validate-only (no apply); the AgentCore CLI is invoked separately in T39 to confirm the Reporter actually runs.

- [x] **T37: `modules/observability`** — CloudWatch Log Group for `firm` (90-day retention), OTLP Collector ECS service (otelcol-contrib image) listening on 4317/4318 inside the VPC, CloudWatch dashboard JSON with widgets for: heartbeat duration p50/p95, cost-by-model bar chart, failure_mode pie, decision-action histogram.

- [x] **T38: `terraform plan` capture + `PLAN.md`** — Run `terraform plan -var-file=envs/dev.tfvars > PLAN.md`. Commit the resulting plan (sanitised — no AWS account IDs, no real ARNs; replace with `<account-id>` / `<region>` placeholders via a sed pass in `scripts/sanitise_plan.sh`). Doc invariant: `PLAN.md` exists, > 100 lines, contains every module name. CI checks file existence; the actual plan-clean assertion runs in `main.yml` (T28).

- [x] **T38a: `make deploy-dev` target** — `terraform -chdir=infra/terraform apply -var-file=envs/dev.tfvars -auto-approve`. **Human-gated** — Makefile prints `WARNING: This will create real AWS resources` and waits for `READ -p "Type 'DEPLOY' to continue: "`. CI never runs this. Documented in runbook §"Deploying to dev".

### Section G — AWS Bedrock AgentCore (spec §11.1)

- [x] **T39: `docs/agentcore_mapping.md`** — The full migration table from spec §11.1, expanded with concrete code references and LOC estimates for each row. Includes a "Why Reporter first" rationale (no broker, no state mutation, simplest blast radius). Includes the `firm/agentcore/reporter_adapter.py` design sketch (T40).

- [x] **T40: `firm/agentcore/reporter_adapter.py`** — AgentCore Runtime adapter for the Reporter agent. Wraps `firm.agents.reporter.heartbeat_summary()` in the AgentCore SDK's `@agent` decorator; the adapter handles input/output marshalling (AgentCore's `InvocationRequest` → Reporter's `(decisions, cost_ledger, traces)` args). Lazy import so `pip install -e .` (without `[agentcore]` extra) still works. Test in `tests/integration/test_agentcore_reporter.py`: invoke through the AgentCore local runtime, assert the returned markdown matches the LangGraph Reporter's output byte-for-byte.

- [x] **T41: `firm/agentcore/__init__.py` + extra `[agentcore]` in pyproject** — Optional install group. `pip install -e ".[agentcore]"` brings `bedrock-agentcore-sdk`. Without the extra, importing `firm.agentcore.reporter_adapter` raises a friendly `ImportError("install firm[agentcore]")`. Test: with extra → works; without extra → graceful ImportError; the rest of `firm/` keeps importing cleanly.

### Section H — Documentation + repo polish

- [x] **T42: `docs/eval.md`** — Spec §9 in long form: framing, determinism foundation, 3 regimes, performance metrics (why both SPY and basket), process metrics, "Not Measured" section, sample report shape, Inspect AI reference (§9.9). Cross-link to `reports/eval/summary.md`. Test: link-checker passes.

- [x] **T43: `docs/threat_model.md`** — Spec §8.6: architecture-as-defense, the 50-case red-team corpus is the *measurement*, not the defense; hygiene (sanitisation, allowlists) is supplementary. Walks through the 5 architectural invariants from T06 with file:line references to where each is enforced in code.

- [x] **T44: `docs/path-to-production.md`** — The deltas between take-home and production: Postgres swap (SQLAlchemy `DATABASE_URL`), Qdrant cluster, multi-region AgentCore, Inspect AI eval, GHE for repo, KMS-rotated secrets, Datadog/Honeycomb in addition to CloudWatch. References every "Production path" cell from spec §3.6.

- [x] **T45: README + status table marks Plan 4 done** — Status table updates: Plan 4 row gets `- [x]`. New "Eval" + "Deployment" sections in the quickstart. Eval badge link wired (T30). Test: regex check on README for the four status-table rows being marked done.

- [x] **T46: `docs/runbook.md`** — Three new sections: (1) `make eval` operational guide (how to re-record cassettes; what to do when determinism gate fails), (2) `terraform plan` walkthrough (how to read PLAN.md, what each module owns), (3) AgentCore Reporter deployment (how to invoke via the local runtime, then via the deployed Runtime).

- [x] **T47: Plan 4 status update + final memory update** — Mark all Plan 4 task checkboxes done in this file. Update `MEMORY.md` Plan 4 entry from "drafted" to "shipped". Update spec §9.7 `FailureMode coverage` string to read `14/14` (drop the Plan 3 transition phrasing). Final commit: `chore(plan4): mark all 47 tasks complete`.

---

## CI invariants delivered by this plan

| Invariant | Test | Lives in |
|---|---|---|
| `make eval` produces byte-identical reports across runs | `scripts/check_reports_clean.sh` in `pr.yml` + `test_determinism_gate.py` | `tests/eval/` + CI |
| 50/50 red-team cases pass (each on architectural invariant) | `make red-team` | `tests/red_team/` |
| `FailureMode` coverage 14/14 fixtures (ALLOWED_GAPS empty) | `test_failure_mode_coverage.py` | `tests/integration/` |
| All 3 regimes complete inside the CI time budget | `main.yml::eval-all-regimes` step | `.github/workflows/main.yml` |
| Terraform validates + plans clean | `pr.yml::terraform-validate` + `main.yml::terraform-plan` | `.github/workflows/` |
| Docker image builds on every PR; pushes on main | `pr.yml::docker-build` + `main.yml::docker-push` | `.github/workflows/` |
| Reporter agent runs on AgentCore local runtime (output equiv. to LangGraph) | `test_agentcore_reporter.py` | `tests/integration/` |
| `pyproject.toml` `[agentcore]` extra is optional (firm imports without it) | `test_agentcore_optional_extra` | `tests/unit/` |
| Frozen RNG → identical PM voter ordering on tied vote | `test_random_facade.py` | `tests/unit/` |
| VCR cassette round-trip identical responses | `test_vcr_cassettes.py` | `tests/unit/` |

---

## Risks / notes

Each risk below has a planned mitigation; the right column names the task that delivers it.

- **Cassette drift on prompt changes → T01 + T16.** Prompts get edited often during development; stale cassettes silently replay old responses. T01 keys cassettes on a hash of `(model, messages, tools, system)` so any prompt change forces a replay-miss → CI fails loudly. T16's one-shot capture script + a runbook section document the re-record procedure.
- **Eval flakiness from non-determinism in transitive deps → T03 + T04.** `random.shuffle()` in any new dep or any unfrozen embedding model destroys the determinism gate. T03 forces every callsite through `get_rng()`; T04's twice-and-diff CI gate makes regressions impossible to merge silently.
- **Red-team false positives (over-strict invariants) → T06.** Architectural-invariant assertions are wide; an off-by-one in the broker-mock could fail every red-team case at once. T06's assertion helpers each have their own positive-control test (a case that *should* trip the invariant) so the helper is exercised independently of the corpus.
- **AgentCore SDK churn → T41.** The SDK is pre-1.0; an upstream rename breaks `firm/agentcore/`. T41 keeps the dep in an optional `[agentcore]` extra so the core test matrix doesn't tie to its release cadence; the AgentCore integration test runs in a separate CI job with `continue-on-error: true` until the SDK stabilises (post-Plan-4 hardening).
- **Terraform-plan diff churn → T38.** `terraform plan` output drifts on every AWS provider release (resource attribute defaults, etc.). T38's `sanitise_plan.sh` strips volatile lines (timestamps, computed values marked `(known after apply)`) so PR diffs against `PLAN.md` only highlight semantic changes.
- **CI runtime budget overrun on cold runners → T27.** Cold-runner pip install + sentence-transformer weight download can blow past the 15-minute PR budget. T27 uses GitHub Actions cache for both `~/.cache/pip` and the model-weight directory, and splits `requires_models` tests into a separate job that's allowed to time out without blocking the merge.
- **Postgres swap surprises in prod → T44.** SQLite-specific SQL (e.g. `INSERT OR IGNORE`, `PRAGMA wal_autocheckpoint`) doesn't translate. T44's `path-to-production.md` enumerates every such site discovered during the audit (with file:line refs) so the migration is a checklist, not an investigation.

**Carried forward documented limitations:**
- PIT forward-reference leakage (spec §6.4, runbook §"Known Limitations" added in Plan 2).
- Inspect AI integration is a future-direction reference (spec §9.9), not a Plan 4 deliverable.
- Live `terraform apply` is human-gated (T38a); production deployment is not in CI by design.

---

**End of Plan 4.**
