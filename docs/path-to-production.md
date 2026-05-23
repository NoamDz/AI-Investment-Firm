# Path to Production

Take-home → firm-scale gap. The take-home is paper trading at single-host scale; production is the same firm at firm scale. The load-bearing invariants (typed `Decision` boundary, HMAC-signed HITL gate, outbox exactly-once, PIT-filtered RAG, byte-deterministic eval) are production-ready as-is — what changes is mostly operational: replica counts, where the DB lives, how secrets rotate, where traces land.

## Availability matrix (spec §3.6)

| Component | Tier | Take-home mechanism | Production path |
|---|---|---|---|
| Position Monitor, Reporter | Stateless, replicatable | Docker restart + healthcheck | N replicas behind LB |
| Research, PM | Stateless, replicatable | Docker restart | N replicas; deliberation idempotent per `decision_id` |
| Risk, Execution | **Singleton** (single-writer to broker) | Single process + restart | Leader election (etcd/Consul); broker idempotency keys make failover safe |
| SQLite `firm.db` | Single-writer fundamental | WAL + `synchronous=FULL` + Litestream | Postgres via SQLAlchemy + LangGraph `PostgresSaver` (one-import swap, §1) |
| Qdrant | Stateless w.r.t. business state | Single container; rebuild from source on loss | Qdrant Cloud or `replication_factor=3` cluster |
| MCP servers | Stateless | Restart on crash | Multi-replica behind AgentCore Gateway |

## 1. Postgres swap

**Two SQLite use-cases**: business state (`firm/db/schema.sql`, accessed via `firm/db/connection.py:8`) and LangGraph checkpoints (`SqliteSaver` at `firm/orchestrator/graph.py:83`). Both move to one Postgres database under `DATABASE_URL`.

**Changes**: rewrite `connection.py` to use SQLAlchemy + Postgres equivalents (`synchronous_commit=on`); swap `SqliteSaver` → `PostgresSaver` (literal one-import); add Alembic for versioned migrations.

**Portable as-is**: queries (`ON CONFLICT DO NOTHING` works on both), the outbox pattern, schemas (minor syntax: `INTEGER PRIMARY KEY AUTOINCREMENT` → `BIGSERIAL`).

**Needs verification**: `executescript` → loop over statements; NULL vs empty-string semantics (Postgres stricter); BOOLEAN type (SQLite stores 0/1, Postgres has native). Validation gate is the determinism check (`firm/ops/check_reports_clean.sh`) — Postgres-backed firm must produce byte-equivalent reports.

**Risk: Medium** — concentrated in the one-time data migration.

## 2. Risk + Execution HA (leader election)

The load-bearing row. Execution is fundamentally singleton — only one process can hold authoritative broker write access. HA is leader election with fast failover, not horizontal scaling.

Deploy N=2–3 replicas, etcd/Consul lease, only the leader holds the broker handle. On leader loss, standby promotes within ~5s, picks up `pending` outbox rows via `recover_pending`, retries with the **same idempotency key** — broker dedupes any in-flight orders.

**No code change** to Execution; the outbox already preserves idempotency keys across crashes. New: a single-purpose leader-election shim wrapping the existing execution path.

**Risk: High** — split-brain, failover gap (>5s), spurious failover under load. All have well-understood mitigations but require staging validation against realistic load. The outbox (spec §5.2) is what makes hot-standby safe; without it, dual-fire.

## 3. Multi-region AgentCore

Per-region module instantiation via Terraform `for_each` over `var.regions` with AWS provider `alias`. Route 53 health-based failover fronts the Runtime endpoints. Per-region IAM trust (no global wildcard).

**Hard problem**: AgentCore Memory is regional. Multi-region requires per-region namespaces + cross-region sync — likely DynamoDB Global Tables under a managed AgentCore option once it exists, or a custom replication shim until then. Reporter is the lowest-blast-radius agent (per `agentcore_mapping.md`), so a Memory-replication bug surfaces as a malformed report row, not a trade.

**Risk: Medium** — cross-region Memory consistency until AgentCore exposes managed replication.

## 4. Eval at scale (Inspect AI)

Replace the custom pytest + Jinja harness with [Inspect AI](https://inspect.ai-safety-institute.org.uk/). Keep `firm/ops/check_reports_clean.sh` (framework-agnostic). Re-implement each `compute_<metric>` in `firm/eval/process_metrics.py` as an Inspect scorer; re-implement Jinja templates as Inspect report writers. Data shapes (`MetricResult`, regime context dict) and the three regime windows (`R1_EARNINGS`, `R2_DRAWDOWN`, `R3_QUIET` at `firm/eval/regimes.py:91-109`) survive unchanged. Migration ≈ `firm/eval/runner.py` + two templates, ~500 LOC. See [`eval.md`](eval.md) §8.

**Risk: Low-medium** — vendor-formatter determinism is harder than our own Jinja; accept a one-time report-bytes recapture at cutover.

## 5. GHE + private package mirroring

GHE replaces GitHub.com; runners self-hosted in VPC. PyPI/npm/Docker Hub proxied via internal Artifactory or Verdaccio — sole source for installs, egress restricted to mirror IPs. SBOM (CycloneDX) per build, signed with release key; CI refuses unsigned builds. No application-code change.

**Risk: Low** — mirror becomes a single point of failure for builds; standard Artifactory HA.

## 6. KMS-rotated secrets + cross-region replication

Today: customer-managed KMS with annual key rotation; 6 Secrets Manager entries via `for_each` (`firm/anthropic_api_key`, `firm/slack_signing_secret`, `firm/slack_bot_token`, `firm/firm_hmac_secret`, `firm/firm_hmac_secret_prev`, `firm/firm_hmac_rotated_at`). Values written out-of-band after first apply.

Production adds: rotation Lambdas implementing the AWS `createSecret` / `setSecret` / `testSecret` / `finishSecret` protocol (30-day cadence); cross-region `replica { region kms_key_id }` blocks. The HMAC rotation Lambda matches the dual-key contract `firm/hitl/signing.py:172` `verify_with_rotation` already expects (Plan 3 T35).

**Risk: Medium** — rotation Lambda bug that promotes new key before all consumers fetch causes a transient signed-approval rejection window. Mitigation is the 24h dual-key grace window, already implemented.

## 7. Multi-vendor observability (Datadog/Honeycomb + CloudWatch)

`firm/obs/tracer.py:255` `init_tracer` already honours `OTEL_EXPORTER=otlp` (lazy gRPC exporter). Terraform's `observability` module runs otelcol-contrib on Fargate. Production adds a second exporter — `datadog` or `otlphttp` to Honeycomb — alongside `awscloudwatchlogs`. CloudWatch retains the audit/IAM role; vendor APM adds distributed-tracing + flame-graph + high-cardinality query that CloudWatch is weak at. No application-code change.

**Risk: Low** — span emission is fire-and-forget; a misconfigured exporter drops traces but doesn't corrupt decisions.

---

## What stays the same

The migration deltas above are about **operating at firm scale**, not correctness. The load-bearing invariants — what makes the firm safe with real money — don't move:

| Invariant | Where | Why it survives |
|---|---|---|
| Structured outputs (`Decision.action: ActionEnum`) | `firm/core/models.py:11,127` | Schema barrier, not deployment |
| Least-privilege MCP per agent (only Execution gets `Broker`) | `firm/agents/execution.py:66` | Construction barrier |
| HITL gate (`interrupt_before=["hitl"]`) | `firm/orchestrator/graph.py:84` | Graph-topology barrier |
| Signed-approval HMAC chain (dual-key rotation) | `firm/hitl/signing.py:109,172` | Already production-shaped |
| Red-team architectural invariants (5 asserts vs audit_log + outbox + broker) | `docs/threat_model.md` §7 | Channels are DB-agnostic |
| Determinism foundation (clock, cassettes, PIT-RAG, frozen RNG, fake broker, gate) | `firm/core/clock.py`, `firm/llm/cassettes.py`, `firm/rag/retrieve.py:136`, `firm/core/random.py:42`, `firm/broker/fake_broker.py:96`, `firm/ops/check_reports_clean.sh` | The determinism gate is the validation contract for every migration delta |
| Outbox pattern (SHA-256 idempotency keys) | `firm/outbox/outbox.py:51` | DB-agnostic, transport-agnostic, broker-agnostic |
| `FailureMode` enum (15 values, 14/14 fixture coverage) | `firm/core/models.py:19` | Coverage invariant carries over |

---

## Risk summary

| Delta | Section | Risk |
|---|---|---|
| Postgres swap | §1 | Medium |
| Risk/Execution HA (leader election) | §2 | **High** |
| Multi-region AgentCore | §3 | Medium |
| Inspect AI eval | §4 | Low-medium |
| GHE + private mirror | §5 | Low |
| KMS rotation + replication | §6 | Medium |
| Multi-vendor observability | §7 | Low |
| LB-fronted Monitor/Reporter | matrix | Low |
| N-replica Research/PM | matrix | Low-medium |
| Qdrant cluster | matrix | Low |
| MCP multi-replica | matrix | Low |
