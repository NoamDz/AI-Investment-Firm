# Path to Production — From Take-Home to Firm-Scale

> Long-form companion to spec §3.6. Authoritative source: `docs/superpowers/specs/2026-05-18-ai-investment-firm-design.md` lines 155-170.
> Last updated: 2026-05-22 (Plan 4 T44).
> Related: [`docs/agentcore_mapping.md`](agentcore_mapping.md) (AgentCore migration baseline), [`docs/eval.md`](eval.md) §8 (Inspect AI reference), [`docs/threat_model.md`](threat_model.md) (architectural invariants that survive the migration).

---

## 1. Framing — what the take-home delivers, what production adds

The take-home is **paper trading at single-host scale**. The production target
is a **fiduciary system at firm scale**. The architectural distance between
those two is smaller than it looks, because the take-home is not a sketch —
it is a working system that already enforces the load-bearing invariants
(typed `Decision` boundary, HMAC-signed HITL gate, outbox exactly-once,
PIT-filtered RAG, byte-deterministic eval). What changes from single-host
to firm-scale is mostly operational: how many replicas, where the database
lives, how secrets rotate, where traces land. The schemas, the deliberation
flow, the safety barriers — those are production-ready as-is. We say this
explicitly in §9 ("What stays the same") because a reader who comes away from
this document thinking the take-home is a toy will have misread it.

The structure of this doc mirrors the structure of the gap. §2 reproduces
spec §3.6's component-by-component availability table and walks each row in
detail. §3 through §8 are deep-dives on the five highest-impact deltas
(Postgres swap, multi-region AgentCore, Inspect AI at scale, GHE +
private-package mirroring, KMS-rotated secrets with cross-region replication,
multi-vendor observability). §9 names what does **not** change — the
invariants the migration is required to preserve.

**What we are not doing in this doc.** We are not writing a runbook for the
migration; that belongs in a deployment plan. We are not estimating
person-weeks per delta. We are not picking a specific cloud-managed Postgres
SKU or a specific Datadog plan tier. The point here is to make the gap
**concrete** — every cell in spec §3.6's "Production path" column traces to
a named component, a target replacement, and a migration risk rating — so a
production engineer arriving at the codebase can read this doc and know
exactly what changes and what does not.

---

## 2. The take-home → production matrix

Reproduced verbatim from spec §3.6 (lines 159-166):

| Component | Tier | Mechanism (take-home) | Production path |
|---|---|---|---|
| Position Monitor, Reporter | Stateless, replicatable | Docker restart policy + healthcheck | Run N replicas behind load balancer |
| Research, PM | Stateless, replicatable | Docker restart policy | Run N replicas; deliberation idempotent within a `decision_id` |
| Risk, Execution | **Must be singleton** (single-writer to broker) | Single process + restart policy | Leader election (etcd/Consul); broker idempotency keys make failover safe |
| SQLite (`firm.db`) | Single-writer fundamental | WAL + `synchronous=FULL` + Litestream continuous backup | Migrate to Postgres via SQLAlchemy `DATABASE_URL` + LangGraph `PostgresSaver` (one-import swap, §5.3) |
| Qdrant | Stateless w.r.t. business state | Single container; on loss, re-ingest from source corpus | Qdrant Cloud or replicated cluster |
| MCP servers | Stateless | Restart on crash | Multi-replica; clients reconnect |

Plus the additional production deltas the Plan 4 spec calls out outside
§3.6 (multi-region AgentCore, Inspect AI for eval, GHE for repo, KMS-rotated
secrets, Datadog/Honeycomb observability) — see §4 through §8.

The next six subsections walk each row in detail.

### 2.1 Position Monitor, Reporter — stateless, replicatable

**Take-home mechanism.** Both agents are constructed by factories
([`firm/agents/monitor.py:10`](../firm/agents/monitor.py) `make_monitor`,
[`firm/agents/reporter.py:88`](../firm/agents/reporter.py) `make_reporter`)
that return pure closures over `WorkingState`. Neither holds mutable
per-process state; both write only to append-only artifacts (reports/JSONL,
audit_log rows). Docker restart policy with healthcheck is enough at
single-host scale because a restarted replica re-reads `firm.db` and picks
up exactly where the previous instance left off.

**Production target.** Run N replicas of each behind an internal load
balancer. Heartbeats are work units indexed by `decision_id`, so the LB can
hash on `decision_id` for sticky routing or round-robin for symmetric
treatment. Reporter is a pure projection (per
[`docs/agentcore_mapping.md`](agentcore_mapping.md) "Why Reporter First"), so
duplicated work across replicas is idempotent — two replicas processing the
same heartbeat write the same JSONL row, which is dedup-safe by content hash.

**Migration shape.** Config change. The factory signatures already accept
all per-instance state as constructor arguments. The deployment-layer
change is: scale `firm-monitor` and `firm-reporter` ECS services from
`desired_count = 1` to `desired_count = N` in
[`infra/terraform/modules/compute/`](../infra/terraform/modules/compute/),
add an ALB target group, configure healthcheck on the existing
heartbeat-served `/healthz` endpoint.

**Risk.** Low. Both agents are explicitly designed for horizontal scaling;
the take-home runs them as N=1 only because single-host is the operational
scope, not because the code requires it.

### 2.2 Research, PM — stateless, replicatable, idempotent per `decision_id`

**Take-home mechanism.** Research ([`firm/agents/research.py:731`](../firm/agents/research.py)
`make_research`) and PM ([`firm/agents/pm.py:550`](../firm/agents/pm.py)
`make_pm`) are also closure-based. Both consume `WorkingState`, both emit
`Decision` rows, and neither mutates anything outside the on-graph state.
PM deliberation is the vote-of-three pattern (Quality / Valuation / Catalyst
voters aggregated by majority) — the entire deliberation is a pure function
of the retrieved chunks and the voter prompts.

**Production target.** N replicas, same as Monitor/Reporter, with one
additional invariant: **deliberation must be idempotent within a `decision_id`**.
If two replicas pick up the same heartbeat, both must produce the same
`Decision` for the same `decision_id`. This already holds in the take-home
because: (1) LLM responses are deterministic in `replay` mode via
[`firm/llm/cassettes.py:39`](../firm/llm/cassettes.py), (2) the RNG is
seeded ([`firm/core/random.py:42`](../firm/core/random.py)), (3) retrieval is
PIT-filtered ([`firm/rag/retrieve.py:136`](../firm/rag/retrieve.py)). In
production, the same determinism foundation (see [`docs/eval.md`](eval.md)
§2) ensures duplicate deliberation produces duplicate decisions, and the
outbox idempotency-key collapse handles the storage side.

**Migration shape.** Config change plus an explicit at-most-once write barrier
on the `decisions` table — currently the `id` column is the primary key, so
duplicate inserts already collapse to a single row by `ON CONFLICT DO NOTHING`.

**Risk.** Low-medium. The medium is because the eval-grade determinism
foundation is what makes idempotency-by-construction work; if a production
deployment turns off cassette replay and uses live LLM calls, the same
`decision_id` may produce two different decisions across replicas. The fix
is: continue using deterministic LLM modes in production (production cassettes
are recorded against the production prompt set), or accept that the first
write wins and the second is discarded (which is correct per the schema
contract but loses one replica's work).

### 2.3 Risk, Execution — singleton, single-writer to broker

**Take-home mechanism.** Risk
([`firm/agents/risk.py:77`](../firm/agents/risk.py) `evaluate_risk`) is a
pure function of `RiskInput`; Execution
([`firm/agents/execution.py:66`](../firm/agents/execution.py) `make_execution`)
is the **only** agent constructed with a broker handle. The single-host
deployment runs both in the same process as the orchestrator. The outbox
([`firm/outbox/outbox.py:51`](../firm/outbox/outbox.py) `place_order_via_outbox`)
provides exactly-once semantics against the broker via SHA-256 idempotency
keys (per spec §5.2). Even with a single writer, the outbox is the
durability primitive: it's what makes crash-mid-submit safe.

**Production target.** This is the load-bearing row. Execution is
fundamentally singleton — only one process can hold authoritative write
access to the broker. HA for Execution is **leader election with fast
failover**, not horizontal scaling. The mechanism: deploy N=2 (or N=3 for
quorum) execution replicas, run an etcd or Consul leader election, and only
the leader holds the broker handle. On leader loss, the standby promotes
within ~5 seconds (the etcd lease timeout), picks up the outbox `pending`
rows via [`firm/outbox/outbox.py`](../firm/outbox/outbox.py)'s
`recover_pending`, and retries each with the **same idempotency key** — the
broker dedupes any in-flight orders the previous leader already submitted.

**Migration shape.** Net-new component (leader election library + lease
management) plus a deployment topology change. No code change to the
existing Execution agent: it already submits via the outbox, and the outbox
already preserves idempotency keys across crashes. The leader-election
shim is a single-purpose process that holds the broker handle and shells
out to the existing execution code-path.

**Risk.** High. This is the highest-risk migration in the doc. Failure
modes: split-brain (two replicas both believe they are leader and both
submit orders — caught by broker-side idempotency-key dedupe, but only if
the dedupe is configured), failover gap (no leader for >5s, heartbeats
queue up), spurious failover (network partition triggers re-election under
load). All three are well-known patterns with well-understood mitigations,
but they require staging validation against a realistic load profile
before production cut-over. The outbox pattern (§5.2 of spec) is what
makes this safely possible — without it, hot-standby would double-fire
orders.

### 2.4 SQLite (`firm.db`) — single-writer fundamental → Postgres

**Take-home mechanism.** SQLite with WAL mode + `synchronous=FULL` +
`foreign_keys=ON` ([`firm/db/connection.py:8`](../firm/db/connection.py)
`get_conn`). Continuous backup via Litestream
([`config/litestream.yml`](../config/litestream.yml)) gives RPO ~seconds and
RTO ~seconds via restore. The LangGraph checkpointer uses `SqliteSaver`
([`firm/orchestrator/graph.py:83`](../firm/orchestrator/graph.py)).

**Production target.** Postgres via SQLAlchemy `DATABASE_URL` + LangGraph
`PostgresSaver` (a one-import swap per spec §5.3 line 312). The full
deep-dive is in §3 below.

**Migration shape.** Code swap in two places (`connection.py` and
`graph.py`) plus migration tooling (Alembic) and a one-time data migration
from SQLite to Postgres. Schemas, queries, and the outbox pattern are
mostly portable as-is; some SQLite-specific syntax (`INTEGER PRIMARY KEY
AUTOINCREMENT`, `ON CONFLICT DO NOTHING`) needs verification against
Postgres equivalents (`BIGSERIAL`, `ON CONFLICT DO NOTHING` is portable).

**Risk.** Medium. The schema layer is the bulk of the work; the application
layer barely notices. The risk concentrates in the one-time data migration:
NULL semantics differ (SQLite stores zero-length blobs as NULL; Postgres
distinguishes them), and a botched migration can silently corrupt the
audit_log table. Validation gate: re-run the full eval (per
[`docs/eval.md`](eval.md) §2.6) against the Postgres-backed firm and
assert byte-for-byte report equivalence with the SQLite-backed firm.

### 2.5 Qdrant — stateless w.r.t. business state

**Take-home mechanism.** Single Qdrant container in
[`docker-compose.yml`](../docker-compose.yml) (`qdrant/qdrant:v1.11.0` on
port 6333). The vector store ([`firm/rag/qdrant_store.py:36`](../firm/rag/qdrant_store.py)
`VectorStore`) wraps a single `QdrantClient`. On loss of the container, the
operator re-runs `python -m firm.cli ingest` from
[`firm/rag/ingest.py`](../firm/rag/ingest.py) against the source corpus
(FinanceBench + SEC filings + news) and the index is rebuilt within hours.
Embeddings are deterministic given the same input chunks, so a rebuilt
index produces identical retrieval results.

**Production target.** Qdrant Cloud (managed) or a self-hosted replicated
cluster. Qdrant supports built-in replication via the "replication_factor"
collection setting; production would set `replication_factor=3` and use the
distributed deployment topology.

**Migration shape.** Config change only — replace the single
`QDRANT_URL=http://qdrant:6333` env var with the cluster's discovery URL
(or the Qdrant Cloud endpoint). No code change; `QdrantClient` natively
supports cluster mode.

**Risk.** Low. The corpus is rebuildable from source, so even a total
cluster loss is recoverable within hours. The risk is not data loss — it's
retrieval-time latency variance under cluster load, which has no bearing on
correctness and only affects heartbeat budget.

### 2.6 MCP servers — stateless

**Take-home mechanism.** The two MCP-style tools (Fundamentals at
[`firm/tools/fundamentals.py:53`](../firm/tools/fundamentals.py)
`FundamentalsTool`, RiskMetrics at
[`firm/tools/risk_metrics.py:34`](../firm/tools/risk_metrics.py)
`RiskMetricsTool`) are stateless PIT parquet lookups. On crash, the
operator restarts the host process; the tool is re-imported on the next
agent invocation.

**Production target.** Multi-replica MCP server deployment; clients
reconnect on backend failure. AgentCore Gateway is MCP-native and ingests
the existing `ToolDef` descriptors without modification (per
[`docs/agentcore_mapping.md`](agentcore_mapping.md) "MCP Servers → AgentCore
Gateway"), so the production target is a Gateway-managed deployment with
multiple replicas behind the Gateway's load balancer.

**Migration shape.** Config change. The tools already implement the MCP
`tool_def` schema; the operational change is deploying them as
Gateway-managed endpoints rather than in-process callables.

**Risk.** Low. Tools are stateless parquet lookups. A backend failure
surfaces as a `KeyError` which the existing `INSUFFICIENT_EVIDENCE`
failure-mode chain handles (per [`docs/eval.md`](eval.md) §5.10).

---

## 3. Postgres swap (deep-dive)

This is the most impactful and clearest single swap in the doc, so it gets
its own section. Spec §5.3 line 312 frames the swap as "a one-import
swap" — that is mostly accurate, with the caveats below.

### 3.1 Current code paths

The take-home uses SQLite in two distinct ways, each of which has a different
migration shape:

1. **Business state** — the firm's own tables (decisions, outbox, positions,
   audit_log, cost_ledger, hitl_queue, etc.) defined in
   [`firm/db/schema.sql`](../firm/db/schema.sql) and accessed via
   [`firm/db/connection.py:8`](../firm/db/connection.py)'s `get_conn`.
2. **LangGraph checkpoint state** — the graph's per-node checkpoint
   payloads, persisted by `SqliteSaver` at
   [`firm/orchestrator/graph.py:83`](../firm/orchestrator/graph.py) using
   the `_FIRM_SERDE` msgpack allowlist
   ([`firm/orchestrator/graph.py:20`](../firm/orchestrator/graph.py)).

Both currently land in the same `firm.db` file. In production, both move
to the same Postgres database (a single `DATABASE_URL`), but the swap
mechanics differ.

### 3.2 What changes

**Connection module.** [`firm/db/connection.py`](../firm/db/connection.py)
currently uses `sqlite3` directly and applies SQLite-specific PRAGMAs (WAL,
`synchronous=FULL`, `foreign_keys=ON`, `wal_autocheckpoint`). Production
replaces this with a SQLAlchemy engine constructed from `DATABASE_URL` and
configures Postgres-equivalent settings via the engine URL or session-level
`SET` statements (e.g. `synchronous_commit=on` is the Postgres equivalent
of `synchronous=FULL`).

**Checkpointer.** Line 83 of `graph.py` changes from `SqliteSaver(conn,
serde=_FIRM_SERDE)` to `PostgresSaver(conn, serde=_FIRM_SERDE)` — a literal
one-import swap. `PostgresSaver` is part of the same
`langgraph.checkpoint` package and shares the saver protocol. The
`_FIRM_SERDE` msgpack allowlist applies unchanged.

**Migration tooling.** The take-home applies schema via
[`firm/db/migrations.py:10`](../firm/db/migrations.py) `init_db`, which
reads `schema.sql` and runs `executescript`. Production needs versioned
migrations (Alembic is the standard for SQLAlchemy projects) so that
schema changes can be rolled forward and back across deployments without
data loss. Alembic adoption is the largest piece of net-new tooling in
the migration.

### 3.3 What does not change

- **Application-layer queries.** The SQL in the firm is mostly portable:
  basic `INSERT`, `SELECT`, `UPDATE`, `DELETE` with parameter binding.
  `ON CONFLICT DO NOTHING` (used in the outbox) is portable to Postgres.
- **The outbox pattern.** [`firm/outbox/outbox.py:51`](../firm/outbox/outbox.py)
  `place_order_via_outbox` is DB-agnostic by construction (the comments in
  spec §5.2 use generic SQL). The transaction boundaries and the
  idempotency-key invariant carry over unchanged.
- **Schemas.** [`firm/db/schema.sql`](../firm/db/schema.sql) needs minor
  syntax adjustments (e.g. `INTEGER PRIMARY KEY AUTOINCREMENT` →
  `BIGSERIAL PRIMARY KEY`), but the column types, constraints, and indexes
  carry over directly.

### 3.4 What needs verification

- **SQLite-specific syntax.** `executescript` runs multiple statements
  separated by semicolons; SQLAlchemy's `execute` runs one at a time. The
  schema-init path is rewritten to loop over statements.
- **NULL vs. empty-string semantics.** SQLite is loose about coercing
  empty strings to NULL in some indexed contexts; Postgres is strict.
  Audit-log tests need to assert empty-string handling explicitly during
  the data migration.
- **Boolean storage.** SQLite stores booleans as INTEGER 0/1; Postgres
  has a native `BOOLEAN` type. Pydantic models that round-trip through
  the DB need a deserialization pass against the migrated rows.

The validation gate is exactly the determinism gate from
[`docs/eval.md`](eval.md) §2.6:
[`scripts/check_reports_clean.sh`](../scripts/check_reports_clean.sh) runs
`make eval` twice and diffs. If the Postgres-backed firm produces the
same byte-for-byte reports as the SQLite-backed firm, the migration is
correct. If it doesn't, the diff names exactly the column or row that
drifted.

---

## 4. Multi-region AgentCore

Production needs multi-region failover for the Reporter Runtime. The
current single-region Terraform module is the baseline.

### 4.1 Current single-region module

[`infra/terraform/modules/bedrock/main.tf`](../infra/terraform/modules/bedrock/main.tf)
provisions one IAM role (`aws_iam_role.agentcore_runtime`) and one
CloudWatch log group (`aws_cloudwatch_log_group.agentcore_reporter`) in a
single region (us-east-1 per the default `var.aws_region`). The symbolic
AgentCore entity names (`firm-reporter` runtime,
`firm-desk-state` memory namespace) are encoded as Terraform locals at
[`infra/terraform/modules/bedrock/main.tf:36`](../infra/terraform/modules/bedrock/main.tf)
and emitted as outputs for the T39 CLI step to consume.

### 4.2 Multi-region production topology

The production target is N regions (typically us-east-1 + us-west-2 for
US deployments, plus eu-west-1 for European data residency if applicable).
The shape:

1. **Per-region module instantiation.** Use Terraform `for_each` over a
   `var.regions` list, with the AWS provider's `alias` mechanism to scope
   each module instance to its region.
2. **Cross-region Memory replication.** AgentCore Memory is regional (per
   the SDK risks called out in [`docs/agentcore_mapping.md`](agentcore_mapping.md)
   §"Open Questions / SDK Risks"). Multi-region deployment requires a Memory
   namespace per region, plus an active-active sync mechanism (likely
   DynamoDB Global Tables under the hood once AgentCore exposes a managed
   cross-region option; until then, a custom replication shim that
   subscribes to one region's Memory writes and replays them into the
   others).
3. **Route 53 health-based failover.** A Route 53 hosted zone fronts the
   AgentCore Runtime endpoints. Health checks ping the Reporter's
   AgentCore endpoint per region; failure routes traffic to the next
   region in the priority list.
4. **IAM trust boundaries per region.** Each region's IAM role
   (`firm-prod-agentcore-runtime` in region X) trusts only that region's
   AgentCore service principal — no global wildcard.

### 4.3 Migration shape

Terraform-only. The application code does not change. The AgentCore
adapter at `firm/agentcore/reporter_adapter.py` (per
[`docs/agentcore_mapping.md`](agentcore_mapping.md) §"Reporter Adapter
Design Sketch") is region-agnostic: it reads its config from env vars
that the per-region ECS task definition sets.

### 4.4 Risk

Medium. The hard problem is cross-region Memory consistency: a Reporter
invocation in us-west-2 must see the same desk state that a previous
invocation in us-east-1 wrote. Until AgentCore Memory exposes a managed
cross-region option, the custom replication shim is a net-new component
that needs its own failure-mode coverage. The Reporter is the lowest-blast-radius
agent (per "Why Reporter First" in
[`docs/agentcore_mapping.md`](agentcore_mapping.md)), so a Memory replication
bug surfaces as a malformed report row, not an unintended trade.

---

## 5. Eval at scale (Inspect AI)

The production target for eval is Inspect AI — the UK AI Security
Institute's open-source eval framework. The take-home uses a custom
pytest + Jinja harness. The trade-off is covered in detail in
[`docs/eval.md`](eval.md) §8 ("Inspect AI reference") and we do not
duplicate it here.

**Summary of the migration path** (per [`docs/eval.md`](eval.md) §8 last
paragraph):

- Keep the determinism gate
  ([`scripts/check_reports_clean.sh`](../scripts/check_reports_clean.sh))
  — it is framework-agnostic.
- Re-implement each `compute_<metric>` function in
  [`firm/eval/process_metrics.py`](../firm/eval/process_metrics.py) as an
  Inspect AI scorer.
- Re-implement the Jinja templates as Inspect AI report writers.
- Parallelize via Inspect AI's task runner.

Estimated migration cost: roughly the size of
[`firm/eval/runner.py`](../firm/eval/runner.py) plus the two Jinja
templates, ~500 LOC. The data shapes (`MetricResult`, the regime context
dict) survive the migration; only the formatting and orchestration layers
change. The three regime windows in
[`firm/eval/regimes.py:91-109`](../firm/eval/regimes.py) (`R1_EARNINGS`,
`R2_DRAWDOWN`, `R3_QUIET`) carry over unchanged.

**Risk.** Low-medium. Inspect AI's scorer abstraction maps cleanly to our
`MetricResult` dataclass, and the framework's parallelism story is well
understood. The medium-risk piece is the report-writer rewrite: byte-for-byte
determinism through a third-party formatter is harder than through our own
Jinja templates. The mitigation is to keep the determinism gate as the CI
contract regardless of framework, and accept that the report bytes may
change once at the migration cut-over (a one-time recapture of the
committed `reports/eval/` baseline).

---

## 6. GHE and private package mirroring

The take-home repo lives on GitHub.com. Production lives on GitHub
Enterprise (GHE), either self-hosted or GHE Cloud with data residency
controls. The migration is a `git remote set-url` plus a CI re-wire; the
substantive change is supply-chain hardening.

### 6.1 What changes

- **Repo host.** GHE replaces GitHub.com. CI runners are self-hosted
  (GHE-Runners) inside the firm's VPC.
- **Private package mirror.** PyPI, npm, Docker Hub are all replaced (or
  proxied) by an internal Artifactory or Verdaccio mirror. The mirror is
  the single source for all dependency installs; outbound network egress
  from CI runners is restricted to the mirror's IP range.
- **SBOM + signed releases.** Every release build produces a CycloneDX
  SBOM, signed with the firm's release key. The CI gate refuses to deploy
  a build without a valid SBOM.

### 6.2 Migration shape

CI rewire and a mirror provisioning project. No application-code change.
The existing `pyproject.toml` and `requirements.txt` pin every direct
dependency to a specific version; the mirror just serves those versions
from its own storage.

### 6.3 Risk

Low. The take-home's CI is already configured to install from
`requirements-lock.txt`; the mirror just changes the URL. The risk is
operational (the mirror itself becoming a single point of failure for all
internal builds), which is the standard Artifactory HA configuration.

---

## 7. KMS-rotated secrets, cross-region replication

The take-home ships customer-managed KMS encryption-at-rest with annual
key rotation. Production adds Lambda-driven secret-value rotation and
cross-region replica sets.

### 7.1 Current secrets module

[`infra/terraform/modules/secrets/main.tf`](../infra/terraform/modules/secrets/main.tf)
provisions:

- One customer-managed KMS key with `enable_key_rotation = true`
  ([`infra/terraform/modules/secrets/main.tf:35-43`](../infra/terraform/modules/secrets/main.tf))
  — AWS rotates the backing key material annually.
- One KMS alias for human-readable references.
- Six Secrets Manager entries via `for_each` over a local set
  ([`infra/terraform/modules/secrets/main.tf:74-93`](../infra/terraform/modules/secrets/main.tf)):
  `firm/anthropic_api_key`, `firm/slack_signing_secret`,
  `firm/slack_bot_token`, `firm/firm_hmac_secret`,
  `firm/firm_hmac_secret_prev`, `firm/firm_hmac_rotated_at`.

Secret **values** are written out-of-band by operators after the first
`apply` — Terraform manages the entry but not the contents.

### 7.2 Production additions

**Rotation Lambdas.** Each secret gets a `rotation_lambda_arn` and a
`rotation_rules` block (typically 30-day cadence). The Lambda implements
the AWS-standard `createSecret` / `setSecret` / `testSecret` /
`finishSecret` four-step rotation protocol. For the HMAC secrets, the
rotation Lambda generates a new high-entropy value, writes it to
`firm/firm_hmac_secret`, demotes the previous value to
`firm/firm_hmac_secret_prev`, and timestamps the change in
`firm/firm_hmac_rotated_at` — which is exactly the dual-key contract
[`firm/hitl/signing.py:172`](../firm/hitl/signing.py)'s
`verify_with_rotation` expects.

**Cross-region replication.** Add `replica { region = "us-west-2"
kms_key_id = aws_kms_key.secrets_uswest2.arn }` to each
`aws_secretsmanager_secret`. Secrets Manager handles the cross-region sync;
the application reads from its local-region replica.

### 7.3 Migration shape

Terraform-only. The application code does not change — it already calls
`aws secretsmanager get-secret-value` via the boto3 client, which
transparently uses the local-region endpoint. The
`verify_with_rotation` dual-key scheme is already production-shaped (per
Plan 3 T35).

### 7.4 Risk

Medium. The risk concentrates in the rotation Lambdas: a bug in
`finishSecret` that promotes the new key before all consumers have
fetched it causes a transient verification-failure window where signed
approvals are rejected. The mitigation is the dual-key grace window in
`verify_with_rotation` (default 24 hours — per
[`firm/hitl/signing.py:172`](../firm/hitl/signing.py)), which is **already
designed for exactly this case**. Production rotation Lambdas must
respect the grace window when sequencing the four steps.

---

## 8. Multi-vendor observability (Datadog/Honeycomb + CloudWatch)

The take-home exports OTLP traces to CloudWatch via the observability
Terraform module. Production layers vendor-grade APM (Datadog or
Honeycomb) on top, keeping CloudWatch for IAM/audit logs.

### 8.1 Current observability stack

- **Local dev path.** [`firm/obs/tracer.py:60`](../firm/obs/tracer.py)
  `JsonlFileExporter` writes spans to
  `traces/YYYY-MM-DD/run-<run_id>.jsonl`. Used by `OTEL_EXPORTER=file`
  (default).
- **Production-shaped path.** [`firm/obs/tracer.py:255`](../firm/obs/tracer.py)
  `init_tracer` honours `OTEL_EXPORTER=otlp` and lazy-imports the OTLP
  gRPC exporter ([`firm/obs/tracer.py:314`](../firm/obs/tracer.py)
  `_init_otlp_provider`). Already implemented; production just sets
  the env var.
- **Terraform.** [`infra/terraform/modules/observability/main.tf`](../infra/terraform/modules/observability/main.tf)
  provisions a CloudWatch log group `/firm/<env>`, an otelcol-contrib
  Fargate service listening on 4317/4318 inside the VPC, and a 4-widget
  CloudWatch dashboard. The collector ingests OTLP and exports to
  CloudWatch.

### 8.2 Production additions

**Dual export.** Configure the otelcol collector with two exporters:
`awscloudwatchlogs` (existing) **plus** `datadog` (or `otlphttp` pointed
at Honeycomb's ingest endpoint). The OpenTelemetry collector's pipeline
DSL handles dual-export natively — no application code change.

**Why both, not just vendor APM.** CloudWatch retains its role as the
audit/IAM-side observability backbone — the same CloudWatch log group
that AgentCore writes to (per
[`infra/terraform/modules/bedrock/main.tf:152`](../infra/terraform/modules/bedrock/main.tf))
needs to remain queryable via CloudWatch Logs Insights so security/compliance
teams have a single AWS-native pane. Datadog or Honeycomb adds the
distributed-tracing, flame-graph, and high-cardinality query capabilities
that CloudWatch is comparatively weak at.

### 8.3 Migration shape

Config change. Add the vendor's exporter to the otelcol config in
[`infra/terraform/modules/observability/main.tf`](../infra/terraform/modules/observability/main.tf),
add a Secrets Manager entry for the vendor's API key (using the same
`firm/...` pattern as the existing secrets in §7), grant the collector's
task role `secretsmanager:GetSecretValue` on the new ARN. No application
code change.

### 8.4 Risk

Low. Span emission is fire-and-forget; a misconfigured exporter drops
traces but does not corrupt decisions or orders (per the blast-radius
note in [`docs/agentcore_mapping.md`](agentcore_mapping.md) §"OpenTelemetry
Traces → AgentCore Observability"). The vendor APM is an addition, not a
replacement — if it fails, CloudWatch keeps recording.

---

## 9. What stays the same

This section is the most important part of the doc. The migration deltas
above are about **operating at firm scale**, not about correctness. The
load-bearing invariants — the things that make the firm safe to run with
real money — are production-ready as-is. We name them explicitly so a
reader cannot miss it.

### 9.1 The five architectural invariants (from threat model)

The architectural barriers in [`docs/threat_model.md`](threat_model.md) §3,
§5, and §7 do not move during the migration:

- **Structured outputs.** `Decision.action: ActionEnum`
  ([`firm/core/models.py:127`](../firm/core/models.py),
  [`firm/core/models.py:11`](../firm/core/models.py)) rejects any value
  outside the five enum members. This is a schema barrier, not a
  deployment barrier.
- **Least-privilege MCP per agent.** Only
  [`firm/agents/execution.py:66`](../firm/agents/execution.py)
  `make_execution` takes `broker: Broker`. Research, PM, risk, HITL,
  reporter, monitor have no broker handle and never can. This is a
  construction barrier, not a deployment barrier.
- **HITL gate.** The orchestrator graph is compiled with
  `interrupt_before=["hitl"]`
  ([`firm/orchestrator/graph.py:84`](../firm/orchestrator/graph.py)).
  Every execution path passes through a Slack approval signed with HMAC
  over `(decision_id, approver_id, ts)`
  ([`firm/hitl/signing.py:109`](../firm/hitl/signing.py)). This is a
  graph-topology barrier, not a deployment barrier.
- **Signed-approval HMAC chain.** `verify_with_rotation`
  ([`firm/hitl/signing.py:172`](../firm/hitl/signing.py)) supports
  dual-key rotation with a 24-hour grace window. This is exactly the
  contract a production rotation Lambda needs (see §7) — no API change.
- **Red-team architectural invariants.** The five invariants in
  [`docs/threat_model.md`](threat_model.md) §7 (`assert_no_privileged_action`,
  `assert_no_schema_bypass`, `assert_no_unapproved_trade`,
  `assert_no_forged_citation`, `assert_no_forged_approval`) all assert
  against the audit_log, outbox, and broker mock. None of those channels
  changes during the migration — Postgres has audit_log rows just like
  SQLite did.

### 9.2 Determinism foundation

The six determinism mechanisms in [`docs/eval.md`](eval.md) §2 are
production-ready:

- Clock injection ([`firm/core/clock.py:17`](../firm/core/clock.py))
- VCR LLM cassettes ([`firm/llm/cassettes.py:39`](../firm/llm/cassettes.py))
- PIT-filtered RAG ([`firm/rag/retrieve.py:136`](../firm/rag/retrieve.py))
- Deterministic broker fills
  ([`firm/broker/fake_broker.py:96`](../firm/broker/fake_broker.py))
- Frozen RNG seed ([`firm/core/random.py:42`](../firm/core/random.py))
- Determinism gate
  ([`scripts/check_reports_clean.sh`](../scripts/check_reports_clean.sh))

The same gate is the validation contract for every migration delta in
this doc — re-run the eval after each delta and assert byte-for-byte
report equivalence.

### 9.3 Decision schema and contracts

`Decision`, `Claim`, `Citation`, `ActionEnum`, `FailureMode`, and the
payload subclasses in [`firm/core/models.py`](../firm/core/models.py) are
the contracts that every agent crosses. They do not change in production.
The `FailureMode` enum
([`firm/core/models.py:19`](../firm/core/models.py)) is fully covered by
fixtures (14/14, `ALLOWED_GAPS` empty per Plan 4 T26 — see
[`docs/eval.md`](eval.md) §5.10), and the coverage invariant carries over
to production unchanged.

### 9.4 The outbox pattern

[`firm/outbox/outbox.py:51`](../firm/outbox/outbox.py)
`place_order_via_outbox` is the durability primitive that makes Risk and
Execution's singleton-with-failover topology safely possible (per §2.3
above). The SHA-256 idempotency key contract is DB-agnostic, transport-agnostic,
and broker-agnostic. It survives the Postgres swap (§3), the multi-region
deployment (§4), and any future broker migration.

### 9.5 The take-home is not a toy

The above invariants are non-trivial. They are what a senior reviewer
would look for in any production trading system: typed outputs, least-privilege
construction, signed approvals, exactly-once writes, deterministic
reproduction. The take-home ships all five. The migration deltas in this
doc are about **scale and operations**, not about correctness — they
extend a working firm from single-host to firm-scale without rewriting
the load-bearing parts.

---

## 10. Cross-reference summary

| Production delta | Doc section | Code paths | Migration risk |
|---|---|---|---|
| Postgres swap | §3 | [`firm/db/connection.py:8`](../firm/db/connection.py), [`firm/orchestrator/graph.py:83`](../firm/orchestrator/graph.py), [`firm/db/schema.sql`](../firm/db/schema.sql) | Medium |
| Multi-region AgentCore | §4 | [`infra/terraform/modules/bedrock/main.tf`](../infra/terraform/modules/bedrock/main.tf), [`docs/agentcore_mapping.md`](agentcore_mapping.md) | Medium |
| Inspect AI eval | §5 | [`docs/eval.md`](eval.md) §8 | Low-medium |
| GHE + private mirror | §6 | (CI + supply-chain, no application code) | Low |
| KMS-rotated secrets + replication | §7 | [`infra/terraform/modules/secrets/main.tf`](../infra/terraform/modules/secrets/main.tf), [`firm/hitl/signing.py:172`](../firm/hitl/signing.py) | Medium |
| Multi-vendor observability | §8 | [`firm/obs/tracer.py:255`](../firm/obs/tracer.py), [`infra/terraform/modules/observability/main.tf`](../infra/terraform/modules/observability/main.tf) | Low |
| LB-fronted Monitor/Reporter | §2.1 | [`firm/agents/monitor.py:10`](../firm/agents/monitor.py), [`firm/agents/reporter.py:88`](../firm/agents/reporter.py) | Low |
| N-replica Research/PM | §2.2 | [`firm/agents/research.py:731`](../firm/agents/research.py), [`firm/agents/pm.py:550`](../firm/agents/pm.py) | Low-medium |
| Leader election for Risk/Execution | §2.3 | [`firm/agents/risk.py:77`](../firm/agents/risk.py), [`firm/agents/execution.py:66`](../firm/agents/execution.py), [`firm/outbox/outbox.py:51`](../firm/outbox/outbox.py) | High |
| Qdrant cluster | §2.5 | [`firm/rag/qdrant_store.py:36`](../firm/rag/qdrant_store.py), [`docker-compose.yml`](../docker-compose.yml) | Low |
| MCP server multi-replica | §2.6 | [`firm/tools/fundamentals.py:53`](../firm/tools/fundamentals.py), [`firm/tools/risk_metrics.py:34`](../firm/tools/risk_metrics.py) | Low |

| Authoritative source | Path |
|---|---|
| Spec §3.6 (availability model) | [`docs/superpowers/specs/2026-05-18-ai-investment-firm-design.md`](superpowers/specs/2026-05-18-ai-investment-firm-design.md) lines 155-170 |
| Spec §5.3 (LangGraph checkpointer swap) | [`docs/superpowers/specs/2026-05-18-ai-investment-firm-design.md`](superpowers/specs/2026-05-18-ai-investment-firm-design.md) line 312 |
| Plan 4 task spec | [`docs/superpowers/plans/2026-05-21-eval-redteam-cicd-deploy.md`](superpowers/plans/2026-05-21-eval-redteam-cicd-deploy.md) T44 |
| AgentCore migration mapping | [`docs/agentcore_mapping.md`](agentcore_mapping.md) |
| Eval harness (Inspect AI reference) | [`docs/eval.md`](eval.md) §8 |
| Threat model (invariants that survive) | [`docs/threat_model.md`](threat_model.md) |

---

*Document generated for Plan 4 T44.*
