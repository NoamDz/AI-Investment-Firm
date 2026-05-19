# Plan 1 — what it is, in plain English

Plan 1 builds the **skeleton** of an AI investment firm. The bones are real; the muscles (the LLM-powered analysis) come in Plan 2. Today, if you run `make demo`, it walks all the way from "agent decides to buy AAPL" to "paper broker confirms the trade" — using a dummy research agent that always picks AAPL, but with every piece of production infrastructure in place around it.

There are basically **four ideas** worth understanding. Everything else is plumbing.

---

## Idea 1: Every agent speaks the same language

Every agent in the system — the researcher, the portfolio manager, the risk checker, the executor — returns the same object: a `Decision`.

A `Decision` has things like:
- an **id** (unique, like `01KRYYHHM5...`)
- the **chain of decisions that led to it** (so you can trace "this trade happened because of *that* analysis")
- an **action**: BUY, SELL, HOLD, ESCALATE, or REFUSE
- a **payload** matching the action (BUY carries a ticker + shares; REFUSE carries a reason)
- a **rationale** in plain English
- a **confidence** number
- a **falsification condition** — "this decision is wrong if X happens" (this is the spec's anti-hallucination trick: forces the agent to commit to what would prove it wrong)
- a **nonce** — a cryptographic signature that makes the decision tamper-evident

**Why this matters:** because every agent emits the same shape, you can chain them, replay them, audit them, and store them in one table. The whole rest of the system is built on top of "I trust that whatever flows through here is a `Decision`." When the research agent becomes LLM-powered in Plan 2, the interface doesn't change — only the implementation behind it.

The discriminated payload (`firm/core/models.py:50-86`) means the type checker knows that if `action == BUY`, then `payload.ticker` and `payload.shares` exist. No `if hasattr(...)` defensive code. This is what made the I3 bug obvious during validation — the CLI was using `hasattr` to peek at the payload, which silently misbehaved for HOLD/REFUSE/ESCALATE. We replaced it with proper type narrowing.

---

## Idea 2: The outbox — how you avoid placing the same trade twice

This is the single most important piece of the whole project, so let me be detailed.

**The problem:** You want to submit an order to your broker and record it in your local DB. If you crash between those two steps, what happens?

- Submit first, then record locally → if you crash after submit, you have a real trade with no record of it. Next time you boot, you might submit it again. **Double-bought.**
- Record locally first, then submit → if you crash after recording, you never actually placed the order. **Phantom trade in your DB.**

Neither is acceptable when real money is involved.

**The solution — the outbox pattern** (`firm/outbox/outbox.py`):

```
1. Open a SQLite transaction.
2. INSERT INTO outbox (key, status='pending', ...) ON CONFLICT DO NOTHING.
3. COMMIT.
4. NOW call broker.submit(idempotency_key=key).
5. UPDATE outbox SET status='confirmed', result=<broker response>.
```

The key insight is step 2: the `key` is `sha256(decision_id + nonce)` — a value that's **derived deterministically** from the decision itself. We use the **same key** to talk to the broker.

Now think about every crash point:

- **Crash before step 2 commits** → nothing happened anywhere. Fine.
- **Crash between 3 and 4** → outbox has a `pending` row. On reboot, `recover_pending()` finds it, replays step 4 with the same key. The broker sees the same key and **deduplicates** — Alpaca's API is built for this. Either we get back the order it already placed, or it places it for the first time. Either way, exactly one order.
- **Crash between 4 and 5** → broker has the order, outbox still says `pending`. Same recovery: replay step 4, get the same answer back, then mark `confirmed`.

The reason this works is: **the broker is the tiebreaker.** It uses the idempotency key as the source of truth for "have I seen this before?" SQLite is the system of record for our intent. We trust the broker to be exactly-once on its side, and we use SQLite to remember what we asked for. The combination is exactly-once end-to-end.

There's a test that literally kills the Python process mid-order (`tests/integration/test_crash_recovery.py`) and verifies that on restart, you get exactly one filled order, not zero and not two.

---

## Idea 3: SQLite is the brain, and foreign keys hold it together

Everything important goes into one SQLite file with WAL mode on (so reads don't block writes) and foreign keys enforced.

The main tables:

- **`decisions`** — every `Decision` any agent ever made. Append-only. This is the audit log.
- **`outbox`** — orders we've sent or are about to send. Has a `FOREIGN KEY` to `decisions`.
- **`hitl_queue`** — decisions waiting for human approval. Also has a `FOREIGN KEY` to `decisions`.
- **`positions`** and **`cash`** — local mirror of what the broker thinks we own.
- **`audit_log`** — append-only log of events (approvals, halts, etc.) separate from decisions.
- **`reconciliations`** — when boot-time reconciliation runs, the diff between local state and broker state is saved here.

**Why foreign keys matter — the bug they caught:**

During post-implementation validation, the HITL flow crashed with `FOREIGN KEY constraint failed`. Here's what was happening:

1. Risk agent produces an `ESCALATE` decision.
2. Graph routes to HITL node.
3. HITL tries `INSERT INTO hitl_queue (decision_id, ...)` where `decision_id` points to the ESCALATE decision.
4. **But that ESCALATE decision was never written to `decisions` table.**
5. FK violation. Crash.

The fix (`firm/agents/hitl.py:23`) was one line: persist the risk decision into `decisions` *before* inserting into `hitl_queue`. The FK literally pointed at the bug. If we hadn't enforced FKs, this would have silently created orphan rows that broke audit traceability — exactly the kind of thing you'd discover months later when reading logs and going "wait, where did this come from?"

**Why SQLite at all** (instead of Postgres)? Because the whole firm fits in a single process on a single machine. SQLite in WAL mode handles this load trivially, has no server to manage, and the database is just a file you can copy. The litestream config is in the tree to stream that file to S3 for durability (turned on in Plan 3).

---

## Idea 4: HITL — making the graph pause for a human

This is the trickiest control-flow piece. LangGraph is the workflow engine; it runs the DAG `monitor → research → pm → risk → (hitl?) → execution → reporter`.

Sometimes risk says "this trade is too big to auto-approve — escalate to a human." We need the graph to **pause**, wait for someone to type `firm ack <decision_id>`, and then **resume from exactly where it stopped**.

LangGraph supports this with two features:
1. **Checkpointer** — saves the entire graph state to SQLite after every node.
2. **`interrupt_before=["hitl"]`** — when about to run the `hitl` node, halt and return.

This produces a **4-step dance**:

1. `graph.invoke({})` — runs monitor → research → pm → risk, then halts before hitl. Returns to the CLI.
2. `graph.invoke(None)` (with the same `thread_id`) — resumes, runs the hitl node. The hitl node inserts a `pending` row into `hitl_queue`. Execution node sees `hitl_required=True, hitl_approved=False` and skips.
3. The human runs `firm ack <decision_id>` from a terminal. This calls `mark_approved()`, which flips `hitl_queue.status` from `pending` to `approved`.
4. `graph.invoke({})` + `graph.invoke(None)` again on the same `thread_id`. Hitl node re-reads `hitl_queue`, sees `approved`, sets `hitl_approved=True`. Execution **unwraps** the ESCALATE payload (which contains the originally proposed BUY) and submits it through the outbox.

The unwrap step deserves a note. An `ESCALATE` decision contains a `proposed` field — the BUY or SELL that risk wanted to flag. After approval, execution doesn't create a *new* decision; it reuses the ESCALATE decision's `id` and `nonce`. Why? Because the outbox idempotency key is `sha256(id + nonce)`. By reusing them, **if step 4 runs twice for any reason, the outbox sees the same key and refuses to double-submit.** Idempotency carries through replay.

This is the kind of design that looks paranoid until you remember: real money, distributed systems, crash recovery, audit trails.

---

## The supporting cast (briefly)

These matter but don't need a deep dive:

- **Clock injection.** Nothing in the code calls `datetime.now()` directly. Every component takes a `Clock` parameter. In production it's `WallClock`; in tests it's `ReplayClock(fixed_time)`. This is how the demo is fully deterministic and why every test can assert on exact timestamps.

- **Nonce signing.** Every `Decision` carries an HMAC signature. If anyone alters a row in the `decisions` table, the signature won't verify. Belt and suspenders alongside the append-only constraint.

- **Two brokers, one interface.** `FakeBroker` (in-memory, used by tests and the demo) and `AlpacaBroker` (real paper-trading API). They implement the same `Broker` Protocol. The rest of the system can't tell which is plugged in.

- **Boot reconciliation.** On startup, ask the broker "what do you actually have?" and diff against the local `positions` and `cash` tables. Write the diff to `reconciliations`. Broker wins any disagreement — it's the source of truth.

---

## So what is Plan 1, in one paragraph?

Plan 1 is a real, working, end-to-end paper-trading pipeline with **exactly-once order semantics, a tamper-evident audit log, deterministic replay, and a working human-in-the-loop approval flow** — but with the actual "decide what to buy" logic replaced by a stub that always picks AAPL. Plan 2 swaps the stub for LLMs grounded in retrieval. Plan 3 hardens operations (Slack approvals, real-time replication, daily reports, tracing). Plan 4 adds the eval harness and CI. The reason to build the skeleton first is that **the hard problems aren't in the LLM call — they're in everything around it**: crash safety, idempotency, audit trails, type safety, replay determinism, and not double-buying when your process dies mid-trade.
