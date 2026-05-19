-- Append-only decision log (the audit log). See spec §3.4.
CREATE TABLE IF NOT EXISTS decisions (
    id              TEXT PRIMARY KEY,
    parent_chain    TEXT NOT NULL,           -- JSON array
    action          TEXT NOT NULL,
    payload         TEXT NOT NULL,           -- JSON
    rationale       TEXT NOT NULL,
    confidence      REAL NOT NULL,
    citations       TEXT NOT NULL,           -- JSON array
    falsification   TEXT NOT NULL,
    escalation      TEXT,
    failure_mode    TEXT,
    metadata        TEXT NOT NULL,           -- JSON
    nonce           TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

-- Transactional outbox for broker orders. See spec §5.2.
CREATE TABLE IF NOT EXISTS outbox (
    key             TEXT PRIMARY KEY,        -- sha256(decision_id || nonce)
    decision_id     TEXT NOT NULL,
    payload         TEXT NOT NULL,           -- JSON Decision
    status          TEXT NOT NULL CHECK (status IN ('pending','confirmed','failed')),
    result          TEXT,                    -- JSON OrderResult after confirm
    error           TEXT,                    -- non-null when status='failed'
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status);
CREATE INDEX IF NOT EXISTS idx_outbox_decision_id ON outbox(decision_id);

-- Local view of broker positions. Source of truth is the broker (§5.7).
CREATE TABLE IF NOT EXISTS positions (
    ticker          TEXT PRIMARY KEY,
    shares          TEXT NOT NULL,           -- Decimal as text
    avg_cost        TEXT NOT NULL,           -- Decimal as text
    updated_at      TEXT NOT NULL
);

-- Local view of broker cash.
CREATE TABLE IF NOT EXISTS cash (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    amount          TEXT NOT NULL,           -- Decimal as text
    updated_at      TEXT NOT NULL
);

-- Decisions awaiting human approval. See spec §3.1, §8.4.
CREATE TABLE IF NOT EXISTS hitl_queue (
    decision_id     TEXT PRIMARY KEY,
    queued_at       TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('pending','approved','rejected','timed_out')),
    approver        TEXT,
    approval_nonce  TEXT,
    decided_at      TEXT,
    FOREIGN KEY (decision_id) REFERENCES decisions(id)
);
CREATE INDEX IF NOT EXISTS idx_hitl_queue_status ON hitl_queue(status);

-- Boot-time and EOD reconciliation results. See spec §5.7.
CREATE TABLE IF NOT EXISTS reconciliations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT NOT NULL CHECK (kind IN ('boot','eod')),
    ran_at          TEXT NOT NULL,
    broker_snapshot TEXT NOT NULL,           -- JSON
    local_snapshot  TEXT NOT NULL,           -- JSON
    diff            TEXT NOT NULL,           -- JSON
    status          TEXT NOT NULL CHECK (status IN ('ok','mismatch','acked'))
);
CREATE INDEX IF NOT EXISTS idx_recon_ran_at ON reconciliations(ran_at);

-- General append-only audit log (covers acks, halts, etc.).
CREATE TABLE IF NOT EXISTS audit_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL,
    event           TEXT NOT NULL,
    detail          TEXT NOT NULL            -- JSON
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
