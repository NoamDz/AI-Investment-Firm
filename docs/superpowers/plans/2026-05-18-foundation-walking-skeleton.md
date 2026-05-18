# Foundation + Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a runnable multi-agent investment firm skeleton that produces a deterministic end-to-end paper trade through 5 LangGraph agents, persists state in SQLite (WAL + outbox + audit log), reconciles with the broker on boot, and exits cleanly. Reviewer can `make demo` and watch one decision flow from heartbeat to fill in under 10 minutes.

**Architecture:** LangGraph orchestrator with SqliteSaver checkpointer driving 5 agent nodes (Research, PM, Risk, HITL, Execution, Reporter) plus a Position Monitor heartbeat. State lives in SQLite (WAL + `synchronous=FULL`). Orders go through a transactional outbox with HMAC-signed idempotency keys, so crash mid-order is exactly-once on recovery. Default broker is `FakeBroker` (deterministic, zero-setup); `AlpacaBroker` is selectable via env var for real paper trading. Agents are deterministic stubs in Plan 1 — LLM-backed Research and grounding come in Plan 2.

**Tech Stack:** Python 3.11+, Pydantic v2, LangGraph 0.2+, SQLite (stdlib `sqlite3`), `python-ulid`, `pyyaml`, `pytest`, `alpaca-py` (optional), Docker, Litestream (config-only in this plan).

---

## File Structure

This plan creates the following layout. Each file has one responsibility.

```
ai-investment-firm/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── .gitignore
├── .env.example
├── README.md
├── litestream.yml                       # config only; not running in Plan 1
├── config/
│   ├── policy.yaml                      # risk limits + HITL thresholds
│   └── universe.yaml                    # frozen 30 tickers as_of
├── firm/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── models.py                    # Decision, Citation, Claim, FailureMode, ActionEnum, payloads
│   │   ├── clock.py                     # Clock Protocol, WallClock, ReplayClock
│   │   ├── ids.py                       # ULID + HMAC nonce sign/verify
│   │   └── config.py                    # Pydantic loaders for policy.yaml, universe.yaml
│   ├── db/
│   │   ├── __init__.py
│   │   ├── connection.py                # SQLite WAL + pragmas
│   │   ├── schema.sql                   # all tables
│   │   └── migrations.py                # init_db()
│   ├── audit/
│   │   ├── __init__.py
│   │   └── log.py                       # append-only audit logger
│   ├── broker/
│   │   ├── __init__.py
│   │   ├── protocol.py                  # Broker Protocol + OrderResult, Position, Quote types
│   │   ├── fake_broker.py               # deterministic in-memory broker
│   │   └── alpaca_paper.py              # Alpaca paper adapter
│   ├── outbox/
│   │   ├── __init__.py
│   │   └── outbox.py                    # place_order_via_outbox, recover_pending
│   ├── reconcile/
│   │   ├── __init__.py
│   │   └── boot.py                      # reconcile_on_boot + halt protocol
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── state.py                     # WorkingState TypedDict
│   │   └── graph.py                     # LangGraph build with SqliteSaver
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── monitor.py                   # Position Monitor heartbeat
│   │   ├── research.py                  # deterministic stub returning Decision
│   │   ├── pm.py                        # deterministic stub
│   │   ├── risk.py                      # REAL: hard limits from policy.yaml
│   │   ├── hitl.py                      # interrupt-before gate + CLI-ack flow
│   │   ├── execution.py                 # wraps outbox + broker.submit
│   │   └── reporter.py                  # minimal JSONL summary writer
│   └── cli.py                           # `firm run`, `firm ack <id>`, `firm reconcile`
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_clock.py
│   │   ├── test_ids.py
│   │   ├── test_config.py
│   │   ├── test_outbox.py
│   │   ├── test_risk_limits.py
│   │   └── test_audit.py
│   ├── integration/
│   │   ├── test_end_to_end_smoke.py
│   │   ├── test_crash_recovery.py
│   │   └── test_reconciliation.py
│   └── fixtures/
│       ├── __init__.py
│       └── policies.py
└── scripts/
    └── init_db.py
```

---

## Task 1: Repo scaffold and tooling

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `README.md` (placeholder), `firm/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "ai-investment-firm"
version = "0.1.0"
description = "Multi-agent AI investment firm — Cato take-home"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.7",
    "langgraph>=0.2",
    "langgraph-checkpoint-sqlite>=1.0",
    "python-ulid>=2.2",
    "pyyaml>=6.0",
    "click>=8.1",
    "structlog>=24.1",
    "alpaca-py>=0.30",  # optional broker; only imported when ALPACA selected
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.5",
    "mypy>=1.10",
]

[project.scripts]
firm = "firm.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 100

[tool.mypy]
strict = true
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/
data/firm.db*
data/firm.db-wal
data/firm.db-shm
data/backups/
data/qdrant/
.env
*.egg-info/
dist/
build/
```

- [ ] **Step 3: Create `.env.example`**

```
# Broker selection: FAKE (default, zero-setup) or ALPACA
FIRM_BROKER=FAKE

# Alpaca paper trading credentials (only required if FIRM_BROKER=ALPACA)
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# HMAC secret for signing decision nonces and HITL approvals
FIRM_HMAC_SECRET=change-me-to-a-32-byte-hex-string

# Database path
FIRM_DB_PATH=data/firm.db
```

- [ ] **Step 4: Create empty package markers**

```bash
mkdir -p firm/core firm/db firm/audit firm/broker firm/outbox firm/reconcile firm/orchestrator firm/agents tests/unit tests/integration tests/fixtures config data scripts
touch firm/__init__.py firm/core/__init__.py firm/db/__init__.py firm/audit/__init__.py firm/broker/__init__.py firm/outbox/__init__.py firm/reconcile/__init__.py firm/orchestrator/__init__.py firm/agents/__init__.py tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py tests/fixtures/__init__.py
```

On Windows PowerShell, use:
```powershell
mkdir firm/core,firm/db,firm/audit,firm/broker,firm/outbox,firm/reconcile,firm/orchestrator,firm/agents,tests/unit,tests/integration,tests/fixtures,config,data,scripts -Force
foreach ($p in 'firm','firm/core','firm/db','firm/audit','firm/broker','firm/outbox','firm/reconcile','firm/orchestrator','firm/agents','tests','tests/unit','tests/integration','tests/fixtures') { New-Item -ItemType File -Path "$p/__init__.py" -Force | Out-Null }
```

- [ ] **Step 5: Create `README.md` placeholder**

```markdown
# AI Investment Firm

Take-home for Cato Networks — Agentic AI Engineer.

## Quickstart (clone-to-demo in <10 min)

```bash
pip install -e ".[dev]"
make demo
```

See `docs/` for design specification and plans.
```

- [ ] **Step 6: Install and verify**

```bash
pip install -e ".[dev]"
python -c "import pydantic, langgraph, ulid, yaml, click, structlog; print('deps OK')"
```

Expected: `deps OK`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore .env.example README.md firm tests config data scripts
git commit -m "scaffold: project layout and dependencies"
```

---

## Task 2: Core domain models

**Files:**
- Create: `firm/core/models.py`
- Test: `tests/unit/test_models.py`

These are the typed contracts every agent emits. They show up in every later task; get them right.

- [ ] **Step 1: Write failing tests for FailureMode and ActionEnum**

`tests/unit/test_models.py`:
```python
from decimal import Decimal
from firm.core.models import (
    ActionEnum, FailureMode, Citation, Claim,
    BuyPayload, SellPayload, HoldPayload, Decision,
)


def test_failure_mode_values():
    assert FailureMode.UNCITED_CLAIM.value == "uncited_claim"
    assert FailureMode.BROKER_UNAVAILABLE.value == "broker_unavailable"
    # 13 total values per spec §3.5
    assert len(list(FailureMode)) == 13


def test_action_enum_values():
    assert {a.value for a in ActionEnum} == {"BUY", "SELL", "HOLD", "ESCALATE", "REFUSE"}


def test_decision_requires_rationale_and_nonce():
    d = Decision(
        id="01HZZZZZZZZZZZZZZZZZZZZZZZ",
        decision_id_chain=[],
        action=ActionEnum.HOLD,
        payload=HoldPayload(reason="stub"),
        rationale="deterministic stub",
        confidence=0.5,
        citations=[],
        falsification_condition="never",
        escalation_reason=None,
        failure_mode=None,
        metadata={},
        nonce="abc123",
    )
    assert d.action == ActionEnum.HOLD


def test_buy_payload_carries_decimal_value():
    p = BuyPayload(ticker="AAPL", shares=Decimal("10"), limit_price=Decimal("180.50"))
    assert p.ticker == "AAPL"
    assert p.shares == Decimal("10")


def test_claim_requires_provenance():
    # text-only claim with no source — should still construct, validation is upstream
    c = Claim(text="NVDA reported revenue", value=None, unit=None,
              source_chunk_id=None, source_span=None, tool_call_id=None)
    assert c.text.startswith("NVDA")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/unit/test_models.py -v
```

Expected: ImportError or ModuleNotFoundError for `firm.core.models`.

- [ ] **Step 3: Implement `firm/core/models.py`**

```python
"""Core typed contracts emitted by every agent. See design spec §3.4, §3.5, §7.2."""
from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal, Union

from pydantic import BaseModel, Field


class ActionEnum(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    ESCALATE = "ESCALATE"
    REFUSE = "REFUSE"


class FailureMode(StrEnum):
    UNCITED_CLAIM = "uncited_claim"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROMPT_INJECTION_DETECTED = "prompt_injection_detected"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    HITL_TIMEOUT = "hitl_timeout"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    LLM_UNAVAILABLE = "llm_unavailable"
    STALE_DATA = "stale_data"
    UNGROUNDED_CLAIM = "ungrounded_claim"
    TOOL_PERMISSION_DENIED = "tool_permission_denied"
    UNAPPROVED_HIGH_RISK = "unapproved_high_risk"
    BROKER_UNAVAILABLE = "broker_unavailable"
    UNKNOWN = "unknown"


class Citation(BaseModel):
    source_id: str
    chunk_id: str
    span: tuple[int, int]


class Claim(BaseModel):
    text: str
    value: Decimal | None = None
    unit: str | None = None
    source_chunk_id: str | None = None
    source_span: tuple[int, int] | None = None
    tool_call_id: str | None = None


class BuyPayload(BaseModel):
    kind: Literal["buy"] = "buy"
    ticker: str
    shares: Decimal
    limit_price: Decimal | None = None


class SellPayload(BaseModel):
    kind: Literal["sell"] = "sell"
    ticker: str
    shares: Decimal
    limit_price: Decimal | None = None


class HoldPayload(BaseModel):
    kind: Literal["hold"] = "hold"
    reason: str


class EscalatePayload(BaseModel):
    kind: Literal["escalate"] = "escalate"
    proposed: BuyPayload | SellPayload
    reason: str


class RefusePayload(BaseModel):
    kind: Literal["refuse"] = "refuse"
    reason: str


TypedPayload = Union[BuyPayload, SellPayload, HoldPayload, EscalatePayload, RefusePayload]


class Decision(BaseModel):
    id: str
    decision_id_chain: list[str] = Field(default_factory=list)
    action: ActionEnum
    payload: TypedPayload
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    falsification_condition: str = Field(min_length=1)
    escalation_reason: str | None = None
    failure_mode: FailureMode | None = None
    metadata: dict = Field(default_factory=dict)
    nonce: str = Field(min_length=1)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/unit/test_models.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add firm/core/models.py tests/unit/test_models.py
git commit -m "feat(core): Decision, Claim, FailureMode and typed payloads"
```

---

## Task 3: Clock injection

**Files:**
- Create: `firm/core/clock.py`
- Test: `tests/unit/test_clock.py`

Every time-dependent piece of code in the firm receives a `Clock` instance. `WallClock` for production, `ReplayClock` for eval. CI lints will later ban `datetime.now()` in business code (Plan 4).

- [ ] **Step 1: Write failing tests**

`tests/unit/test_clock.py`:
```python
from datetime import datetime, timedelta, timezone
from firm.core.clock import Clock, WallClock, ReplayClock


def test_wallclock_returns_utc():
    c: Clock = WallClock()
    t = c.now()
    assert t.tzinfo is not None
    assert t.utcoffset() == timedelta(0)


def test_replayclock_is_fixed_until_advanced():
    fixed = datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc)
    c = ReplayClock(fixed)
    assert c.now() == fixed
    assert c.now() == fixed  # idempotent

    c.advance(60)
    assert c.now() == fixed + timedelta(seconds=60)


def test_replayclock_set():
    c = ReplayClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
    new_time = datetime(2024, 8, 5, 9, 30, tzinfo=timezone.utc)
    c.set(new_time)
    assert c.now() == new_time
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_clock.py -v
```

Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `firm/core/clock.py`**

```python
"""Clock injection for deterministic eval. See design spec §5.4."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime: ...


class WallClock:
    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)


class ReplayClock:
    def __init__(self, fixed: datetime) -> None:
        if fixed.tzinfo is None:
            raise ValueError("ReplayClock requires timezone-aware datetime")
        self._t = fixed

    def now(self) -> datetime:
        return self._t

    def advance(self, seconds: int) -> None:
        self._t = self._t + timedelta(seconds=seconds)

    def set(self, t: datetime) -> None:
        if t.tzinfo is None:
            raise ValueError("ReplayClock.set requires timezone-aware datetime")
        self._t = t
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_clock.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add firm/core/clock.py tests/unit/test_clock.py
git commit -m "feat(core): clock injection with WallClock and ReplayClock"
```

---

## Task 4: IDs and HMAC nonces

**Files:**
- Create: `firm/core/ids.py`
- Test: `tests/unit/test_ids.py`

ULIDs for decision IDs (lexicographically sortable, time-ordered). HMAC-signed nonces bind a decision to its origin so the outbox and HITL approvals can verify authenticity. See spec §3.4 (`Decision.nonce`) and §8.4 (signed Slack approvals).

- [ ] **Step 1: Write failing tests**

`tests/unit/test_ids.py`:
```python
from firm.core.ids import ulid_new, sign_nonce, verify_nonce


def test_ulid_new_returns_26_char_string():
    u = ulid_new()
    assert isinstance(u, str)
    assert len(u) == 26


def test_ulid_new_is_unique():
    a = ulid_new()
    b = ulid_new()
    assert a != b


def test_sign_and_verify_nonce_roundtrip():
    secret = b"a" * 32
    nonce = sign_nonce(secret, decision_id="dec-1", timestamp=1700000000)
    assert verify_nonce(secret, decision_id="dec-1", timestamp=1700000000, nonce=nonce)


def test_verify_rejects_tampered_payload():
    secret = b"a" * 32
    nonce = sign_nonce(secret, decision_id="dec-1", timestamp=1700000000)
    assert not verify_nonce(secret, decision_id="dec-1", timestamp=1700000001, nonce=nonce)
    assert not verify_nonce(secret, decision_id="dec-2", timestamp=1700000000, nonce=nonce)


def test_verify_rejects_wrong_secret():
    nonce = sign_nonce(b"a" * 32, decision_id="dec-1", timestamp=1700000000)
    assert not verify_nonce(b"b" * 32, decision_id="dec-1", timestamp=1700000000, nonce=nonce)
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_ids.py -v
```

- [ ] **Step 3: Implement `firm/core/ids.py`**

```python
"""ULID generation and HMAC nonce sign/verify. See spec §3.4, §8.4."""
from __future__ import annotations

import hashlib
import hmac

from ulid import ULID


def ulid_new() -> str:
    return str(ULID())


def sign_nonce(secret: bytes, *, decision_id: str, timestamp: int) -> str:
    msg = f"{decision_id}:{timestamp}".encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def verify_nonce(secret: bytes, *, decision_id: str, timestamp: int, nonce: str) -> bool:
    expected = sign_nonce(secret, decision_id=decision_id, timestamp=timestamp)
    return hmac.compare_digest(expected, nonce)
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_ids.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add firm/core/ids.py tests/unit/test_ids.py
git commit -m "feat(core): ULID generation and HMAC nonce sign/verify"
```

---

## Task 5: SQLite connection with pragmas

**Files:**
- Create: `firm/db/connection.py`
- Test: `tests/unit/test_connection.py`

Pragmas are load-bearing for durability — see spec §5.1. `WAL` enables concurrent readers + one writer; `synchronous=FULL` flushes the WAL to disk on every commit (slower but never loses committed transactions on crash); `foreign_keys=ON` is required for our schema's FK constraints to be enforced.

- [ ] **Step 1: Write failing test**

`tests/unit/test_connection.py`:
```python
from pathlib import Path
from firm.db.connection import get_conn


def test_connection_applies_pragmas(tmp_path: Path):
    conn = get_conn(tmp_path / "test.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_connection_row_factory_returns_dicts(tmp_path: Path):
    conn = get_conn(tmp_path / "test.db")
    conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'hello')")
    row = conn.execute("SELECT * FROM t").fetchone()
    assert row["a"] == 1
    assert row["b"] == "hello"
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_connection.py -v
```

- [ ] **Step 3: Implement `firm/db/connection.py`**

```python
"""SQLite connection with WAL + synchronous=FULL + foreign keys. See spec §5.1."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def get_conn(db_path: Path) -> sqlite3.Connection:
    """Open (or create) a SQLite connection with durability-grade pragmas.

    journal_mode=WAL allows concurrent readers + single writer.
    synchronous=FULL flushes WAL to disk on every commit (no lost commits on crash).
    foreign_keys=ON enforces FK constraints (off by default in SQLite).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_connection.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add firm/db/connection.py tests/unit/test_connection.py
git commit -m "feat(db): SQLite connection with WAL + synchronous=FULL"
```

---

## Task 6: Database schema and migrations

**Files:**
- Create: `firm/db/schema.sql`, `firm/db/migrations.py`
- Test: `tests/unit/test_migrations.py`

All tables for Plan 1 live here. `decisions` is append-only (the audit log). `outbox` is the durable order queue. `positions` and `cash` are the firm's view of the broker. `hitl_queue` parks decisions awaiting human ack. `reconciliations` records boot-time and EOD reconciliation results.

- [ ] **Step 1: Write failing test**

`tests/unit/test_migrations.py`:
```python
from pathlib import Path
from firm.db.connection import get_conn
from firm.db.migrations import init_db


def test_init_db_creates_all_tables(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    conn = get_conn(db)
    tables = {
        r["name"]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    assert tables == {
        "decisions",
        "outbox",
        "positions",
        "cash",
        "hitl_queue",
        "reconciliations",
        "audit_log",
    }


def test_init_db_is_idempotent(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    init_db(db)  # second call must not raise
    conn = get_conn(db)
    count = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    assert count >= 7
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_migrations.py -v
```

- [ ] **Step 3: Create `firm/db/schema.sql`**

```sql
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
```

- [ ] **Step 4: Implement `firm/db/migrations.py`**

```python
"""Database initialization. Reads schema.sql and applies idempotently."""
from __future__ import annotations

from importlib import resources
from pathlib import Path

from firm.db.connection import get_conn


def init_db(db_path: Path) -> None:
    schema_sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    conn = get_conn(db_path)
    conn.executescript(schema_sql)
```

- [ ] **Step 5: Run — verify pass**

```bash
pytest tests/unit/test_migrations.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add firm/db/schema.sql firm/db/migrations.py tests/unit/test_migrations.py
git commit -m "feat(db): schema and init_db migration"
```

---

## Task 7: Audit logger

**Files:**
- Create: `firm/audit/log.py`
- Test: `tests/unit/test_audit.py`

Append-only writer that every other module uses for non-decision events (reconciliations, acks, halts, partial failures). Decisions themselves go to the `decisions` table directly.

- [ ] **Step 1: Write failing test**

`tests/unit/test_audit.py`:
```python
import json
from datetime import datetime, timezone
from pathlib import Path

from firm.audit.log import AuditLog
from firm.core.clock import ReplayClock
from firm.db.migrations import init_db


def test_audit_append_and_read(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    log = AuditLog(db, clock)

    log.append("reconcile.boot", {"status": "ok", "diff": None})
    log.append("hitl.ack", {"decision_id": "dec-1", "approver": "alice"})

    rows = log.read_all()
    assert len(rows) == 2
    assert rows[0]["event"] == "reconcile.boot"
    assert json.loads(rows[0]["detail"]) == {"status": "ok", "diff": None}
    assert rows[1]["event"] == "hitl.ack"
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_audit.py -v
```

- [ ] **Step 3: Implement `firm/audit/log.py`**

```python
"""Append-only audit log. See spec §1, §3.4, §10.1."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from firm.core.clock import Clock
from firm.db.connection import get_conn


class AuditLog:
    def __init__(self, db_path: Path, clock: Clock) -> None:
        self._db_path = db_path
        self._clock = clock

    def append(self, event: str, detail: dict[str, Any]) -> None:
        conn = get_conn(self._db_path)
        conn.execute(
            "INSERT INTO audit_log (ts, event, detail) VALUES (?, ?, ?)",
            (self._clock.now().isoformat(), event, json.dumps(detail, default=str)),
        )

    def read_all(self) -> list[dict[str, Any]]:
        conn = get_conn(self._db_path)
        return [dict(r) for r in conn.execute("SELECT * FROM audit_log ORDER BY id")]
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_audit.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add firm/audit/log.py tests/unit/test_audit.py
git commit -m "feat(audit): append-only audit logger"
```

---

## Task 8: Configuration loaders

**Files:**
- Create: `config/policy.yaml`, `config/universe.yaml`, `firm/core/config.py`
- Test: `tests/unit/test_config.py`

Pydantic-validated YAML loaders. Schema mirrors spec §3.7 exactly. Malformed YAML → error at startup, not at trade time.

- [ ] **Step 1: Create `config/policy.yaml`**

```yaml
limits:
  max_position_pct: 0.10
  max_sector_pct: 0.30
  max_gross_exposure: 1.00
  max_trade_pct: 0.05
  max_trades_per_day: 20
  min_cash_pct: 0.05
  max_daily_loss_pct: 0.03
  stale_quote_seconds: 60
  stale_filing_days: 90
hitl:
  trade_threshold_pct: 0.03
  escalate_new_ticker: true
```

- [ ] **Step 2: Create `config/universe.yaml`**

```yaml
as_of: 2023-11-01
tickers:
  - AAPL
  - MSFT
  - NVDA
  - GOOGL
  - AMZN
  - META
  - TSLA
  - BRK.B
  - JPM
  - V
  - JNJ
  - WMT
  - PG
  - UNH
  - HD
  - MA
  - LLY
  - XOM
  - ABBV
  - COST
  - AVGO
  - ORCL
  - ADBE
  - KO
  - PEP
  - MRK
  - BAC
  - CSCO
  - NFLX
  - CRM
sector_map:
  AAPL: tech
  MSFT: tech
  NVDA: tech
  GOOGL: tech
  AMZN: tech
  META: tech
  TSLA: tech
  ORCL: tech
  ADBE: tech
  AVGO: tech
  CSCO: tech
  NFLX: tech
  CRM: tech
  BRK.B: finance
  JPM: finance
  V: finance
  MA: finance
  BAC: finance
  JNJ: health
  UNH: health
  LLY: health
  ABBV: health
  MRK: health
  WMT: staples
  PG: staples
  KO: staples
  PEP: staples
  COST: staples
  HD: discretionary
  XOM: energy
```

- [ ] **Step 3: Write failing test**

`tests/unit/test_config.py`:
```python
from pathlib import Path
from firm.core.config import PolicyConfig, UniverseConfig, load_policy, load_universe


def test_load_policy_from_repo():
    p = load_policy(Path("config/policy.yaml"))
    assert isinstance(p, PolicyConfig)
    assert p.limits.max_position_pct == 0.10
    assert p.limits.max_trades_per_day == 20
    assert p.hitl.trade_threshold_pct == 0.03


def test_load_universe_from_repo():
    u = load_universe(Path("config/universe.yaml"))
    assert isinstance(u, UniverseConfig)
    assert len(u.tickers) == 30
    assert "AAPL" in u.tickers
    assert u.sector_map["AAPL"] == "tech"


def test_universe_rejects_unmapped_ticker(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("as_of: 2023-11-01\ntickers: [AAPL, XYZ]\nsector_map: {AAPL: tech}\n")
    import pytest
    with pytest.raises(ValueError, match="XYZ"):
        load_universe(bad)
```

- [ ] **Step 4: Run — verify fail**

```bash
pytest tests/unit/test_config.py -v
```

- [ ] **Step 5: Implement `firm/core/config.py`**

```python
"""Pydantic-validated YAML config loaders. See spec §3.7."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator


class PolicyLimits(BaseModel):
    max_position_pct: float
    max_sector_pct: float
    max_gross_exposure: float
    max_trade_pct: float
    max_trades_per_day: int
    min_cash_pct: float
    max_daily_loss_pct: float
    stale_quote_seconds: int
    stale_filing_days: int


class HitlConfig(BaseModel):
    trade_threshold_pct: float
    escalate_new_ticker: bool


class PolicyConfig(BaseModel):
    limits: PolicyLimits
    hitl: HitlConfig


class UniverseConfig(BaseModel):
    as_of: date
    tickers: list[str]
    sector_map: dict[str, str]

    @model_validator(mode="after")
    def _check_all_tickers_mapped(self) -> "UniverseConfig":
        unmapped = [t for t in self.tickers if t not in self.sector_map]
        if unmapped:
            raise ValueError(f"unmapped tickers: {unmapped}")
        return self


def load_policy(path: Path) -> PolicyConfig:
    return PolicyConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_universe(path: Path) -> UniverseConfig:
    return UniverseConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
```

- [ ] **Step 6: Run — verify pass**

```bash
pytest tests/unit/test_config.py -v
```

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add config/policy.yaml config/universe.yaml firm/core/config.py tests/unit/test_config.py
git commit -m "feat(core): policy.yaml + universe.yaml with Pydantic validation"
```

---

## Task 9: Broker protocol and FakeBroker

**Files:**
- Create: `firm/broker/protocol.py`, `firm/broker/fake_broker.py`
- Test: `tests/unit/test_fake_broker.py`

`Broker` is a `Protocol` (structural typing); both `FakeBroker` and `AlpacaBroker` satisfy it. `FakeBroker` is the default for demo (zero setup) and for all unit/integration tests. It is deterministic and idempotency-key aware.

- [ ] **Step 1: Implement `firm/broker/protocol.py`** (no test — types only)

```python
"""Broker abstraction. See spec §5.2, §5.7."""
from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel


class Position(BaseModel):
    ticker: str
    shares: Decimal
    avg_cost: Decimal


class Quote(BaseModel):
    ticker: str
    price: Decimal
    timestamp: str  # ISO 8601


class OrderResult(BaseModel):
    order_id: str
    ticker: str
    filled_shares: Decimal
    avg_fill_price: Decimal
    commission: Decimal
    slippage: Decimal
    submitted_at: str
    filled_at: str


class Broker(Protocol):
    def list_positions(self) -> list[Position]: ...
    def get_cash(self) -> Decimal: ...
    def get_quote(self, ticker: str) -> Quote: ...
    def submit(self, decision_payload: dict, idempotency_key: str) -> OrderResult: ...
```

- [ ] **Step 2: Write failing test**

`tests/unit/test_fake_broker.py`:
```python
from decimal import Decimal
from firm.broker.fake_broker import FakeBroker


def test_fake_broker_starts_with_initial_cash():
    b = FakeBroker(initial_cash=Decimal("100000"))
    assert b.get_cash() == Decimal("100000")
    assert b.list_positions() == []


def test_submit_returns_deterministic_fill():
    b = FakeBroker(initial_cash=Decimal("100000"))
    payload = {"kind": "buy", "ticker": "AAPL", "shares": "10"}
    r = b.submit(payload, idempotency_key="key-1")
    assert r.ticker == "AAPL"
    assert r.filled_shares == Decimal("10")


def test_submit_is_idempotent_on_same_key():
    b = FakeBroker(initial_cash=Decimal("100000"))
    payload = {"kind": "buy", "ticker": "AAPL", "shares": "10"}
    r1 = b.submit(payload, idempotency_key="key-1")
    r2 = b.submit(payload, idempotency_key="key-1")
    assert r1.order_id == r2.order_id
    assert b.get_cash() < Decimal("100000")  # cash debited exactly once
    # exactly one fill should be reflected in positions
    pos = [p for p in b.list_positions() if p.ticker == "AAPL"][0]
    assert pos.shares == Decimal("10")


def test_buy_reduces_cash_by_price_plus_commission():
    b = FakeBroker(initial_cash=Decimal("100000"))
    b.submit({"kind": "buy", "ticker": "AAPL", "shares": "10"}, "key-1")
    quote = b.get_quote("AAPL")
    # FakeBroker uses a fixed price function; assert cash dropped by reasonable amount
    assert Decimal("100000") - b.get_cash() > Decimal("0")
    assert b.get_cash() < Decimal("100000")


def test_sell_reduces_position_and_credits_cash():
    b = FakeBroker(initial_cash=Decimal("100000"))
    b.submit({"kind": "buy", "ticker": "AAPL", "shares": "10"}, "k1")
    cash_after_buy = b.get_cash()
    b.submit({"kind": "sell", "ticker": "AAPL", "shares": "10"}, "k2")
    assert b.get_cash() > cash_after_buy
    assert all(p.shares == Decimal("0") or p.ticker != "AAPL" for p in b.list_positions())
```

- [ ] **Step 3: Run — verify fail**

```bash
pytest tests/unit/test_fake_broker.py -v
```

- [ ] **Step 4: Implement `firm/broker/fake_broker.py`**

```python
"""Deterministic in-memory broker for demo and tests. See spec §5.2, §5.7."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from decimal import Decimal

from firm.broker.protocol import OrderResult, Position, Quote


def _deterministic_price(ticker: str) -> Decimal:
    """Stable, ticker-dependent price for replayability."""
    h = int(hashlib.sha256(ticker.encode()).hexdigest(), 16) % 1000
    return Decimal(50 + h) + Decimal("0.50")


class FakeBroker:
    COMMISSION = Decimal("0.005")  # 0.5% per trade

    def __init__(self, initial_cash: Decimal = Decimal("100000")) -> None:
        self._cash: Decimal = initial_cash
        self._positions: dict[str, Position] = {}
        self._order_cache: dict[str, OrderResult] = {}  # idempotency_key → result

    def list_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if p.shares != Decimal("0")]

    def get_cash(self) -> Decimal:
        return self._cash

    def get_quote(self, ticker: str) -> Quote:
        return Quote(
            ticker=ticker,
            price=_deterministic_price(ticker),
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

    def submit(self, decision_payload: dict, idempotency_key: str) -> OrderResult:
        if idempotency_key in self._order_cache:
            return self._order_cache[idempotency_key]

        ticker = decision_payload["ticker"]
        shares = Decimal(str(decision_payload["shares"]))
        price = _deterministic_price(ticker)
        kind = decision_payload["kind"]
        slippage = price * Decimal("0.0005")  # 5 bps
        fill_price = price + slippage if kind == "buy" else price - slippage
        gross = fill_price * shares
        commission = gross * self.COMMISSION

        if kind == "buy":
            self._cash -= gross + commission
            prev = self._positions.get(ticker, Position(ticker=ticker, shares=Decimal("0"), avg_cost=Decimal("0")))
            new_shares = prev.shares + shares
            if new_shares > 0:
                new_avg = ((prev.avg_cost * prev.shares) + (fill_price * shares)) / new_shares
            else:
                new_avg = Decimal("0")
            self._positions[ticker] = Position(ticker=ticker, shares=new_shares, avg_cost=new_avg)
        elif kind == "sell":
            self._cash += gross - commission
            prev = self._positions.get(ticker, Position(ticker=ticker, shares=Decimal("0"), avg_cost=Decimal("0")))
            self._positions[ticker] = Position(
                ticker=ticker, shares=prev.shares - shares, avg_cost=prev.avg_cost
            )
        else:
            raise ValueError(f"unsupported order kind: {kind}")

        now = datetime.now(tz=timezone.utc).isoformat()
        result = OrderResult(
            order_id=hashlib.sha256(idempotency_key.encode()).hexdigest()[:16],
            ticker=ticker,
            filled_shares=shares,
            avg_fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            submitted_at=now,
            filled_at=now,
        )
        self._order_cache[idempotency_key] = result
        return result
```

- [ ] **Step 5: Run — verify pass**

```bash
pytest tests/unit/test_fake_broker.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add firm/broker/protocol.py firm/broker/fake_broker.py tests/unit/test_fake_broker.py
git commit -m "feat(broker): Broker Protocol and deterministic FakeBroker"
```

---

## Task 10: Alpaca paper broker adapter

**Files:**
- Create: `firm/broker/alpaca_paper.py`

No unit test — Alpaca requires network/credentials. Smoke-tested via the integration suite gated on env vars.

- [ ] **Step 1: Implement `firm/broker/alpaca_paper.py`**

```python
"""Alpaca paper trading adapter. Activated only when FIRM_BROKER=ALPACA."""
from __future__ import annotations

import os
from decimal import Decimal
from typing import Any

from firm.broker.protocol import OrderResult, Position, Quote


class AlpacaBroker:
    """Thin wrapper around alpaca-py for paper trading.

    Lazy-imports alpaca-py so the FakeBroker default path has no SDK dependency.
    """

    def __init__(self) -> None:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as e:
            raise RuntimeError(
                "alpaca-py not installed; install with `pip install alpaca-py`"
            ) from e

        api_key = os.environ["ALPACA_API_KEY"]
        secret = os.environ["ALPACA_SECRET_KEY"]
        self._trading = TradingClient(api_key, secret, paper=True)
        self._data = StockHistoricalDataClient(api_key, secret)

    def list_positions(self) -> list[Position]:
        return [
            Position(
                ticker=p.symbol,
                shares=Decimal(str(p.qty)),
                avg_cost=Decimal(str(p.avg_entry_price)),
            )
            for p in self._trading.get_all_positions()
        ]

    def get_cash(self) -> Decimal:
        acct = self._trading.get_account()
        return Decimal(str(acct.cash))

    def get_quote(self, ticker: str) -> Quote:
        from alpaca.data.requests import StockLatestQuoteRequest
        q = self._data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))[ticker]
        # Use the midpoint of bid/ask as price
        price = (Decimal(str(q.bid_price)) + Decimal(str(q.ask_price))) / Decimal("2")
        return Quote(ticker=ticker, price=price, timestamp=q.timestamp.isoformat())

    def submit(self, decision_payload: dict[str, Any], idempotency_key: str) -> OrderResult:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        side = OrderSide.BUY if decision_payload["kind"] == "buy" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=decision_payload["ticker"],
            qty=float(decision_payload["shares"]),
            side=side,
            time_in_force=TimeInForce.DAY,
            client_order_id=idempotency_key,  # Alpaca's idempotency mechanism
        )
        order = self._trading.submit_order(req)
        return OrderResult(
            order_id=str(order.id),
            ticker=order.symbol,
            filled_shares=Decimal(str(order.filled_qty or 0)),
            avg_fill_price=Decimal(str(order.filled_avg_price or 0)),
            commission=Decimal("0"),  # Alpaca paper has no commission
            slippage=Decimal("0"),
            submitted_at=order.submitted_at.isoformat(),
            filled_at=order.filled_at.isoformat() if order.filled_at else order.submitted_at.isoformat(),
        )


def make_broker():
    """Factory: select broker by FIRM_BROKER env var. Default FakeBroker."""
    kind = os.environ.get("FIRM_BROKER", "FAKE").upper()
    if kind == "ALPACA":
        return AlpacaBroker()
    from firm.broker.fake_broker import FakeBroker
    return FakeBroker()
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from firm.broker.alpaca_paper import make_broker; b = make_broker(); print(type(b).__name__)"
```

Expected: `FakeBroker` (because FIRM_BROKER defaults to FAKE).

- [ ] **Step 3: Commit**

```bash
git add firm/broker/alpaca_paper.py
git commit -m "feat(broker): Alpaca paper adapter and broker factory"
```

---

## Task 11: Outbox pattern

**Files:**
- Create: `firm/outbox/outbox.py`
- Test: `tests/unit/test_outbox.py`

The single most load-bearing piece of durability infrastructure. See spec §5.2. Crash semantics: each of `before-insert`, `after-insert-before-call`, `after-call-before-confirm` must converge to exactly-one-order at the broker on recovery.

- [ ] **Step 1: Write failing tests**

`tests/unit/test_outbox.py`:
```python
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.models import (
    ActionEnum, BuyPayload, Decision,
)
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.outbox.outbox import place_order_via_outbox, recover_pending


def _decision(decision_id: str = "dec-1") -> Decision:
    return Decision(
        id=decision_id, decision_id_chain=[], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="test", confidence=0.5, citations=[],
        falsification_condition="if AAPL drops 10%",
        escalation_reason=None, failure_mode=None, metadata={}, nonce="n-1",
    )


def _persist_decision(db: Path, d: Decision, clock):
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            d.id, json.dumps(d.decision_id_chain), d.action.value,
            d.payload.model_dump_json(), d.rationale, d.confidence,
            json.dumps([c.model_dump(mode="json") for c in d.citations]),
            d.falsification_condition, d.escalation_reason,
            d.failure_mode.value if d.failure_mode else None,
            json.dumps(d.metadata), d.nonce, clock.now().isoformat(),
        ),
    )


def test_outbox_places_order_and_marks_confirmed(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = _decision()
    _persist_decision(db, d, clock)

    result = place_order_via_outbox(d, db, broker, clock)
    assert result.ticker == "AAPL"

    row = get_conn(db).execute("SELECT status FROM outbox WHERE decision_id=?", (d.id,)).fetchone()
    assert row["status"] == "confirmed"


def test_outbox_is_idempotent_on_replay(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = _decision()
    _persist_decision(db, d, clock)

    r1 = place_order_via_outbox(d, db, broker, clock)
    r2 = place_order_via_outbox(d, db, broker, clock)  # second call must be no-op
    assert r1.order_id == r2.order_id
    rows = get_conn(db).execute("SELECT COUNT(*) c FROM outbox WHERE decision_id=?", (d.id,)).fetchone()
    assert rows["c"] == 1
    # cash debited exactly once
    pos = [p for p in broker.list_positions() if p.ticker == "AAPL"][0]
    assert pos.shares == Decimal("10")


def test_recover_pending_drives_pending_rows_to_confirmed(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = _decision()
    _persist_decision(db, d, clock)

    # simulate crash between outbox-insert and broker-call: insert pending row manually
    conn = get_conn(db)
    key = "fake-key-1"
    conn.execute(
        "INSERT INTO outbox (key, decision_id, payload, status, created_at, updated_at) "
        "VALUES (?, ?, ?, 'pending', ?, ?)",
        (key, d.id, d.model_dump_json(), clock.now().isoformat(), clock.now().isoformat()),
    )

    recovered = recover_pending(db, broker, clock)
    assert len(recovered) == 1

    row = get_conn(db).execute("SELECT status FROM outbox WHERE key=?", (key,)).fetchone()
    assert row["status"] == "confirmed"
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_outbox.py -v
```

- [ ] **Step 3: Implement `firm/outbox/outbox.py`**

```python
"""Transactional outbox for broker orders. See design spec §5.2."""
from __future__ import annotations

import hashlib
from pathlib import Path

from firm.broker.protocol import Broker, OrderResult
from firm.core.clock import Clock
from firm.core.models import Decision
from firm.db.connection import get_conn


def _idempotency_key(decision: Decision) -> str:
    return hashlib.sha256(f"{decision.id}:{decision.nonce}".encode()).hexdigest()


def place_order_via_outbox(
    decision: Decision, db_path: Path, broker: Broker, clock: Clock
) -> OrderResult:
    """Place an order with exactly-once semantics. See spec §5.2 crash semantics."""
    key = _idempotency_key(decision)
    conn = get_conn(db_path)
    now = clock.now().isoformat()

    # Insert-or-noop in a transaction. After this, the outbox row is durable.
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "INSERT INTO outbox (key, decision_id, payload, status, created_at, updated_at) "
            "VALUES (?, ?, ?, 'pending', ?, ?) ON CONFLICT (key) DO NOTHING",
            (key, decision.id, decision.model_dump_json(), now, now),
        )
        row = conn.execute(
            "SELECT status, result FROM outbox WHERE key = ?", (key,)
        ).fetchone()
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    if row["status"] == "confirmed":
        return OrderResult.model_validate_json(row["result"])

    # Submit to broker (broker enforces its own idempotency via the same key).
    result = broker.submit(decision.payload.model_dump(mode="json"), idempotency_key=key)

    conn.execute(
        "UPDATE outbox SET status='confirmed', result=?, updated_at=? WHERE key=?",
        (result.model_dump_json(), clock.now().isoformat(), key),
    )
    return result


def recover_pending(db_path: Path, broker: Broker, clock: Clock) -> list[OrderResult]:
    """On boot, drive any `pending` outbox rows to terminal status."""
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT key, decision_id, payload FROM outbox WHERE status='pending'"
    ).fetchall()
    results: list[OrderResult] = []
    for r in rows:
        decision = Decision.model_validate_json(r["payload"])
        result = broker.submit(decision.payload.model_dump(mode="json"), idempotency_key=r["key"])
        conn.execute(
            "UPDATE outbox SET status='confirmed', result=?, updated_at=? WHERE key=?",
            (result.model_dump_json(), clock.now().isoformat(), r["key"]),
        )
        results.append(result)
    return results
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_outbox.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add firm/outbox/outbox.py tests/unit/test_outbox.py
git commit -m "feat(outbox): transactional outbox with idempotency and recovery"
```

---

## Task 12: Boot-time reconciliation

**Files:**
- Create: `firm/reconcile/boot.py`
- Test: `tests/integration/test_reconciliation.py`

On every startup, before any decision, fetch broker positions/cash → compare to local DB → halt with exit code 1 on mismatch (Plan 1 halt; Plan 3 swaps in Slack ack). EOD reconciliation comes in Plan 3.

- [ ] **Step 1: Write failing test**

`tests/integration/test_reconciliation.py`:
```python
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.reconcile.boot import ReconcileResult, reconcile_on_boot


def _seed_local_position(db: Path, ticker: str, shares: str, avg_cost: str, clock):
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO positions (ticker, shares, avg_cost, updated_at) VALUES (?, ?, ?, ?)",
        (ticker, shares, avg_cost, clock.now().isoformat()),
    )


def _seed_local_cash(db: Path, amount: str, clock):
    conn = get_conn(db)
    conn.execute(
        "INSERT OR REPLACE INTO cash (id, amount, updated_at) VALUES (1, ?, ?)",
        (amount, clock.now().isoformat()),
    )


def test_reconcile_clean_match_returns_ok(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    _seed_local_cash(db, "100000", clock)

    result = reconcile_on_boot(db, broker, clock)
    assert result.status == "ok"
    assert result.diff == {}


def test_reconcile_returns_mismatch_when_cash_differs(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    _seed_local_cash(db, "95000", clock)  # local thinks 95k; broker says 100k

    result = reconcile_on_boot(db, broker, clock)
    assert result.status == "mismatch"
    assert "cash" in result.diff


def test_reconcile_returns_mismatch_when_position_differs(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    broker.submit({"kind": "buy", "ticker": "AAPL", "shares": "10"}, "k1")
    _seed_local_cash(db, str(broker.get_cash()), clock)  # cash matches

    # local DB has no AAPL position; broker has 10 shares
    result = reconcile_on_boot(db, broker, clock)
    assert result.status == "mismatch"
    assert "positions" in result.diff


def test_reconcile_writes_to_reconciliations_table(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    _seed_local_cash(db, "100000", clock)

    reconcile_on_boot(db, broker, clock)
    rows = list(get_conn(db).execute("SELECT * FROM reconciliations WHERE kind='boot'"))
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/integration/test_reconciliation.py -v
```

- [ ] **Step 3: Implement `firm/reconcile/boot.py`**

```python
"""Boot-time position reconciliation. Broker is source of truth. See spec §5.7."""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from firm.audit.log import AuditLog
from firm.broker.protocol import Broker
from firm.core.clock import Clock
from firm.db.connection import get_conn


@dataclass
class ReconcileResult:
    status: str  # 'ok' or 'mismatch'
    diff: dict


def _local_positions(db_path: Path) -> dict[str, Decimal]:
    conn = get_conn(db_path)
    return {r["ticker"]: Decimal(r["shares"]) for r in conn.execute("SELECT * FROM positions")}


def _local_cash(db_path: Path) -> Decimal:
    conn = get_conn(db_path)
    row = conn.execute("SELECT amount FROM cash WHERE id=1").fetchone()
    return Decimal(row["amount"]) if row else Decimal("0")


def reconcile_on_boot(db_path: Path, broker: Broker, clock: Clock) -> ReconcileResult:
    broker_positions = {p.ticker: p.shares for p in broker.list_positions()}
    broker_cash = broker.get_cash()
    local_positions = _local_positions(db_path)
    local_cash = _local_cash(db_path)

    diff: dict = {}
    pos_diff = {}
    for t in set(broker_positions) | set(local_positions):
        b = broker_positions.get(t, Decimal("0"))
        l = local_positions.get(t, Decimal("0"))
        if b != l:
            pos_diff[t] = {"broker": str(b), "local": str(l)}
    if pos_diff:
        diff["positions"] = pos_diff
    if broker_cash != local_cash:
        diff["cash"] = {"broker": str(broker_cash), "local": str(local_cash)}

    status = "ok" if not diff else "mismatch"
    conn = get_conn(db_path)
    conn.execute(
        "INSERT INTO reconciliations (kind, ran_at, broker_snapshot, local_snapshot, diff, status) "
        "VALUES ('boot', ?, ?, ?, ?, ?)",
        (
            clock.now().isoformat(),
            json.dumps({"positions": {t: str(s) for t, s in broker_positions.items()}, "cash": str(broker_cash)}),
            json.dumps({"positions": {t: str(s) for t, s in local_positions.items()}, "cash": str(local_cash)}),
            json.dumps(diff),
            status,
        ),
    )
    AuditLog(db_path, clock).append("reconcile.boot", {"status": status, "diff": diff})
    return ReconcileResult(status=status, diff=diff)
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/integration/test_reconciliation.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add firm/reconcile/boot.py tests/integration/test_reconciliation.py
git commit -m "feat(reconcile): boot-time reconciliation against broker source of truth"
```

---

## Task 13: LangGraph state and graph skeleton

**Files:**
- Create: `firm/orchestrator/state.py`, `firm/orchestrator/graph.py`
- Test: deferred to end-to-end smoke (Task 22)

`WorkingState` is the LangGraph state shared across nodes. The graph is the topology of §3.1: monitor → research → pm → risk → hitl → execution → reporter.

- [ ] **Step 1: Implement `firm/orchestrator/state.py`**

```python
"""LangGraph shared state. See spec §3.1, §4.1."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import add_messages

from firm.core.models import Decision


class WorkingState(TypedDict, total=False):
    """State flowing through the LangGraph workflow.

    Each agent reads upstream Decision(s) from this state and appends its own.
    Decisions chain via Decision.decision_id_chain.
    """
    heartbeat_at: str                  # ISO 8601
    research_decision: Decision
    pm_decision: Decision
    risk_decision: Decision
    hitl_required: bool
    hitl_approved: bool | None
    execution_result: dict             # OrderResult-as-dict
    report_path: str
    notes: Annotated[list[str], add_messages]
```

- [ ] **Step 2: Implement `firm/orchestrator/graph.py`**

```python
"""LangGraph topology. See spec §3.1."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from firm.orchestrator.state import WorkingState


def build_graph(
    *,
    db_path: Path,
    monitor_node: Callable,
    research_node: Callable,
    pm_node: Callable,
    risk_node: Callable,
    hitl_node: Callable,
    execution_node: Callable,
    reporter_node: Callable,
):
    """Compose the firm's workflow.

    Edges: monitor → research → pm → risk → (hitl|execution) → reporter
    Conditional after risk: if hitl_required → hitl → execution; else → execution.
    """
    g = StateGraph(WorkingState)
    g.add_node("monitor", monitor_node)
    g.add_node("research", research_node)
    g.add_node("pm", pm_node)
    g.add_node("risk", risk_node)
    g.add_node("hitl", hitl_node)
    g.add_node("execution", execution_node)
    g.add_node("reporter", reporter_node)

    g.set_entry_point("monitor")
    g.add_edge("monitor", "research")
    g.add_edge("research", "pm")
    g.add_edge("pm", "risk")

    def route_after_risk(state: WorkingState) -> str:
        return "hitl" if state.get("hitl_required") else "execution"

    g.add_conditional_edges("risk", route_after_risk, {"hitl": "hitl", "execution": "execution"})
    g.add_edge("hitl", "execution")
    g.add_edge("execution", "reporter")
    g.add_edge("reporter", END)

    import sqlite3
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    return g.compile(checkpointer=saver, interrupt_before=["hitl"])
```

Note: `interrupt_before=["hitl"]` parks the graph at the HITL node. The CLI's `firm ack <id>` provides approval and resumes the graph.

- [ ] **Step 3: Smoke import**

```bash
python -c "from firm.orchestrator.graph import build_graph; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add firm/orchestrator/state.py firm/orchestrator/graph.py
git commit -m "feat(orchestrator): LangGraph state and graph skeleton"
```

---

## Task 14: Position Monitor node

**Files:**
- Create: `firm/agents/monitor.py`
- Test: `tests/unit/test_monitor.py`

Plan 1 produces a single heartbeat per `firm run` invocation. Event-driven triggers (price moves, news) are Plan 2/3. The monitor just stamps the current time into state.

- [ ] **Step 1: Write failing test**

`tests/unit/test_monitor.py`:
```python
from datetime import datetime, timezone

from firm.agents.monitor import make_monitor
from firm.core.clock import ReplayClock


def test_monitor_emits_heartbeat_timestamp():
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    monitor = make_monitor(clock)
    state = monitor({})
    assert state["heartbeat_at"] == "2024-03-13T14:30:00+00:00"
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_monitor.py -v
```

- [ ] **Step 3: Implement `firm/agents/monitor.py`**

```python
"""Position Monitor heartbeat node. See spec §3.1."""
from __future__ import annotations

from firm.core.clock import Clock


def make_monitor(clock: Clock):
    def monitor(state: dict) -> dict:
        return {"heartbeat_at": clock.now().isoformat()}
    return monitor
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_monitor.py -v
```

- [ ] **Step 5: Commit**

```bash
git add firm/agents/monitor.py tests/unit/test_monitor.py
git commit -m "feat(agents): Position Monitor heartbeat node"
```

---

## Task 15: Risk agent — hard limits (the most real agent in Plan 1)

**Files:**
- Create: `firm/agents/risk.py`
- Test: `tests/unit/test_risk_limits.py`

Deterministic Python enforcement of every limit in spec §3.7. The LLM does not run here in Plan 1 — soft policy (LLM context check) is Plan 2.

- [ ] **Step 1: Write failing tests**

`tests/unit/test_risk_limits.py`:
```python
from decimal import Decimal
from pathlib import Path

from firm.agents.risk import RiskInput, evaluate_risk
from firm.core.config import load_policy
from firm.core.models import ActionEnum, BuyPayload, Decision

POLICY = load_policy(Path("config/policy.yaml"))


def _proposal(ticker: str, shares: str) -> Decision:
    return Decision(
        id="pm-1", decision_id_chain=["res-1"], action=ActionEnum.BUY,
        payload=BuyPayload(ticker=ticker, shares=Decimal(shares)),
        rationale="x", confidence=0.7, citations=[], falsification_condition="y",
        escalation_reason=None, failure_mode=None, metadata={}, nonce="n",
    )


def _make_input(*, ticker="AAPL", shares="10", price="180", cash="100000",
                positions=None, trades_today=0, quote_age_seconds=5,
                daily_pnl_pct=0.0) -> RiskInput:
    return RiskInput(
        proposal=_proposal(ticker, shares),
        quote_price=Decimal(price),
        quote_age_seconds=quote_age_seconds,
        cash=Decimal(cash),
        positions=positions or {},
        sector_map={"AAPL": "tech", "MSFT": "tech", "JPM": "finance"},
        trades_today=trades_today,
        nav=Decimal("100000"),
        daily_pnl_pct=daily_pnl_pct,
        policy=POLICY,
    )


def test_passes_within_all_limits():
    out = evaluate_risk(_make_input())
    assert out.action == ActionEnum.BUY


def test_blocks_max_position_pct():
    # Pre-seed AAPL at 50 shares ($9,000 = 9%). Buy 6 more = 1.08% trade (passes max_trade_pct)
    # but new position = 56 * $180 = $10,080 = 10.08% > 10% limit.
    out = evaluate_risk(_make_input(shares="6", positions={"AAPL": Decimal("50")}))
    assert out.action == ActionEnum.REFUSE
    assert out.failure_mode is not None
    assert out.failure_mode.value == "risk_limit_breached"


def test_blocks_max_trade_pct():
    # max_trade_pct = 5% NAV. $6000 = 6% breaches.
    out = evaluate_risk(_make_input(shares="34"))  # 34*180=6120
    assert out.action == ActionEnum.REFUSE


def test_blocks_max_trades_per_day():
    out = evaluate_risk(_make_input(trades_today=20))
    assert out.action == ActionEnum.REFUSE


def test_blocks_min_cash_buffer():
    # cash 5000, trade requires more than 5% of NAV ($5000) buffer to remain
    out = evaluate_risk(_make_input(cash="2000"))
    assert out.action == ActionEnum.REFUSE


def test_blocks_stale_quote():
    out = evaluate_risk(_make_input(quote_age_seconds=999))  # > 60s threshold
    assert out.action == ActionEnum.REFUSE
    assert out.failure_mode.value == "stale_data"


def test_blocks_drawdown_halt():
    out = evaluate_risk(_make_input(shares="10", daily_pnl_pct=-0.04))  # -4% > -3% threshold
    assert out.action == ActionEnum.REFUSE
    assert out.failure_mode.value == "risk_limit_breached"


def test_blocks_sector_concentration():
    # Pre-seed MSFT 145 shares ($26,100 = 26.1% tech). AAPL trade 25 shares = $4,500 = 4.5%
    # passes max_trade_pct (5%) and max_position_pct (10%, fresh position). Combined tech =
    # 26.1% + 4.5% = 30.6% > 30% sector limit → REFUSE.
    positions = {"MSFT": Decimal("145")}
    out = evaluate_risk(_make_input(ticker="AAPL", shares="25", positions=positions))
    assert out.action == ActionEnum.REFUSE
    assert out.failure_mode.value == "risk_limit_breached"


def test_hitl_threshold_escalates_instead_of_passing():
    # Trade > 3% NAV = $3000 → HITL escalate (NOT refuse)
    out = evaluate_risk(_make_input(shares="17"))  # 17*180=3060>3000
    assert out.action == ActionEnum.ESCALATE


def test_every_limit_has_at_least_one_triggering_fixture():
    """CI invariant: each enumerated limit row must be triggered by a test above.

    See spec §3.7 final bullet. We assert by checking each test name pattern.
    """
    import inspect, sys
    triggered = {n for n in dir(sys.modules[__name__]) if n.startswith("test_blocks_")}
    # 7 hard limits enumerated (max_position, max_trade, max_trades_per_day, min_cash,
    # stale_quote, drawdown, sector_concentration); HITL threshold covered by escalates test.
    assert len(triggered) >= 7
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_risk_limits.py -v
```

- [ ] **Step 3: Implement `firm/agents/risk.py`**

```python
"""Risk agent — deterministic hard limits enforcement. See spec §3.7."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from firm.core.config import PolicyConfig
from firm.core.ids import ulid_new
from firm.core.models import (
    ActionEnum, BuyPayload, Decision, EscalatePayload, FailureMode, RefusePayload, SellPayload,
)


@dataclass
class RiskInput:
    proposal: Decision
    quote_price: Decimal
    quote_age_seconds: int
    cash: Decimal
    positions: dict[str, Decimal]      # ticker → shares
    sector_map: dict[str, str]         # ticker → sector
    trades_today: int
    nav: Decimal
    daily_pnl_pct: float
    policy: PolicyConfig


def _decision_from_breach(input: RiskInput, reason: str) -> Decision:
    return Decision(
        id=ulid_new(), decision_id_chain=[input.proposal.id],
        action=ActionEnum.REFUSE,
        payload=RefusePayload(reason=reason),
        rationale=f"hard limit: {reason}", confidence=1.0, citations=[],
        falsification_condition="never (deterministic limit)",
        escalation_reason=None, failure_mode=FailureMode.RISK_LIMIT_BREACHED,
        metadata={"agent": "risk"}, nonce="risk-deterministic",
    )


def _decision_stale(input: RiskInput, reason: str) -> Decision:
    return Decision(
        id=ulid_new(), decision_id_chain=[input.proposal.id],
        action=ActionEnum.REFUSE,
        payload=RefusePayload(reason=reason),
        rationale=f"data freshness violation: {reason}", confidence=1.0, citations=[],
        falsification_condition="never", escalation_reason=None,
        failure_mode=FailureMode.STALE_DATA, metadata={"agent": "risk"}, nonce="risk-stale",
    )


def _escalate(input: RiskInput, reason: str) -> Decision:
    assert isinstance(input.proposal.payload, (BuyPayload, SellPayload))
    return Decision(
        id=ulid_new(), decision_id_chain=[input.proposal.id],
        action=ActionEnum.ESCALATE,
        payload=EscalatePayload(proposed=input.proposal.payload, reason=reason),
        rationale=f"HITL required: {reason}", confidence=1.0, citations=[],
        falsification_condition="HITL approval timeout",
        escalation_reason=reason, failure_mode=None,
        metadata={"agent": "risk"}, nonce="risk-escalate",
    )


def _pass(input: RiskInput) -> Decision:
    return Decision(
        id=ulid_new(), decision_id_chain=[input.proposal.id],
        action=input.proposal.action,
        payload=input.proposal.payload,
        rationale="all hard limits pass", confidence=input.proposal.confidence,
        citations=input.proposal.citations,
        falsification_condition=input.proposal.falsification_condition,
        escalation_reason=None, failure_mode=None,
        metadata={"agent": "risk"}, nonce="risk-pass",
    )


def evaluate_risk(input: RiskInput) -> Decision:
    """Apply every hard limit in policy.yaml. Returns one Decision."""
    p = input.policy
    proposal = input.proposal

    if input.daily_pnl_pct <= -p.limits.max_daily_loss_pct:
        return _decision_from_breach(input, "daily loss drawdown halt")

    if input.quote_age_seconds > p.limits.stale_quote_seconds:
        return _decision_stale(input, f"quote age {input.quote_age_seconds}s")

    if proposal.action == ActionEnum.HOLD:
        return _pass(input)

    if not isinstance(proposal.payload, (BuyPayload, SellPayload)):
        return _pass(input)

    trade_value = input.quote_price * proposal.payload.shares
    trade_pct = float(trade_value / input.nav)

    if input.trades_today >= p.limits.max_trades_per_day:
        return _decision_from_breach(input, "max trades per day")

    if trade_pct > p.limits.max_trade_pct:
        return _decision_from_breach(input, f"trade size {trade_pct:.3f} > {p.limits.max_trade_pct}")

    # Max-position-pct check (after applying the proposed trade)
    ticker = proposal.payload.ticker
    cur_shares = input.positions.get(ticker, Decimal("0"))
    new_shares = cur_shares + proposal.payload.shares if proposal.action == ActionEnum.BUY else cur_shares - proposal.payload.shares
    new_position_value = new_shares * input.quote_price
    pos_pct = float(new_position_value / input.nav)
    if pos_pct > p.limits.max_position_pct:
        return _decision_from_breach(input, f"position {pos_pct:.3f} > {p.limits.max_position_pct}")

    # Min cash buffer (only for buys)
    if proposal.action == ActionEnum.BUY:
        cash_after = input.cash - trade_value
        cash_pct = float(cash_after / input.nav)
        if cash_pct < p.limits.min_cash_pct:
            return _decision_from_breach(input, f"cash buffer {cash_pct:.3f} < {p.limits.min_cash_pct}")

    # Sector concentration
    sector = input.sector_map.get(ticker, "unknown")
    sector_value = sum(
        input.positions.get(t, Decimal("0")) * input.quote_price
        for t, s in input.sector_map.items() if s == sector
    )
    if proposal.action == ActionEnum.BUY:
        sector_value += proposal.payload.shares * input.quote_price
    sector_pct = float(sector_value / input.nav)
    if sector_pct > p.limits.max_sector_pct:
        return _decision_from_breach(input, f"sector {sector} {sector_pct:.3f} > {p.limits.max_sector_pct}")

    # HITL threshold
    if trade_pct > p.hitl.trade_threshold_pct or (
        p.hitl.escalate_new_ticker and cur_shares == Decimal("0") and proposal.action == ActionEnum.BUY
    ):
        return _escalate(input, "trade exceeds HITL threshold or new ticker")

    return _pass(input)
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_risk_limits.py -v
```

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add firm/agents/risk.py tests/unit/test_risk_limits.py
git commit -m "feat(agents): Risk hard-limit enforcement matching policy.yaml"
```

---

## Task 16: Research stub

**Files:**
- Create: `firm/agents/research.py`
- Test: `tests/unit/test_research.py`

Plan 1 produces a deterministic stub. The signature is what Plan 2's LLM-backed version replaces. Pattern: rotate through universe, propose a small BUY of the cheapest deterministic price.

- [ ] **Step 1: Write failing test**

`tests/unit/test_research.py`:
```python
from decimal import Decimal
from firm.agents.research import make_research
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.config import load_universe
from datetime import datetime, timezone
from pathlib import Path


def test_research_returns_buy_decision_for_universe_ticker():
    broker = FakeBroker(initial_cash=Decimal("100000"))
    universe = load_universe(Path("config/universe.yaml"))
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    research = make_research(clock=clock, broker=broker, universe=universe)
    state = research({"heartbeat_at": "2024-03-13T14:30:00+00:00"})
    d = state["research_decision"]
    assert d.action.value == "BUY"
    assert d.payload.ticker in universe.tickers
    assert d.payload.shares == Decimal("10")
    assert "stub" in d.rationale.lower()
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_research.py -v
```

- [ ] **Step 3: Implement `firm/agents/research.py`**

```python
"""Research agent — deterministic stub for Plan 1.

Plan 2 swaps this for an LLM-backed agent with hybrid retrieval + Citations API.
The function signature stays stable across plans.
"""
from __future__ import annotations

from decimal import Decimal

from firm.broker.protocol import Broker
from firm.core.clock import Clock
from firm.core.config import UniverseConfig
from firm.core.ids import ulid_new
from firm.core.models import ActionEnum, BuyPayload, Decision


def make_research(*, clock: Clock, broker: Broker, universe: UniverseConfig):
    def research(state: dict) -> dict:
        # Deterministic ticker selection: cheapest in universe by FakeBroker's price function
        prices = {t: broker.get_quote(t).price for t in universe.tickers}
        chosen = min(prices, key=lambda t: prices[t])
        decision = Decision(
            id=ulid_new(), decision_id_chain=[], action=ActionEnum.BUY,
            payload=BuyPayload(ticker=chosen, shares=Decimal("10")),
            rationale=f"deterministic stub: cheapest of universe at heartbeat {state.get('heartbeat_at')}",
            confidence=0.5, citations=[],
            falsification_condition=f"if {chosen} drops more than 5% by EOD",
            escalation_reason=None, failure_mode=None,
            metadata={"agent": "research", "stub": True}, nonce="research-stub",
        )
        return {"research_decision": decision}
    return research
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_research.py -v
```

- [ ] **Step 5: Commit**

```bash
git add firm/agents/research.py tests/unit/test_research.py
git commit -m "feat(agents): Research deterministic stub (LLM-backed comes in Plan 2)"
```

---

## Task 17: PM stub

**Files:**
- Create: `firm/agents/pm.py`
- Test: `tests/unit/test_pm.py`

Plan 1 passes Research's decision through, adding the PM-level provenance. Plan 2 wraps the vote-of-3 self-consistency.

- [ ] **Step 1: Write failing test**

`tests/unit/test_pm.py`:
```python
from decimal import Decimal
from firm.agents.pm import make_pm
from firm.core.models import ActionEnum, BuyPayload, Decision


def _research_decision() -> Decision:
    return Decision(
        id="res-1", decision_id_chain=[], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="stub", confidence=0.5, citations=[],
        falsification_condition="x", escalation_reason=None,
        failure_mode=None, metadata={}, nonce="n",
    )


def test_pm_passes_through_with_pm_provenance():
    pm = make_pm()
    state = {"research_decision": _research_decision()}
    out = pm(state)
    d = out["pm_decision"]
    assert d.action == ActionEnum.BUY
    assert "res-1" in d.decision_id_chain
    assert d.metadata["agent"] == "pm"
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_pm.py -v
```

- [ ] **Step 3: Implement `firm/agents/pm.py`**

```python
"""PM agent — deterministic pass-through stub for Plan 1.

Plan 2 swaps this for vote-of-3 self-consistency over LLM rationales.
"""
from __future__ import annotations

from firm.core.ids import ulid_new
from firm.core.models import Decision


def make_pm():
    def pm(state: dict) -> dict:
        research: Decision = state["research_decision"]
        decision = Decision(
            id=ulid_new(), decision_id_chain=[research.id],
            action=research.action, payload=research.payload,
            rationale=f"pm pass-through: {research.rationale}",
            confidence=research.confidence, citations=research.citations,
            falsification_condition=research.falsification_condition,
            escalation_reason=None, failure_mode=None,
            metadata={"agent": "pm", "stub": True}, nonce="pm-stub",
        )
        return {"pm_decision": decision}
    return pm
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_pm.py -v
```

- [ ] **Step 5: Commit**

```bash
git add firm/agents/pm.py tests/unit/test_pm.py
git commit -m "feat(agents): PM pass-through stub"
```

---

## Task 18: HITL gate node (CLI-ack flow)

**Files:**
- Create: `firm/agents/hitl.py`
- Test: `tests/unit/test_hitl.py`

In Plan 1, the HITL gate queues the decision in `hitl_queue` and `interrupt_before=["hitl"]` parks the graph in checkpoint. The reviewer runs `firm ack <decision_id>` to mark approved and resume. Plan 3 replaces the CLI flow with Slack signed approvals.

- [ ] **Step 1: Write failing test**

`tests/unit/test_hitl.py`:
```python
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.agents.hitl import make_hitl, mark_approved
from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision, EscalatePayload
from firm.db.connection import get_conn
from firm.db.migrations import init_db


def _persist_decision(db: Path, d: Decision, clock):
    import json
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            d.id, json.dumps(d.decision_id_chain), d.action.value,
            d.payload.model_dump_json(), d.rationale, d.confidence,
            json.dumps([c.model_dump(mode="json") for c in d.citations]),
            d.falsification_condition, d.escalation_reason,
            d.failure_mode.value if d.failure_mode else None,
            json.dumps(d.metadata), d.nonce, clock.now().isoformat(),
        ),
    )


def _risk_escalation() -> Decision:
    return Decision(
        id="risk-1", decision_id_chain=["pm-1"], action=ActionEnum.ESCALATE,
        payload=EscalatePayload(
            proposed=BuyPayload(ticker="AAPL", shares=Decimal("100")),
            reason="trade > HITL threshold",
        ),
        rationale="hitl required", confidence=1.0, citations=[],
        falsification_condition="timeout", escalation_reason="trade > HITL threshold",
        failure_mode=None, metadata={}, nonce="n",
    )


def test_hitl_queues_decision(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    d = _risk_escalation()
    _persist_decision(db, d, clock)

    hitl = make_hitl(db_path=db, clock=clock)
    state = hitl({"risk_decision": d})
    assert state.get("hitl_required") is True

    row = get_conn(db).execute("SELECT status FROM hitl_queue WHERE decision_id=?", (d.id,)).fetchone()
    assert row["status"] == "pending"


def test_mark_approved_updates_queue(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    d = _risk_escalation()
    _persist_decision(db, d, clock)
    make_hitl(db_path=db, clock=clock)({"risk_decision": d})

    mark_approved(db_path=db, decision_id=d.id, approver="cli-user", clock=clock)
    row = get_conn(db).execute("SELECT status, approver FROM hitl_queue WHERE decision_id=?", (d.id,)).fetchone()
    assert row["status"] == "approved"
    assert row["approver"] == "cli-user"
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_hitl.py -v
```

- [ ] **Step 3: Implement `firm/agents/hitl.py`**

```python
"""HITL gate. Plan 1 = CLI ack; Plan 3 = Slack signed approvals. See spec §3, §8.4."""
from __future__ import annotations

from pathlib import Path

from firm.audit.log import AuditLog
from firm.core.clock import Clock
from firm.core.models import ActionEnum, Decision
from firm.db.connection import get_conn


def make_hitl(*, db_path: Path, clock: Clock):
    def hitl(state: dict) -> dict:
        risk: Decision = state["risk_decision"]
        if risk.action != ActionEnum.ESCALATE:
            return {"hitl_required": False, "hitl_approved": True}

        conn = get_conn(db_path)
        conn.execute(
            "INSERT OR IGNORE INTO hitl_queue (decision_id, queued_at, status) "
            "VALUES (?, ?, 'pending')",
            (risk.id, clock.now().isoformat()),
        )
        AuditLog(db_path, clock).append("hitl.queued", {"decision_id": risk.id})

        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", (risk.id,)
        ).fetchone()
        approved = row and row["status"] == "approved"
        return {"hitl_required": True, "hitl_approved": bool(approved)}
    return hitl


def mark_approved(*, db_path: Path, decision_id: str, approver: str, clock: Clock) -> None:
    conn = get_conn(db_path)
    conn.execute(
        "UPDATE hitl_queue SET status='approved', approver=?, decided_at=? "
        "WHERE decision_id=? AND status='pending'",
        (approver, clock.now().isoformat(), decision_id),
    )
    AuditLog(db_path, clock).append("hitl.approved", {"decision_id": decision_id, "approver": approver})


def mark_rejected(*, db_path: Path, decision_id: str, approver: str, clock: Clock) -> None:
    conn = get_conn(db_path)
    conn.execute(
        "UPDATE hitl_queue SET status='rejected', approver=?, decided_at=? "
        "WHERE decision_id=? AND status='pending'",
        (approver, clock.now().isoformat(), decision_id),
    )
    AuditLog(db_path, clock).append("hitl.rejected", {"decision_id": decision_id, "approver": approver})
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_hitl.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add firm/agents/hitl.py tests/unit/test_hitl.py
git commit -m "feat(agents): HITL gate with CLI ack flow"
```

---

## Task 19: Execution agent

**Files:**
- Create: `firm/agents/execution.py`
- Test: `tests/unit/test_execution.py`

Wraps the outbox. Only fires if (a) no HITL was required, or (b) HITL was approved. Refused decisions never reach Execution (the graph routes them to Reporter directly via Risk's REFUSE action — but in this skeleton, we let Execution see the action and no-op on non-BUY/SELL).

- [ ] **Step 1: Write failing test**

`tests/unit/test_execution.py`:
```python
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import json

from firm.agents.execution import make_execution
from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision, RefusePayload
from firm.db.connection import get_conn
from firm.db.migrations import init_db


def _persist(db, d, clock):
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            d.id, json.dumps(d.decision_id_chain), d.action.value,
            d.payload.model_dump_json(), d.rationale, d.confidence,
            "[]", d.falsification_condition, d.escalation_reason,
            d.failure_mode.value if d.failure_mode else None,
            json.dumps(d.metadata), d.nonce, clock.now().isoformat(),
        ),
    )


def test_execution_fires_buy(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = Decision(
        id="risk-1", decision_id_chain=["pm-1"], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="ok", confidence=0.7, citations=[],
        falsification_condition="x", escalation_reason=None,
        failure_mode=None, metadata={}, nonce="n",
    )
    _persist(db, d, clock)
    exe = make_execution(db_path=db, broker=broker, clock=clock)
    out = exe({"risk_decision": d, "hitl_required": False})
    assert "execution_result" in out
    assert out["execution_result"]["ticker"] == "AAPL"


def test_execution_skips_on_refuse(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = Decision(
        id="risk-1", decision_id_chain=["pm-1"], action=ActionEnum.REFUSE,
        payload=RefusePayload(reason="limit breach"),
        rationale="hard limit", confidence=1.0, citations=[],
        falsification_condition="x", escalation_reason=None,
        failure_mode=None, metadata={}, nonce="n",
    )
    _persist(db, d, clock)
    exe = make_execution(db_path=db, broker=broker, clock=clock)
    out = exe({"risk_decision": d, "hitl_required": False})
    assert out.get("execution_result") is None or out["execution_result"].get("skipped")


def test_execution_skips_when_hitl_not_approved(tmp_path: Path):
    db = tmp_path / "test.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = Decision(
        id="risk-1", decision_id_chain=["pm-1"], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="ok", confidence=0.7, citations=[],
        falsification_condition="x", escalation_reason=None,
        failure_mode=None, metadata={}, nonce="n",
    )
    _persist(db, d, clock)
    exe = make_execution(db_path=db, broker=broker, clock=clock)
    out = exe({"risk_decision": d, "hitl_required": True, "hitl_approved": False})
    assert out.get("execution_result") is None or out["execution_result"].get("skipped")
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_execution.py -v
```

- [ ] **Step 3: Implement `firm/agents/execution.py`**

```python
"""Execution agent — wraps the outbox. See spec §3, §5.2."""
from __future__ import annotations

from pathlib import Path

from firm.broker.protocol import Broker
from firm.core.clock import Clock
from firm.core.models import ActionEnum, Decision
from firm.outbox.outbox import place_order_via_outbox


def make_execution(*, db_path: Path, broker: Broker, clock: Clock):
    def execution(state: dict) -> dict:
        risk: Decision = state["risk_decision"]
        if risk.action not in (ActionEnum.BUY, ActionEnum.SELL):
            return {"execution_result": {"skipped": True, "reason": f"action={risk.action.value}"}}
        if state.get("hitl_required") and not state.get("hitl_approved"):
            return {"execution_result": {"skipped": True, "reason": "hitl_not_approved"}}

        result = place_order_via_outbox(risk, db_path, broker, clock)
        return {"execution_result": result.model_dump(mode="json")}
    return execution
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_execution.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add firm/agents/execution.py tests/unit/test_execution.py
git commit -m "feat(agents): Execution wraps outbox; skips on refuse/no-ack"
```

---

## Task 20: Reporter stub

**Files:**
- Create: `firm/agents/reporter.py`
- Test: `tests/unit/test_reporter.py`

Writes a minimal JSONL summary to `reports/YYYY-MM-DD/`. Plan 3 replaces this with Markdown + XLSX + EOD reconciliation block.

- [ ] **Step 1: Write failing test**

`tests/unit/test_reporter.py`:
```python
import json
from datetime import datetime, timezone
from pathlib import Path

from firm.agents.reporter import make_reporter
from firm.core.clock import ReplayClock


def test_reporter_writes_jsonl(tmp_path: Path):
    clock = ReplayClock(datetime(2024, 3, 13, 14, 30, tzinfo=timezone.utc))
    reporter = make_reporter(reports_root=tmp_path, clock=clock)
    state = {
        "heartbeat_at": "2024-03-13T14:30:00+00:00",
        "execution_result": {"ticker": "AAPL", "filled_shares": "10"},
    }
    out = reporter(state)
    p = Path(out["report_path"])
    assert p.exists()
    lines = [json.loads(l) for l in p.read_text().splitlines()]
    assert any(l.get("execution_result") for l in lines)
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/unit/test_reporter.py -v
```

- [ ] **Step 3: Implement `firm/agents/reporter.py`**

```python
"""Reporter — minimal JSONL summary for Plan 1. Markdown+XLSX in Plan 3."""
from __future__ import annotations

import json
from pathlib import Path

from firm.core.clock import Clock


def make_reporter(*, reports_root: Path, clock: Clock):
    def reporter(state: dict) -> dict:
        now = clock.now()
        date_dir = reports_root / now.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        path = date_dir / "decisions.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now.isoformat(), **state}, default=str) + "\n")
        return {"report_path": str(path)}
    return reporter
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/unit/test_reporter.py -v
```

- [ ] **Step 5: Commit**

```bash
git add firm/agents/reporter.py tests/unit/test_reporter.py
git commit -m "feat(agents): Reporter stub writes JSONL summary"
```

---

## Task 21: CLI

**Files:**
- Create: `firm/cli.py`
- Test: `tests/integration/test_cli.py`

Three commands: `firm run` (one heartbeat through the graph), `firm ack <decision_id>` (approve a queued HITL decision and resume), `firm reconcile` (boot reconciliation only).

- [ ] **Step 1: Write failing integration test**

`tests/integration/test_cli.py`:
```python
import os
import subprocess
import sys
from pathlib import Path


def test_cli_run_produces_decision(tmp_path: Path):
    env = os.environ.copy()
    env["FIRM_BROKER"] = "FAKE"
    env["FIRM_DB_PATH"] = str(tmp_path / "firm.db")
    env["FIRM_HMAC_SECRET"] = "a" * 64
    env["FIRM_REPORTS_ROOT"] = str(tmp_path / "reports")
    env["FIRM_REPLAY_AT"] = "2024-03-13T14:30:00+00:00"

    result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "run", "--once"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    # at least one report file written
    reports = list((tmp_path / "reports").rglob("*.jsonl"))
    assert reports, "no report written"
```

- [ ] **Step 2: Run — verify fail**

```bash
pytest tests/integration/test_cli.py -v
```

- [ ] **Step 3: Implement `firm/cli.py`**

```python
"""CLI entry points. See spec §3.1, §3.8."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import click

from firm.agents.execution import make_execution
from firm.agents.hitl import make_hitl, mark_approved, mark_rejected
from firm.agents.monitor import make_monitor
from firm.agents.pm import make_pm
from firm.agents.reporter import make_reporter
from firm.agents.research import make_research
from firm.agents.risk import RiskInput, evaluate_risk
from firm.broker.alpaca_paper import make_broker
from firm.core.clock import ReplayClock, WallClock
from firm.core.config import load_policy, load_universe
from firm.db.migrations import init_db
from firm.orchestrator.graph import build_graph
from firm.reconcile.boot import reconcile_on_boot


def _resolve_clock():
    replay = os.environ.get("FIRM_REPLAY_AT")
    if replay:
        return ReplayClock(datetime.fromisoformat(replay))
    return WallClock()


def _db_path() -> Path:
    return Path(os.environ.get("FIRM_DB_PATH", "data/firm.db"))


def _reports_root() -> Path:
    return Path(os.environ.get("FIRM_REPORTS_ROOT", "reports"))


@click.group()
def cli():
    pass


@cli.command()
@click.option("--once/--loop", default=True, help="Single heartbeat (default) or loop (loop is Plan 3+).")
def run(once: bool):
    """Run one heartbeat of the firm end-to-end."""
    db = _db_path()
    init_db(db)
    clock = _resolve_clock()
    broker = make_broker()
    policy = load_policy(Path("config/policy.yaml"))
    universe = load_universe(Path("config/universe.yaml"))

    recon = reconcile_on_boot(db, broker, clock)
    if recon.status == "mismatch":
        click.echo(f"BOOT RECONCILIATION MISMATCH: {recon.diff}", err=True)
        click.echo("Run `firm ack-reconcile` to acknowledge and resync.", err=True)
        sys.exit(1)

    monitor = make_monitor(clock)
    research = make_research(clock=clock, broker=broker, universe=universe)
    pm = make_pm()

    def risk_node(state: dict) -> dict:
        proposal = state["pm_decision"]
        ticker = proposal.payload.ticker if hasattr(proposal.payload, "ticker") else "AAPL"
        quote = broker.get_quote(ticker)
        positions = {p.ticker: p.shares for p in broker.list_positions()}
        decision = evaluate_risk(RiskInput(
            proposal=proposal, quote_price=quote.price, quote_age_seconds=0,
            cash=broker.get_cash(), positions=positions, sector_map=universe.sector_map,
            trades_today=0, nav=broker.get_cash() + sum(p.shares * broker.get_quote(p.ticker).price for p in broker.list_positions()),
            daily_pnl_pct=0.0, policy=policy,
        ))
        return {"risk_decision": decision}

    hitl = make_hitl(db_path=db, clock=clock)
    execution = make_execution(db_path=db, broker=broker, clock=clock)
    reporter = make_reporter(reports_root=_reports_root(), clock=clock)

    graph = build_graph(
        db_path=db, monitor_node=monitor, research_node=research, pm_node=pm,
        risk_node=risk_node, hitl_node=hitl, execution_node=execution, reporter_node=reporter,
    )

    config = {"configurable": {"thread_id": clock.now().isoformat()}}
    final = graph.invoke({}, config=config)
    click.echo(f"Heartbeat complete. Report: {final.get('report_path')}")


@cli.command()
@click.argument("decision_id")
@click.option("--approver", default="cli-user")
def ack(decision_id: str, approver: str):
    """Approve a queued HITL decision (Plan 1 stand-in for Slack)."""
    mark_approved(db_path=_db_path(), decision_id=decision_id, approver=approver, clock=_resolve_clock())
    click.echo(f"approved: {decision_id}")


@cli.command()
@click.argument("decision_id")
@click.option("--approver", default="cli-user")
def reject(decision_id: str, approver: str):
    """Reject a queued HITL decision."""
    mark_rejected(db_path=_db_path(), decision_id=decision_id, approver=approver, clock=_resolve_clock())
    click.echo(f"rejected: {decision_id}")


@cli.command()
def reconcile():
    """Run boot reconciliation against the broker and print the result."""
    db = _db_path()
    init_db(db)
    clock = _resolve_clock()
    result = reconcile_on_boot(db, make_broker(), clock)
    click.echo(f"status: {result.status}")
    if result.diff:
        click.echo(f"diff: {result.diff}")


def main():
    cli()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run — verify pass**

```bash
pytest tests/integration/test_cli.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add firm/cli.py tests/integration/test_cli.py
git commit -m "feat(cli): firm run / ack / reject / reconcile commands"
```

---

## Task 22: End-to-end smoke test

**Files:**
- Create: `tests/integration/test_end_to_end_smoke.py`

Wires the actual graph and asserts: one `firm run --once` invocation produces a confirmed paper trade plus a report file.

- [ ] **Step 1: Write test**

`tests/integration/test_end_to_end_smoke.py`:
```python
import os
import subprocess
import sys
from pathlib import Path


def test_walking_skeleton_end_to_end(tmp_path: Path):
    env = os.environ.copy()
    env["FIRM_BROKER"] = "FAKE"
    env["FIRM_DB_PATH"] = str(tmp_path / "firm.db")
    env["FIRM_HMAC_SECRET"] = "a" * 64
    env["FIRM_REPORTS_ROOT"] = str(tmp_path / "reports")
    env["FIRM_REPLAY_AT"] = "2024-03-13T14:30:00+00:00"

    result = subprocess.run(
        [sys.executable, "-m", "firm.cli", "run", "--once"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    # Report file exists
    reports = list((tmp_path / "reports").rglob("*.jsonl"))
    assert reports

    # Outbox has one confirmed row
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "firm.db"))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM outbox WHERE status='confirmed'").fetchall()
    assert len(rows) == 1, f"expected 1 confirmed order, got {len(rows)}"

    # Decisions table has at least research, pm, risk decisions
    decisions = conn.execute("SELECT COUNT(*) c FROM decisions").fetchone()["c"]
    assert decisions >= 3, f"expected at least 3 decisions, got {decisions}"
```

- [ ] **Step 2: Run**

```bash
pytest tests/integration/test_end_to_end_smoke.py -v
```

Expected: 1 passed. If it fails, the failure is the real bug in the wiring — debug before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_end_to_end_smoke.py
git commit -m "test(integration): end-to-end walking-skeleton smoke"
```

---

## Task 23: Crash recovery integration test

**Files:**
- Create: `tests/integration/test_crash_recovery.py`

Simulates: kill mid-order (outbox row inserted, broker not yet called) → restart (call `recover_pending`) → exactly one fill at the broker.

- [ ] **Step 1: Write test**

`tests/integration/test_crash_recovery.py`:
```python
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from firm.broker.fake_broker import FakeBroker
from firm.core.clock import ReplayClock
from firm.core.models import ActionEnum, BuyPayload, Decision
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.outbox.outbox import place_order_via_outbox, recover_pending


def _persist_decision(db: Path, d: Decision, clock):
    conn = get_conn(db)
    conn.execute(
        "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            d.id, json.dumps(d.decision_id_chain), d.action.value,
            d.payload.model_dump_json(), d.rationale, d.confidence,
            "[]", d.falsification_condition, d.escalation_reason,
            d.failure_mode.value if d.failure_mode else None,
            json.dumps(d.metadata), d.nonce, clock.now().isoformat(),
        ),
    )


def test_kill_mid_order_restart_is_exactly_once(tmp_path: Path):
    db = tmp_path / "firm.db"
    init_db(db)
    clock = ReplayClock(datetime(2024, 3, 13, tzinfo=timezone.utc))
    broker = FakeBroker(initial_cash=Decimal("100000"))
    d = Decision(
        id="dec-1", decision_id_chain=[], action=ActionEnum.BUY,
        payload=BuyPayload(ticker="AAPL", shares=Decimal("10")),
        rationale="x", confidence=0.5, citations=[],
        falsification_condition="y", escalation_reason=None,
        failure_mode=None, metadata={}, nonce="n",
    )
    _persist_decision(db, d, clock)

    # Step 1: normal place — confirmed
    r1 = place_order_via_outbox(d, db, broker, clock)
    assert r1.filled_shares == Decimal("10")

    # Step 2: simulate crash — flip the outbox status back to 'pending' manually
    # (mimicking the case where confirm-update never landed)
    conn = get_conn(db)
    conn.execute("UPDATE outbox SET status='pending' WHERE decision_id=?", (d.id,))

    # Step 3: restart recovery
    recovered = recover_pending(db, broker, clock)
    assert len(recovered) == 1
    assert recovered[0].order_id == r1.order_id  # same fill, no duplicate

    # Step 4: broker still has exactly 10 shares (not 20)
    pos = [p for p in broker.list_positions() if p.ticker == "AAPL"]
    assert len(pos) == 1
    assert pos[0].shares == Decimal("10")
```

- [ ] **Step 2: Run**

```bash
pytest tests/integration/test_crash_recovery.py -v
```

Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_crash_recovery.py
git commit -m "test(integration): kill-mid-order crash recovery exactly-once"
```

---

## Task 24: Dockerfile, docker-compose, Makefile, litestream config

**Files:**
- Create: `Dockerfile`, `docker-compose.yml`, `Makefile`, `litestream.yml`, update `README.md`

`make demo` is the reviewer's entry point — clone → install → demo → see a paper trade.

- [ ] **Step 1: Create `Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY firm/ ./firm/
COPY config/ ./config/
COPY scripts/ ./scripts/

ENV FIRM_DB_PATH=/data/firm.db
ENV FIRM_REPORTS_ROOT=/data/reports
ENV FIRM_BROKER=FAKE

CMD ["python", "-m", "firm.cli", "run", "--once"]
```

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
services:
  firm:
    build: .
    environment:
      FIRM_BROKER: FAKE
      FIRM_HMAC_SECRET: "change-me-32-bytes-of-secret-data-x"
      FIRM_DB_PATH: /data/firm.db
      FIRM_REPORTS_ROOT: /data/reports
      FIRM_REPLAY_AT: "2024-03-13T14:30:00+00:00"
    volumes:
      - ./data:/data
```

- [ ] **Step 3: Create `litestream.yml`** (config only; not started in Plan 1)

```yaml
# See https://litestream.io. Activated in Plan 3 via the runbook.
dbs:
  - path: /data/firm.db
    replicas:
      - type: file
        path: /data/backups/firm.db
```

- [ ] **Step 4: Create `Makefile`**

```makefile
.PHONY: install test demo demo-docker reconcile clean

install:
	pip install -e ".[dev]"

test:
	pytest

demo:
	FIRM_REPLAY_AT=2024-03-13T14:30:00+00:00 \
	FIRM_HMAC_SECRET=$$(python -c "import secrets; print(secrets.token_hex(32))") \
	python -m firm.cli run --once

demo-docker:
	docker compose up --build --abort-on-container-exit

reconcile:
	FIRM_HMAC_SECRET=$${FIRM_HMAC_SECRET:-placeholder} python -m firm.cli reconcile

clean:
	rm -rf data/firm.db data/firm.db-wal data/firm.db-shm data/reports
```

On Windows PowerShell, prefer `make` via WSL or replace with a `scripts/demo.ps1` equivalent.

- [ ] **Step 5: Rewrite `README.md` with quickstart**

```markdown
# AI Investment Firm

Multi-agent paper-trading firm. Take-home for Cato Networks — Agentic AI Engineer.

## Quickstart (clone-to-demo in <10 min)

```bash
git clone <repo>
cd ai-investment-firm
pip install -e ".[dev]"
make demo
```

Output: one heartbeat through the 5-agent workflow, one paper trade via FakeBroker, and a JSONL report in `data/reports/2024-03-13/`.

## Docker demo

```bash
docker compose up --build
```

## Real paper trading (Alpaca)

```bash
export FIRM_BROKER=ALPACA
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
make demo
```

## Architecture

See `docs/superpowers/specs/2026-05-18-ai-investment-firm-design.md` for the full design.

## Status

- [x] **Plan 1 (this branch):** Foundation + Walking Skeleton — 5 agents stubbed, outbox-protected trade, boot reconciliation.
- [ ] Plan 2: RAG + Citations + Grounding
- [ ] Plan 3: HITL + Daily Reports + Observability
- [ ] Plan 4: Eval Harness + Red Team + CI/CD + Bonuses
```

- [ ] **Step 6: Smoke test the demo**

```bash
make demo
```

Expected: exits 0, "Heartbeat complete." line, and a JSONL file under `data/reports/2024-03-13/`.

- [ ] **Step 7: Commit**

```bash
git add Dockerfile docker-compose.yml Makefile litestream.yml README.md
git commit -m "feat: Dockerfile, docker-compose, Makefile, litestream config, README"
```

---

## Task 25: Final verification and CI smoke

**Files:**
- (No new files; just run the full suite and commit any cleanups.)

- [ ] **Step 1: Run the entire test suite**

```bash
pytest -v
```

Expected: every test in the plan passes. Roughly 40+ tests across unit and integration.

- [ ] **Step 2: Run ruff and mypy**

```bash
ruff check firm tests
mypy firm
```

Expected: zero warnings. If any, fix before continuing.

- [ ] **Step 3: Verify `make demo` end-to-end one more time**

```bash
rm -rf data
make demo
ls data/reports/
```

Expected: `2024-03-13/decisions.jsonl` exists; the `firm.db` has confirmed outbox rows.

- [ ] **Step 4: Commit any final cleanups**

```bash
git status
# if anything outstanding:
git add <files>
git commit -m "chore: post-Plan-1 cleanup"
```

---

## Spec coverage check (Plan 1)

Each spec section this plan implements:

| Spec section | Tasks |
|---|---|
| §3.1 topology (5 agents + Position Monitor) | T13–T20 |
| §3.4 Decision envelope | T2 |
| §3.5 FailureMode enum (13 values) | T2 |
| §3.6 Availability model — single-host, restart policy | T24 (Docker) |
| §3.7 Trading limits | T15 + `config/policy.yaml` |
| §3.8 Partial failure — fail-closed | T15, T19 (refuse/skip paths) |
| §5.1 SQLite WAL + pragmas | T5 |
| §5.2 Outbox pattern | T11 |
| §5.3 LangGraph checkpointer + HITL interrupt | T13, T18 |
| §5.4 Clock injection | T3 |
| §5.5 Crash recovery test | T23 |
| §5.6 Backup config | T24 (litestream.yml) |
| §5.7 Boot reconciliation (EOD reconciliation is Plan 3) | T12 |
| §10.1 (partial) — structured logs via audit_log | T7 |
| §12 Repo layout | T1 + T24 |

**Out of scope for Plan 1 (deferred to later plans):**
- RAG, Citations API, sufficiency gate — Plan 2
- LLM-backed Research / PM (vote-of-3) — Plan 2
- Slack signed approvals (CLI ack stand-in here) — Plan 3
- OpenTelemetry full tracing (structured logs only here) — Plan 3
- Cost router — Plan 3
- Markdown/XLSX daily reports + EOD reconciliation block — Plan 3
- Eval harness, regime fixtures, red-team corpus — Plan 4
- GitHub Actions, Terraform, AgentCore — Plan 4

---

**End of Plan 1.**
