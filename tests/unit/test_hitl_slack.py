"""Tests for firm.hitl.slack — FastAPI endpoint for Slack interactive callbacks.

Golden-payload tests: all 8 cases from T11 spec.
Clock is pinned at epoch 1700000000 (2023-11-14 22:13:20 UTC) so Slack
timestamp and internal HMAC timestamp are self-consistent.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

from fastapi.testclient import TestClient

from firm.core.clock import ReplayClock
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.hitl.signing import sign
from firm.hitl.slack import build_app

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SLACK_SECRET = b"slack-secret-test"
_INTERNAL_SECRET = b"internal-secret-test"
_TS = "1700000000"  # int epoch matching pinned clock
_TS_INT = 1700000000
_DECISION_ID = "dec-slack-t11"
_APPROVER_ID = "U123"

# Clock pinned at exactly epoch 1700000000 so replay-window checks pass.
_CLOCK = ReplayClock(datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_slack_request(
    *, raw_body: str, ts: str, slack_secret: bytes
) -> tuple[dict[str, str], str]:
    """Return (headers, raw_body) for a Slack interactive POST.

    raw_body must be the exact bytes Slack would send (form-urlencoded).
    """
    sig = "v0=" + hmac.new(
        slack_secret, f"v0:{ts}:{raw_body}".encode(), hashlib.sha256
    ).hexdigest()
    return {
        "X-Slack-Signature": sig,
        "X-Slack-Request-Timestamp": ts,
        "Content-Type": "application/x-www-form-urlencoded",
    }, raw_body


def _internal_sig(decision_id: str, approver_id: str, ts: int) -> str:
    return sign(
        decision_id=decision_id,
        approver_id=approver_id,
        ts=ts,
        secret=_INTERNAL_SECRET,
    )


def _make_button_value(
    action: str,
    decision_id: str = _DECISION_ID,
    approver_id: str = _APPROVER_ID,
    ts: int = _TS_INT,
    sig: str | None = None,
) -> str:
    if sig is None:
        sig = _internal_sig(decision_id, approver_id, ts)
    return json.dumps(
        {
            "decision_id": decision_id,
            "approver_id": approver_id,
            "ts": ts,
            "action": action,
            "sig": sig,
        }
    )


def _make_slack_payload(
    action: str = "approve",
    user_id: str = _APPROVER_ID,
    decision_id: str = _DECISION_ID,
    approver_id: str = _APPROVER_ID,
    ts: int = _TS_INT,
    sig: str | None = None,
) -> str:
    """Return the JSON string that Slack would send as the ``payload`` form field."""
    bv = _make_button_value(
        action=action,
        decision_id=decision_id,
        approver_id=approver_id,
        ts=ts,
        sig=sig,
    )
    return json.dumps(
        {
            "user": {"id": user_id},
            "actions": [{"value": bv}],
        }
    )


def _make_form_body(slack_payload_json: str) -> str:
    """Encode the Slack payload JSON as a form-urlencoded body."""
    return "payload=" + quote_plus(slack_payload_json)


def _seed_decision(db_path: Path) -> None:
    """Insert a minimal decisions row and a pending hitl_queue row."""
    now_iso = _CLOCK.now().isoformat()
    with closing(get_conn(db_path)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO decisions "
            "(id, parent_chain, action, payload, rationale, confidence, "
            "citations, falsification, escalation, failure_mode, metadata, nonce, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                _DECISION_ID, "[]", "ESCALATE", "{}", "test-rationale", 0.9,
                "[]", "test-falsification", "test-escalation", None, "{}", "nonce-t11",
                now_iso,
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO hitl_queue (decision_id, queued_at, status) "
            "VALUES (?, ?, 'pending')",
            (_DECISION_ID, now_iso),
        )


def _build_client(db_path: Path) -> TestClient:
    app = build_app(
        db_path=db_path,
        clock=_CLOCK,
        slack_signing_secret=_SLACK_SECRET,
        internal_secret=_INTERNAL_SECRET,
    )
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Test 1 — happy path: approve
# ---------------------------------------------------------------------------


def test_approve_flow_writes_audit_and_returns_200(tmp_path: Path) -> None:
    db = tmp_path / "t11.db"
    init_db(db)
    _seed_decision(db)

    slack_payload = _make_slack_payload(action="approve")
    raw_body = _make_form_body(slack_payload)
    headers, body = _make_slack_request(
        raw_body=raw_body, ts=_TS, slack_secret=_SLACK_SECRET
    )

    client = _build_client(db)
    resp = client.post("/slack/interactive", content=body, headers=headers)

    assert resp.status_code == 200
    assert "approved" in resp.json().get("text", "")

    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", (_DECISION_ID,)
        ).fetchone()
        audit = conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE event='hitl.approved' AND detail LIKE ?",
            (f"%{_DECISION_ID}%",),
        ).fetchone()["n"]

    assert row["status"] == "approved"
    assert audit >= 1


# ---------------------------------------------------------------------------
# Test 2 — happy path: reject
# ---------------------------------------------------------------------------


def test_reject_flow_writes_audit_and_returns_200(tmp_path: Path) -> None:
    db = tmp_path / "t11.db"
    init_db(db)
    _seed_decision(db)

    slack_payload = _make_slack_payload(action="reject")
    raw_body = _make_form_body(slack_payload)
    headers, body = _make_slack_request(
        raw_body=raw_body, ts=_TS, slack_secret=_SLACK_SECRET
    )

    client = _build_client(db)
    resp = client.post("/slack/interactive", content=body, headers=headers)

    assert resp.status_code == 200
    assert "rejected" in resp.json().get("text", "")

    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", (_DECISION_ID,)
        ).fetchone()
        audit = conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE event='hitl.rejected' AND detail LIKE ?",
            (f"%{_DECISION_ID}%",),
        ).fetchone()["n"]

    assert row["status"] == "rejected"
    assert audit >= 1


# ---------------------------------------------------------------------------
# Test 3 — invalid Slack signature → 401, no DB mutation
# ---------------------------------------------------------------------------


def test_invalid_slack_signature_returns_401(tmp_path: Path) -> None:
    db = tmp_path / "t11.db"
    init_db(db)
    _seed_decision(db)

    slack_payload = _make_slack_payload(action="approve")
    raw_body = _make_form_body(slack_payload)
    # Use wrong secret for Slack outer signature
    headers, body = _make_slack_request(
        raw_body=raw_body, ts=_TS, slack_secret=b"wrong-slack-secret"
    )

    client = _build_client(db)
    resp = client.post("/slack/interactive", content=body, headers=headers)

    assert resp.status_code == 401
    assert "invalid Slack signature" in resp.json().get("error", "")

    # DB must not be mutated
    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", (_DECISION_ID,)
        ).fetchone()
    assert row["status"] == "pending"


# ---------------------------------------------------------------------------
# Test 4 — stale Slack timestamp → 401
# ---------------------------------------------------------------------------


def test_stale_slack_timestamp_returns_401(tmp_path: Path) -> None:
    db = tmp_path / "t11.db"
    init_db(db)
    _seed_decision(db)

    stale_ts = str(_TS_INT - 301)  # 301 s before pinned clock → outside 300 s window
    slack_payload = _make_slack_payload(action="approve")
    raw_body = _make_form_body(slack_payload)
    headers, body = _make_slack_request(
        raw_body=raw_body, ts=stale_ts, slack_secret=_SLACK_SECRET
    )

    client = _build_client(db)
    resp = client.post("/slack/interactive", content=body, headers=headers)

    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test 5 — valid Slack sig but invalid internal HMAC → 401, no DB mutation
# ---------------------------------------------------------------------------


def test_invalid_internal_signature_returns_401(tmp_path: Path) -> None:
    db = tmp_path / "t11.db"
    init_db(db)
    _seed_decision(db)

    # Pass a corrupted internal sig
    slack_payload = _make_slack_payload(action="approve", sig="deadbeef" * 8)
    raw_body = _make_form_body(slack_payload)
    headers, body = _make_slack_request(
        raw_body=raw_body, ts=_TS, slack_secret=_SLACK_SECRET
    )

    client = _build_client(db)
    resp = client.post("/slack/interactive", content=body, headers=headers)

    assert resp.status_code == 401
    assert "invalid internal signature" in resp.json().get("error", "")

    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", (_DECISION_ID,)
        ).fetchone()
    assert row["status"] == "pending"


# ---------------------------------------------------------------------------
# Test 6 — approver_id mismatch (different Slack user clicked) → 401
# ---------------------------------------------------------------------------


def test_approver_id_mismatch_returns_401(tmp_path: Path) -> None:
    db = tmp_path / "t11.db"
    init_db(db)
    _seed_decision(db)

    # payload.user.id is U999 but button approver_id is U123
    slack_payload = _make_slack_payload(action="approve", user_id="U999")
    raw_body = _make_form_body(slack_payload)
    headers, body = _make_slack_request(
        raw_body=raw_body, ts=_TS, slack_secret=_SLACK_SECRET
    )

    client = _build_client(db)
    resp = client.post("/slack/interactive", content=body, headers=headers)

    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Test 7 — malformed body (missing payload field) → 400
# ---------------------------------------------------------------------------


def test_malformed_payload_returns_400(tmp_path: Path) -> None:
    db = tmp_path / "t11.db"
    init_db(db)

    raw_body = "not_a_payload_field=something"
    headers, body = _make_slack_request(
        raw_body=raw_body, ts=_TS, slack_secret=_SLACK_SECRET
    )

    client = _build_client(db)
    resp = client.post("/slack/interactive", content=body, headers=headers)

    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Test 8 — unknown action → 400
# ---------------------------------------------------------------------------


def test_unknown_action_returns_400(tmp_path: Path) -> None:
    db = tmp_path / "t11.db"
    init_db(db)
    _seed_decision(db)

    slack_payload = _make_slack_payload(action="delete")
    raw_body = _make_form_body(slack_payload)
    headers, body = _make_slack_request(
        raw_body=raw_body, ts=_TS, slack_secret=_SLACK_SECRET
    )

    client = _build_client(db)
    resp = client.post("/slack/interactive", content=body, headers=headers)

    assert resp.status_code == 400
    assert "unknown action" in resp.json().get("error", "")


# ---------------------------------------------------------------------------
# Test 9 — T13a dual-key rotation: button signed under previous key accepted
# while inside grace window, rejected once expired.
# ---------------------------------------------------------------------------


_PREV_INTERNAL_SECRET = b"internal-secret-previous"


def _build_client_with_rotation(
    db_path: Path,
    *,
    rotated_at: int,
) -> TestClient:
    app = build_app(
        db_path=db_path,
        clock=_CLOCK,
        slack_signing_secret=_SLACK_SECRET,
        internal_secret=_INTERNAL_SECRET,
        previous_internal_secret=_PREV_INTERNAL_SECRET,
        internal_rotated_at=rotated_at,
    )
    return TestClient(app, raise_server_exceptions=True)


def test_rotation_accepts_previous_key_inside_grace_window(tmp_path: Path) -> None:
    db = tmp_path / "t11.db"
    init_db(db)
    _seed_decision(db)

    prev_sig = sign(
        decision_id=_DECISION_ID,
        approver_id=_APPROVER_ID,
        ts=_TS_INT,
        secret=_PREV_INTERNAL_SECRET,
    )
    slack_payload = _make_slack_payload(action="approve", sig=prev_sig)
    raw_body = _make_form_body(slack_payload)
    headers, body = _make_slack_request(
        raw_body=raw_body, ts=_TS, slack_secret=_SLACK_SECRET
    )

    # rotated_at = 1 hour ago (well inside 24h default grace window).
    client = _build_client_with_rotation(db, rotated_at=_TS_INT - 3600)
    resp = client.post("/slack/interactive", content=body, headers=headers)

    assert resp.status_code == 200
    assert "approved" in resp.json().get("text", "")


def test_rotation_rejects_previous_key_after_grace_window(tmp_path: Path) -> None:
    db = tmp_path / "t11.db"
    init_db(db)
    _seed_decision(db)

    prev_sig = sign(
        decision_id=_DECISION_ID,
        approver_id=_APPROVER_ID,
        ts=_TS_INT,
        secret=_PREV_INTERNAL_SECRET,
    )
    slack_payload = _make_slack_payload(action="approve", sig=prev_sig)
    raw_body = _make_form_body(slack_payload)
    headers, body = _make_slack_request(
        raw_body=raw_body, ts=_TS, slack_secret=_SLACK_SECRET
    )

    # rotated_at = 25 hours ago (past default 24h grace).
    client = _build_client_with_rotation(db, rotated_at=_TS_INT - 25 * 3600)
    resp = client.post("/slack/interactive", content=body, headers=headers)

    assert resp.status_code == 401
    assert "invalid internal signature" in resp.json().get("error", "")
