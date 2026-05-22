"""Plan 3 T25 — SIGNED_APPROVAL_INVALID end-to-end triggering fixture.

Verifies that when the Slack interactive handler receives a request with a
valid Slack outer signature but a tampered internal button HMAC, the handler:
  1. Returns HTTP 401.
  2. Writes an audit_log entry with kind='hitl.signature_rejected' and
     failure_mode='signed_approval_invalid'.
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
from firm.core.models import FailureMode
from firm.db.connection import get_conn
from firm.db.migrations import init_db
from firm.hitl.slack import build_app

# ---------------------------------------------------------------------------
# Constants — mirror test_hitl_slack.py defaults for consistency
# ---------------------------------------------------------------------------

_SLACK_SECRET = b"slack-secret-test"
_INTERNAL_SECRET = b"internal-secret-test"
_TS = "1700000000"
_TS_INT = 1700000000
_DECISION_ID = "dec-sig-invalid-t25"
_APPROVER_ID = "U123"

_CLOCK = ReplayClock(datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slack_sig(raw_body: str, ts: str, secret: bytes) -> str:
    return "v0=" + hmac.new(
        secret, f"v0:{ts}:{raw_body}".encode(), hashlib.sha256
    ).hexdigest()


def _make_form_body(slack_payload_json: str) -> str:
    return "payload=" + quote_plus(slack_payload_json)


def _make_slack_payload(sig: str) -> str:
    button_value = json.dumps(
        {
            "decision_id": _DECISION_ID,
            "approver_id": _APPROVER_ID,
            "ts": _TS_INT,
            "action": "approve",
            "sig": sig,
        }
    )
    return json.dumps(
        {
            "user": {"id": _APPROVER_ID},
            "actions": [{"value": button_value}],
        }
    )


def _seed_decision(db_path: Path) -> None:
    now_iso = _CLOCK.now().isoformat()
    with closing(get_conn(db_path)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO decisions "
            "(id, parent_chain, action, payload, rationale, confidence, "
            "citations, falsification, escalation, failure_mode, metadata, nonce, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                _DECISION_ID, "[]", "ESCALATE", "{}", "test-rationale", 0.9,
                "[]", "test-falsification", "test-escalation", None, "{}", "nonce-t25",
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
# Tests
# ---------------------------------------------------------------------------


def test_invalid_internal_signature_audit_logs_signed_approval_invalid(
    tmp_path: Path,
) -> None:
    """Tampered button HMAC → 401 + audit_log entry with SIGNED_APPROVAL_INVALID.

    The Slack outer signature is valid (we sign with the correct slack secret).
    Only the internal button sig is corrupted (truncated by one hex char, making
    it syntactically invalid for verify()).
    """
    db = tmp_path / "sig_invalid.db"
    init_db(db)
    _seed_decision(db)

    # Use a tampered internal sig — 63 hex chars instead of 64
    tampered_sig = "deadbeef" * 7 + "deadbee"  # 63 chars, malformed
    slack_payload_json = _make_slack_payload(sig=tampered_sig)
    raw_body = _make_form_body(slack_payload_json)

    headers = {
        "X-Slack-Signature": _slack_sig(raw_body, _TS, _SLACK_SECRET),
        "X-Slack-Request-Timestamp": _TS,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    client = _build_client(db)
    resp = client.post("/slack/interactive", content=raw_body, headers=headers)

    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.json()}"
    assert "invalid internal signature" in resp.json().get("error", "")

    # audit_log must have the signature-rejected entry
    with closing(get_conn(db)) as conn:
        rows = list(
            conn.execute(
                "SELECT detail FROM audit_log WHERE event='hitl.signature_rejected'"
            )
        )
    assert len(rows) == 1, f"Expected 1 hitl.signature_rejected audit entry, got {len(rows)}"
    detail = json.loads(rows[0]["detail"])
    assert detail.get("failure_mode") == FailureMode.SIGNED_APPROVAL_INVALID.value, (
        f"Expected failure_mode='signed_approval_invalid', got: {detail}"
    )
    assert detail.get("decision_id") == _DECISION_ID
    assert detail.get("approver_id") == _APPROVER_ID

    # hitl_queue must remain 'pending' (no DB mutation on rejection)
    with closing(get_conn(db)) as conn:
        row = conn.execute(
            "SELECT status FROM hitl_queue WHERE decision_id=?", (_DECISION_ID,)
        ).fetchone()
    assert row["status"] == "pending"


def test_valid_internal_signature_does_not_write_rejection_audit(
    tmp_path: Path,
) -> None:
    """A valid approval does NOT produce a hitl.signature_rejected entry."""
    from firm.hitl.signing import sign

    db = tmp_path / "sig_valid.db"
    init_db(db)
    _seed_decision(db)

    valid_sig = sign(
        decision_id=_DECISION_ID,
        approver_id=_APPROVER_ID,
        ts=_TS_INT,
        secret=_INTERNAL_SECRET,
    )
    slack_payload_json = _make_slack_payload(sig=valid_sig)
    raw_body = _make_form_body(slack_payload_json)

    headers = {
        "X-Slack-Signature": _slack_sig(raw_body, _TS, _SLACK_SECRET),
        "X-Slack-Request-Timestamp": _TS,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    client = _build_client(db)
    resp = client.post("/slack/interactive", content=raw_body, headers=headers)

    assert resp.status_code == 200
    with closing(get_conn(db)) as conn:
        count = conn.execute(
            "SELECT COUNT(*) AS n FROM audit_log WHERE event='hitl.signature_rejected'"
        ).fetchone()["n"]
    assert count == 0
