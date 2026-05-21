"""FastAPI application for Slack interactive callbacks (HITL approval/rejection).

Dual-layer verification
-----------------------
Every inbound request is checked at two levels:

1. **Slack outer HMAC** (``X-Slack-Signature`` header):  proves the request
   originated from Slack's servers and has not been tampered with in transit.
   Uses Slack's v0 signing scheme: ``hmac_sha256(slack_signing_secret,
   f"v0:{X-Slack-Request-Timestamp}:{raw_body}")``.

2. **Internal button-payload HMAC** (the ``sig`` field inside the button
   ``value`` JSON):  proves the button payload was constructed by *our*
   notifier (T12) and has not been tampered with between notification time and
   the user's click.  Uses ``firm.hitl.signing.sign/verify`` over
   ``f"{decision_id}|{approver_id}|{ts}"``.

Why both?  The Slack-side signature guarantees Slack delivered the request;
the internal signature guarantees *we* originally created the button content.
An attacker who compromises Slack's channel cannot forge our internal
signature, and an attacker who intercepts our internal notifier cannot spoof
the Slack outer signature.

T12 forward-reference
---------------------
T12 (``firm/hitl/notify.py``) will build Slack Block Kit messages whose
button ``value`` fields carry the ``decision_id``, ``approver_id``, ``ts``,
``action``, and ``sig`` produced by ``firm.hitl.signing.sign``.  This module
consumes exactly that shape.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from firm.agents.hitl import mark_approved, mark_rejected
from firm.core.clock import Clock
from firm.hitl.signing import verify as verify_internal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SLACK_REPLAY_WINDOW_SECONDS: int = 300  # Slack's recommended max age
_SLACK_SIG_HEADER = "X-Slack-Signature"
_SLACK_TS_HEADER = "X-Slack-Request-Timestamp"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_app(
    *,
    db_path: Path,
    clock: Clock,
    slack_signing_secret: bytes,
    internal_secret: bytes,
) -> FastAPI:
    """Construct and return the FastAPI application.

    Parameters
    ----------
    db_path:
        SQLite database path.  Passed through to ``mark_approved`` /
        ``mark_rejected``.
    clock:
        Clock instance used for replay-window checks and audit timestamps.
    slack_signing_secret:
        Slack app signing secret (raw bytes).  Used to verify
        ``X-Slack-Signature``.
    internal_secret:
        Our internal HMAC key.  Used to verify the embedded button ``sig``.
    """
    app = FastAPI(title="HITL Slack Interactive", docs_url=None, redoc_url=None)

    @app.post("/slack/interactive")
    async def slack_interactive(request: Request) -> Response:  # noqa: ANN001
        # ------------------------------------------------------------------
        # Step 0: read raw body (needed for Slack signature verification).
        # ------------------------------------------------------------------
        raw_body_bytes: bytes = await request.body()
        raw_body_str: str = raw_body_bytes.decode("utf-8")

        # ------------------------------------------------------------------
        # Step 1: verify Slack outer signing.
        # ------------------------------------------------------------------
        slack_sig = request.headers.get(_SLACK_SIG_HEADER, "")
        slack_ts_str = request.headers.get(_SLACK_TS_HEADER, "")

        if not slack_ts_str:
            return JSONResponse(
                status_code=401,
                content={"error": "missing X-Slack-Request-Timestamp"},
            )

        try:
            slack_ts = int(slack_ts_str)
        except ValueError:
            return JSONResponse(
                status_code=401,
                content={"error": "invalid X-Slack-Request-Timestamp"},
            )

        now_ts = int(clock.now().timestamp())
        if abs(now_ts - slack_ts) > _SLACK_REPLAY_WINDOW_SECONDS:
            return JSONResponse(
                status_code=401,
                content={"error": "stale X-Slack-Request-Timestamp"},
            )

        expected_sig = (
            "v0="
            + hmac.new(
                slack_signing_secret,
                f"v0:{slack_ts_str}:{raw_body_str}".encode(),
                hashlib.sha256,
            ).hexdigest()
        )
        if not hmac.compare_digest(expected_sig, slack_sig):
            return JSONResponse(
                status_code=401,
                content={"error": "invalid Slack signature"},
            )

        # ------------------------------------------------------------------
        # Step 2: parse form body, extract payload JSON.
        # ------------------------------------------------------------------
        from urllib.parse import parse_qs  # local import to keep top-level clean

        parsed = parse_qs(raw_body_str)
        if "payload" not in parsed:
            return JSONResponse(
                status_code=400,
                content={"error": "missing payload field"},
            )

        try:
            slack_payload = json.loads(parsed["payload"][0])
        except (json.JSONDecodeError, IndexError) as exc:
            logger.debug("Failed to parse Slack payload JSON: %s", exc)
            return JSONResponse(
                status_code=400,
                content={"error": "malformed payload JSON"},
            )

        # ------------------------------------------------------------------
        # Step 3: extract button value from actions[0].value.
        # ------------------------------------------------------------------
        try:
            button_value_raw = slack_payload["actions"][0]["value"]
            button = json.loads(button_value_raw)
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            logger.debug("Failed to extract button value: %s", exc)
            return JSONResponse(
                status_code=400,
                content={"error": "malformed button value"},
            )

        # ------------------------------------------------------------------
        # Step 4: verify payload.user.id == button["approver_id"].
        # ------------------------------------------------------------------
        try:
            slack_user_id: str = slack_payload["user"]["id"]
        except (KeyError, TypeError):
            return JSONResponse(
                status_code=400,
                content={"error": "missing user.id in Slack payload"},
            )

        if slack_user_id != button.get("approver_id"):
            return JSONResponse(
                status_code=401,
                content={"error": "approver_id mismatch"},
            )

        # ------------------------------------------------------------------
        # Step 5: verify internal HMAC.
        # ------------------------------------------------------------------
        ok = verify_internal(
            payload={
                "decision_id": button.get("decision_id"),
                "approver_id": button.get("approver_id"),
                "ts": button.get("ts"),
            },
            signature=button.get("sig", ""),
            secret=internal_secret,
            now=now_ts,
        )
        if not ok:
            return JSONResponse(
                status_code=401,
                content={"error": "invalid internal signature"},
            )

        # ------------------------------------------------------------------
        # Step 6: dispatch based on action.
        # ------------------------------------------------------------------
        decision_id: str = button["decision_id"]
        approver_id: str = button["approver_id"]
        action: str = button.get("action", "")

        if action == "approve":
            mark_approved(
                db_path=db_path,
                decision_id=decision_id,
                approver=approver_id,
                clock=clock,
            )
            past_tense = "approved"
        elif action == "reject":
            mark_rejected(
                db_path=db_path,
                decision_id=decision_id,
                approver=approver_id,
                clock=clock,
            )
            past_tense = "rejected"
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"unknown action: {action!r}"},
            )

        # ------------------------------------------------------------------
        # Step 7: return ephemeral Slack update.
        # ------------------------------------------------------------------
        return JSONResponse(
            status_code=200,
            content={
                "response_type": "ephemeral",
                "replace_original": True,
                "text": f"Decision {decision_id} {past_tense} by {approver_id}.",
            },
        )

    return app
