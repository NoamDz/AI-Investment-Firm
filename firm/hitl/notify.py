"""Slack notifier for HITL ESCALATE decisions.

Posts an approval-prompt message with signed Approve/Reject buttons to a
configured Slack channel when a Decision reaches the ESCALATE node.

Limitation (MVP): one `slack_approver_id` is specified in policy; multi-approver
allowlists require extending the T11 verifier and are out of scope for T12.
"""
from __future__ import annotations

import json
from typing import Protocol

from firm.core.clock import Clock
from firm.core.models import BuyPayload, Decision, EscalatePayload, SellPayload
from firm.hitl.signing import sign

_RATIONALE_MAX_CHARS: int = 500


class _WebClientProtocol(Protocol):
    """Minimal interface satisfied by slack_sdk.WebClient (and MagicMock)."""

    def chat_postMessage(self, **kwargs: object) -> object: ...  # noqa: N802


class SlackNotifier:
    """Posts ESCALATE-decision approval prompts to Slack with signed buttons."""

    def __init__(
        self,
        *,
        web_client: object,
        channel: str,
        approver_id: str,
        clock: Clock,
        internal_secret: bytes,
    ) -> None:
        self._client: _WebClientProtocol = web_client  # type: ignore[assignment]
        self._channel = channel
        self._approver_id = approver_id
        self._clock = clock
        self._secret = internal_secret

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify(self, *, decision: Decision) -> None:
        """Post one chat.postMessage call with Approve + Reject buttons.

        Exceptions from the Slack SDK propagate to the caller.  The orchestrator
        (``firm.agents.hitl.make_hitl``) is the single owner of audit-on-failure:
        it catches the exception, writes a ``hitl.slack_notify_failed`` audit row,
        and continues so a Slack outage never blocks the HITL queue.
        """
        self._post(decision)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post(self, decision: Decision) -> None:
        ts = int(self._clock.now().timestamp())
        sig = sign(
            decision_id=decision.id,
            approver_id=self._approver_id,
            ts=ts,
            secret=self._secret,
        )

        approve_value = json.dumps({
            "decision_id": decision.id,
            "approver_id": self._approver_id,
            "ts": ts,
            "action": "approve",
            "sig": sig,
        })
        reject_value = json.dumps({
            "decision_id": decision.id,
            "approver_id": self._approver_id,
            "ts": ts,
            "action": "reject",
            "sig": sig,
        })

        summary = self._build_summary(decision)

        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": summary},
            },
            {
                "type": "actions",
                "block_id": "hitl-buttons",
                "elements": [
                    {
                        "type": "button",
                        "action_id": "approve",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "value": approve_value,
                    },
                    {
                        "type": "button",
                        "action_id": "reject",
                        "text": {"type": "plain_text", "text": "Reject"},
                        "style": "danger",
                        "value": reject_value,
                    },
                ],
            },
        ]

        self._client.chat_postMessage(
            channel=self._channel,
            text=f"HITL: Decision {decision.id} needs review",
            blocks=blocks,
        )

    def _build_summary(self, decision: Decision) -> str:
        """Build the mrkdwn summary text from a Decision."""
        payload = decision.payload

        # Extract ticker, qty, reason from typed payload
        if isinstance(payload, EscalatePayload):
            inner = payload.proposed
            ticker = inner.ticker if isinstance(inner, (BuyPayload, SellPayload)) else "?"
            qty = str(inner.shares) if isinstance(inner, (BuyPayload, SellPayload)) else "?"
            reason = payload.reason
        elif isinstance(payload, (BuyPayload, SellPayload)):
            ticker = payload.ticker
            qty = str(payload.shares)
            reason = decision.escalation_reason or ""
        else:
            ticker = "?"
            qty = "?"
            reason = getattr(payload, "reason", "")

        rationale = decision.rationale[:_RATIONALE_MAX_CHARS]

        return (
            f"*HITL escalation:* `{decision.id}`\n"
            f"• Action: `{decision.action}` on `{ticker}` (qty `{qty}`)\n"
            f"• Reason: {reason}\n"
            f"• Rationale: {rationale}"
        )
