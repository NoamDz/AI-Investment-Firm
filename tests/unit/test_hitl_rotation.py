"""Tests for T13a: dual-key Slack secret rotation in firm.hitl.signing.

Ten test cases covering:
- Current key matches (no fallback needed)
- Previous key matches within and outside the grace window
- Configurable grace window
- Tampered signatures
- Missing previous_secret or rotated_at disables fallback
- Replay window still enforced by the underlying verify()
"""
from __future__ import annotations

import logging

import pytest

from firm.hitl.signing import sign, verify_with_rotation

# A fixed "now" for deterministic replay-window tests.
_NOW = 1_700_000_000

# Two distinct secrets — neither contains the other's bytes.
_CURRENT_SECRET = b"current-secret-xxxxxxxxxxxxxxxx"
_PREVIOUS_SECRET = b"previous-secret-yyyyyyyyyyyyyyyy"

# A valid payload whose timestamp is fresh relative to _NOW.
_PAYLOAD: dict[str, object] = {
    "decision_id": "dec-001",
    "approver_id": "alice",
    "ts": _NOW,
}


def _sig(secret: bytes) -> str:
    """Produce a valid signature for _PAYLOAD under *secret*."""
    return sign(
        decision_id=str(_PAYLOAD["decision_id"]),
        approver_id=str(_PAYLOAD["approver_id"]),
        ts=int(_PAYLOAD["ts"]),  # type: ignore[arg-type]
        secret=secret,
    )


# ---------------------------------------------------------------------------
# 1. Current key matches — no previous key provided.
# ---------------------------------------------------------------------------


def test_current_key_matches_without_previous(caplog: pytest.LogCaptureFixture) -> None:
    """verify_with_rotation returns True when sig matches current_secret (no previous)."""
    sig = _sig(_CURRENT_SECRET)
    with caplog.at_level(logging.INFO, logger="firm.hitl.signing"):
        result = verify_with_rotation(
            payload=_PAYLOAD,
            signature=sig,
            current_secret=_CURRENT_SECRET,
            now=_NOW,
        )
    assert result is True
    # No fallback log should be emitted for a current-key match.
    assert not any("previous" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# 2. Current key takes precedence even when both keys are set.
# ---------------------------------------------------------------------------


def test_current_key_match_takes_precedence_over_previous(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When sig matches current_secret, return True without emitting a fallback log."""
    sig = _sig(_CURRENT_SECRET)
    rotated_at = _NOW - 3600  # 1h ago — well within 24h window
    with caplog.at_level(logging.INFO, logger="firm.hitl.signing"):
        result = verify_with_rotation(
            payload=_PAYLOAD,
            signature=sig,
            current_secret=_CURRENT_SECRET,
            previous_secret=_PREVIOUS_SECRET,
            rotated_at=rotated_at,
            now=_NOW,
        )
    assert result is True
    assert not any("previous" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# 3. Previous key match within the grace window → accepted + log.
# ---------------------------------------------------------------------------


def test_previous_key_match_within_window(caplog: pytest.LogCaptureFixture) -> None:
    """Sig matching previous key within the 24h grace window → True + INFO log."""
    sig = _sig(_PREVIOUS_SECRET)
    rotated_at = _NOW - 3600  # 1h ago
    with caplog.at_level(logging.INFO, logger="firm.hitl.signing"):
        result = verify_with_rotation(
            payload=_PAYLOAD,
            signature=sig,
            current_secret=_CURRENT_SECRET,
            previous_secret=_PREVIOUS_SECRET,
            rotated_at=rotated_at,
            now=_NOW,
        )
    assert result is True
    rotation_logs = [r for r in caplog.records if r.levelno == logging.INFO]
    assert rotation_logs, "Expected at least one INFO log for previous-key fallback"


# ---------------------------------------------------------------------------
# 4. Previous key match OUTSIDE the grace window → rejected.
# ---------------------------------------------------------------------------


def test_previous_key_match_outside_window() -> None:
    """Sig matching previous key past the 24h grace window → False."""
    sig = _sig(_PREVIOUS_SECRET)
    rotated_at = _NOW - 90_000  # ~25h ago, past the 24h default
    result = verify_with_rotation(
        payload=_PAYLOAD,
        signature=sig,
        current_secret=_CURRENT_SECRET,
        previous_secret=_PREVIOUS_SECRET,
        rotated_at=rotated_at,
        now=_NOW,
    )
    assert result is False


# ---------------------------------------------------------------------------
# 5. Explicit grace_window_seconds is honoured.
# ---------------------------------------------------------------------------


def test_previous_key_match_with_explicit_grace_window() -> None:
    """grace_window_seconds parameter narrows or widens the acceptance window."""
    sig = _sig(_PREVIOUS_SECRET)
    rotated_at = _NOW - 1200  # 20 minutes ago

    # 10-minute window → rejected (20m > 10m).
    result_narrow = verify_with_rotation(
        payload=_PAYLOAD,
        signature=sig,
        current_secret=_CURRENT_SECRET,
        previous_secret=_PREVIOUS_SECRET,
        rotated_at=rotated_at,
        now=_NOW,
        grace_window_seconds=600,
    )
    assert result_narrow is False

    # 30-minute window → accepted (20m < 30m).
    result_wide = verify_with_rotation(
        payload=_PAYLOAD,
        signature=sig,
        current_secret=_CURRENT_SECRET,
        previous_secret=_PREVIOUS_SECRET,
        rotated_at=rotated_at,
        now=_NOW,
        grace_window_seconds=1800,
    )
    assert result_wide is True


# ---------------------------------------------------------------------------
# 6. Tampered signature rejected under both keys.
# ---------------------------------------------------------------------------


def test_tampered_sig_rejected_under_both_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A signature that doesn't match either key → False; no fallback log."""
    tampered_sig = "0" * 64  # all-zero hex — won't match any real HMAC
    rotated_at = _NOW - 3600
    with caplog.at_level(logging.INFO, logger="firm.hitl.signing"):
        result = verify_with_rotation(
            payload=_PAYLOAD,
            signature=tampered_sig,
            current_secret=_CURRENT_SECRET,
            previous_secret=_PREVIOUS_SECRET,
            rotated_at=rotated_at,
            now=_NOW,
        )
    assert result is False
    assert not caplog.records


# ---------------------------------------------------------------------------
# 7. No previous_secret means current-key-only check.
# ---------------------------------------------------------------------------


def test_no_previous_secret_means_current_only() -> None:
    """With previous_secret=None, a sig valid only under previous key → False."""
    sig = _sig(_PREVIOUS_SECRET)
    result = verify_with_rotation(
        payload=_PAYLOAD,
        signature=sig,
        current_secret=_CURRENT_SECRET,
        previous_secret=None,
        rotated_at=None,
        now=_NOW,
    )
    assert result is False


# ---------------------------------------------------------------------------
# 8. rotated_at=None disables fallback even if previous_secret is set.
# ---------------------------------------------------------------------------


def test_rotated_at_none_disables_fallback_even_if_previous_set() -> None:
    """Without a rotation timestamp, fallback is disabled regardless of previous_secret."""
    sig = _sig(_PREVIOUS_SECRET)
    result = verify_with_rotation(
        payload=_PAYLOAD,
        signature=sig,
        current_secret=_CURRENT_SECRET,
        previous_secret=_PREVIOUS_SECRET,
        rotated_at=None,  # no timestamp → no fallback
        now=_NOW,
    )
    assert result is False


# ---------------------------------------------------------------------------
# 9. Replay window still enforced when sig matches current key.
# ---------------------------------------------------------------------------


def test_replay_window_still_enforced_on_current_match() -> None:
    """An expired payload (ts 400s in the past) is rejected even if HMAC is valid."""
    old_ts = _NOW - 400  # outside T10's 300s replay window
    old_payload: dict[str, object] = {
        "decision_id": "dec-001",
        "approver_id": "alice",
        "ts": old_ts,
    }
    sig = sign(
        decision_id="dec-001",
        approver_id="alice",
        ts=old_ts,
        secret=_CURRENT_SECRET,
    )
    result = verify_with_rotation(
        payload=old_payload,
        signature=sig,
        current_secret=_CURRENT_SECRET,
        now=_NOW,
    )
    assert result is False


# ---------------------------------------------------------------------------
# 10. Replay window still enforced when sig matches previous key.
# ---------------------------------------------------------------------------


def test_replay_window_still_enforced_on_previous_match() -> None:
    """An expired payload is rejected even if HMAC matches the previous key."""
    old_ts = _NOW - 400  # outside the 300s replay window
    old_payload: dict[str, object] = {
        "decision_id": "dec-001",
        "approver_id": "alice",
        "ts": old_ts,
    }
    sig = sign(
        decision_id="dec-001",
        approver_id="alice",
        ts=old_ts,
        secret=_PREVIOUS_SECRET,
    )
    rotated_at = _NOW - 3600  # within the 24h grace window
    result = verify_with_rotation(
        payload=old_payload,
        signature=sig,
        current_secret=_CURRENT_SECRET,
        previous_secret=_PREVIOUS_SECRET,
        rotated_at=rotated_at,
        now=_NOW,
    )
    assert result is False
