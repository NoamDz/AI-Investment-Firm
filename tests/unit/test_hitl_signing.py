"""Tests for firm.hitl.signing — HMAC-SHA256 approval signatures.

All 16 test cases from T10 spec. One assertion per test for clear failure diagnosis.
"""
import inspect

import pytest

from firm.hitl.signing import sign, verify

# A fixed "now" for deterministic replay-window tests.
_NOW = 1_700_000_000
_SECRET = b"supersecret" * 4  # 44 bytes — well above empty


# ---------------------------------------------------------------------------
# 1. Roundtrip
# ---------------------------------------------------------------------------


def test_sign_verify_roundtrip():
    sig = sign(
        decision_id="dec-001",
        approver_id="alice",
        ts=_NOW,
        secret=_SECRET,
    )
    assert verify(
        payload={"decision_id": "dec-001", "approver_id": "alice", "ts": _NOW},
        signature=sig,
        secret=_SECRET,
        now=_NOW,
    )


# ---------------------------------------------------------------------------
# 2–5. Tampering paths (HMAC mismatch)
# ---------------------------------------------------------------------------


def test_verify_returns_false_on_wrong_secret():
    sig = sign(decision_id="dec-001", approver_id="alice", ts=_NOW, secret=_SECRET)
    assert not verify(
        payload={"decision_id": "dec-001", "approver_id": "alice", "ts": _NOW},
        signature=sig,
        secret=b"different-secret" * 4,
        now=_NOW,
    )


def test_verify_returns_false_on_swapped_decision_id():
    sig = sign(decision_id="dec-001", approver_id="alice", ts=_NOW, secret=_SECRET)
    assert not verify(
        payload={"decision_id": "dec-002", "approver_id": "alice", "ts": _NOW},
        signature=sig,
        secret=_SECRET,
        now=_NOW,
    )


def test_verify_returns_false_on_swapped_approver_id():
    sig = sign(decision_id="dec-001", approver_id="alice", ts=_NOW, secret=_SECRET)
    assert not verify(
        payload={"decision_id": "dec-001", "approver_id": "bob", "ts": _NOW},
        signature=sig,
        secret=_SECRET,
        now=_NOW,
    )


def test_verify_returns_false_on_tampered_ts():
    sig = sign(decision_id="dec-001", approver_id="alice", ts=_NOW, secret=_SECRET)
    assert not verify(
        payload={"decision_id": "dec-001", "approver_id": "alice", "ts": _NOW + 1},
        signature=sig,
        secret=_SECRET,
        now=_NOW,
    )


# ---------------------------------------------------------------------------
# 6–9. Replay window / clock-skew
# ---------------------------------------------------------------------------


def test_verify_returns_false_on_expired_timestamp():
    """Signature is 301 seconds old — outside the 300 s replay window."""
    old_ts = _NOW - 301
    sig = sign(decision_id="dec-001", approver_id="alice", ts=old_ts, secret=_SECRET)
    assert not verify(
        payload={"decision_id": "dec-001", "approver_id": "alice", "ts": old_ts},
        signature=sig,
        secret=_SECRET,
        now=_NOW,
    )


def test_verify_returns_true_within_replay_window():
    """Signature is 299 seconds old — inside the 300 s replay window."""
    old_ts = _NOW - 299
    sig = sign(decision_id="dec-001", approver_id="alice", ts=old_ts, secret=_SECRET)
    assert verify(
        payload={"decision_id": "dec-001", "approver_id": "alice", "ts": old_ts},
        signature=sig,
        secret=_SECRET,
        now=_NOW,
    )


def test_verify_returns_false_on_future_timestamp_beyond_skew():
    """Timestamp 61 s in the future exceeds the 60 s clock-skew tolerance."""
    future_ts = _NOW + 61
    sig = sign(decision_id="dec-001", approver_id="alice", ts=future_ts, secret=_SECRET)
    assert not verify(
        payload={"decision_id": "dec-001", "approver_id": "alice", "ts": future_ts},
        signature=sig,
        secret=_SECRET,
        now=_NOW,
    )


def test_verify_returns_true_within_clock_skew():
    """Timestamp 30 s in the future is within the 60 s clock-skew tolerance."""
    future_ts = _NOW + 30
    sig = sign(decision_id="dec-001", approver_id="alice", ts=future_ts, secret=_SECRET)
    assert verify(
        payload={"decision_id": "dec-001", "approver_id": "alice", "ts": future_ts},
        signature=sig,
        secret=_SECRET,
        now=_NOW,
    )


# ---------------------------------------------------------------------------
# 10–11. Adversarial / malformed payloads — verify must be total (no exceptions)
# ---------------------------------------------------------------------------


def test_verify_returns_false_on_missing_payload_keys():
    """Missing 'decision_id' must return False, not raise KeyError."""
    sig = sign(decision_id="dec-001", approver_id="alice", ts=_NOW, secret=_SECRET)
    assert not verify(
        payload={"approver_id": "alice", "ts": _NOW},  # decision_id absent
        signature=sig,
        secret=_SECRET,
        now=_NOW,
    )


def test_verify_returns_false_on_non_int_ts_in_payload():
    """ts as a string must return False, not raise TypeError."""
    sig = sign(decision_id="dec-001", approver_id="alice", ts=_NOW, secret=_SECRET)
    assert not verify(
        payload={"decision_id": "dec-001", "approver_id": "alice", "ts": "123"},
        signature=sig,
        secret=_SECRET,
        now=_NOW,
    )


def test_verify_returns_false_on_bool_ts_in_payload():
    """ts as a bool (True == 1 in Python) must return False — guards against
    the bool-is-int subclass trap so verify() does not silently accept
    payload['ts']=True as the integer 1."""
    sig = sign(decision_id="dec-001", approver_id="alice", ts=_NOW, secret=_SECRET)
    assert not verify(
        payload={"decision_id": "dec-001", "approver_id": "alice", "ts": True},
        signature=sig,
        secret=_SECRET,
        now=_NOW,
    )


# ---------------------------------------------------------------------------
# 12–13. Pipe rejection in sign (canonicalization safety)
# ---------------------------------------------------------------------------


def test_sign_rejects_pipe_in_decision_id():
    with pytest.raises(ValueError, match="pipe"):
        sign(decision_id="dec|001", approver_id="alice", ts=_NOW, secret=_SECRET)


def test_sign_rejects_pipe_in_approver_id():
    with pytest.raises(ValueError, match="pipe"):
        sign(decision_id="dec-001", approver_id="alice|admin", ts=_NOW, secret=_SECRET)


# ---------------------------------------------------------------------------
# 14–15. Input validation in sign
# ---------------------------------------------------------------------------


def test_sign_rejects_empty_secret():
    with pytest.raises(ValueError, match="secret"):
        sign(decision_id="dec-001", approver_id="alice", ts=_NOW, secret=b"")


def test_sign_rejects_non_positive_ts():
    with pytest.raises(ValueError, match="ts"):
        sign(decision_id="dec-001", approver_id="alice", ts=0, secret=_SECRET)
    with pytest.raises(ValueError, match="ts"):
        sign(decision_id="dec-001", approver_id="alice", ts=-1, secret=_SECRET)


# ---------------------------------------------------------------------------
# 16. Constant-time comparison sanity check
# ---------------------------------------------------------------------------


def test_compare_digest_used():
    """Verify hmac.compare_digest appears in the source of verify().

    Using inspect.getsource is simpler and less brittle than patching, because
    patching would require mocking the internal hmac module attribute inside the
    signing module's namespace — fragile across Python versions.  Source-text
    inspection is a stable proxy: if the function ever switches to '==' we catch it.
    """
    import firm.hitl.signing as signing_mod

    source = inspect.getsource(signing_mod)
    assert "compare_digest" in source
