"""HMAC-SHA256 signing and verification for HITL approval decisions.

Canonical message format
------------------------
    "{decision_id}|{approver_id}|{ts}"

The ``|`` separator was chosen deliberately to be distinct from the ``:``
separator used in ``firm.core.ids.sign_nonce``, so that a signature from one
context cannot be replayed into the other.

Tampering definitions
---------------------
Any of the following causes ``verify`` to return ``False``:

* HMAC mismatch (wrong secret, wrong decision_id, wrong approver_id, wrong ts).
* Expired timestamp: ``now - payload["ts"] > _MAX_AGE_SECONDS`` (replay defence).
* Future timestamp beyond clock-skew: ``payload["ts"] - now > _MAX_FUTURE_SKEW_SECONDS``.
* Missing or non-int ``ts``, missing ``decision_id`` or ``approver_id`` keys.
* Any other exception raised while extracting payload fields.

``verify`` is **total** — it never raises on adversarial input; it returns False.

Dual-key rotation (T13a)
------------------------
``verify_with_rotation`` extends verification to accept signatures produced by
either ``current_secret`` or a ``previous_secret`` that is still within its
configurable grace window (default 24 hours).  The runbook (T29) documents the
rotation procedure: set previous → set new → wait window → unset previous.

Rotation events are logged at INFO level so audit trails can trace which key
matched.  The existing ``verify`` function is unchanged.
"""
from __future__ import annotations

import hashlib
import hmac
import logging

_log = logging.getLogger(__name__)

# Maximum age of a valid signature.  Older signatures are rejected as potential
# replays even when the HMAC itself is valid.
_MAX_AGE_SECONDS: int = 300  # 5 minutes

# How far in the future we allow a timestamp to be, to tolerate clock skew
# between signers and verifiers running on different hosts.
_MAX_FUTURE_SKEW_SECONDS: int = 60  # 1 minute


def _canonical(decision_id: str, approver_id: str, ts: int) -> bytes:
    """Return the UTF-8 encoded canonical message for HMAC computation."""
    return f"{decision_id}|{approver_id}|{ts}".encode()


def sign(
    *,
    decision_id: str,
    approver_id: str,
    ts: int,
    secret: bytes,
) -> str:
    """Compute an HMAC-SHA256 hex-digest for an approval decision.

    Parameters
    ----------
    decision_id:
        Unique identifier for the decision being approved.  Must not contain
        ``|`` (used as canonical separator).
    approver_id:
        Identifier of the approving user.  Must not contain ``|``.
    ts:
        Unix timestamp (integer seconds) at the moment of signing.  Must be > 0.
    secret:
        HMAC key.  Must be non-empty.

    Returns
    -------
    str
        Lowercase hex HMAC-SHA256 digest (64 characters).

    Raises
    ------
    ValueError
        If ``secret`` is empty, ``ts <= 0``, ``decision_id`` is empty,
        ``approver_id`` is empty, or either ID field contains a ``|`` character.
    """
    if not secret:
        raise ValueError("secret must be non-empty")
    if ts <= 0:
        raise ValueError(f"ts must be a positive integer, got {ts!r}")
    if not decision_id:
        raise ValueError("decision_id must be non-empty")
    if not approver_id:
        raise ValueError("approver_id must be non-empty")
    if "|" in decision_id:
        raise ValueError(
            f"decision_id must not contain a pipe character ('|') to preserve "
            f"canonical separability; got {decision_id!r}"
        )
    if "|" in approver_id:
        raise ValueError(
            f"approver_id must not contain a pipe character ('|') to preserve "
            f"canonical separability; got {approver_id!r}"
        )
    msg = _canonical(decision_id, approver_id, ts)
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def verify(
    *,
    payload: dict[str, object],
    signature: str,
    secret: bytes,
    now: int,
) -> bool:
    """Verify an HMAC-SHA256 approval signature in constant time.

    This function is **total**: it returns ``False`` rather than raising on any
    form of adversarial or malformed input (missing keys, wrong types, etc.).

    Parameters
    ----------
    payload:
        Dict containing ``"decision_id"`` (str), ``"approver_id"`` (str), and
        ``"ts"`` (int).
    signature:
        Hex-encoded HMAC digest to verify.
    secret:
        HMAC key used during signing.
    now:
        Current Unix timestamp (integer seconds).  Passed explicitly so the
        caller (and T13a's dual-key wrapper) controls the clock source.

    Returns
    -------
    bool
        ``True`` only when the HMAC matches **and** the timestamp is within
        ``[now - _MAX_AGE_SECONDS, now + _MAX_FUTURE_SKEW_SECONDS]``.
    """
    try:
        decision_id = payload["decision_id"]
        approver_id = payload["approver_id"]
        ts = payload["ts"]

        # Enforce type safety before arithmetic to avoid silent mis-comparison.
        if not isinstance(ts, int) or isinstance(ts, bool):
            return False

        # Replay-window check: reject expired and implausibly future timestamps.
        age = now - ts
        if age > _MAX_AGE_SECONDS:
            return False
        skew = ts - now
        if skew > _MAX_FUTURE_SKEW_SECONDS:
            return False

        # Recompute expected HMAC over the canonical message.
        expected = hmac.new(secret, _canonical(decision_id, approver_id, ts), hashlib.sha256).hexdigest()  # type: ignore[arg-type]
        return hmac.compare_digest(expected, signature)

    except Exception:  # noqa: BLE001 — intentional total-function contract
        return False


# ---------------------------------------------------------------------------
# T13a: Dual-key rotation
# ---------------------------------------------------------------------------

_DEFAULT_GRACE_WINDOW_SECONDS: int = 86400  # 24 hours


def verify_with_rotation(
    *,
    payload: dict[str, object],
    signature: str,
    current_secret: bytes,
    previous_secret: bytes | None = None,
    rotated_at: int | None = None,
    now: int,
    grace_window_seconds: int = _DEFAULT_GRACE_WINDOW_SECONDS,
) -> bool:
    """Verify a signature against ``current_secret``, falling back to
    ``previous_secret`` if rotation is still within the grace window.

    Returns ``True`` iff:

    * signature verifies under ``current_secret``, **or**
    * signature verifies under ``previous_secret`` **and**
      ``previous_secret`` is not ``None`` **and**
      ``rotated_at`` is not ``None`` **and**
      ``now - rotated_at <= grace_window_seconds``.

    The existing replay-window check inside :func:`verify` is applied for every
    key tried, so an expired ``payload["ts"]`` is still rejected even when the
    HMAC matches.

    Parameters
    ----------
    payload:
        Dict containing ``"decision_id"`` (str), ``"approver_id"`` (str), and
        ``"ts"`` (int).
    signature:
        Hex-encoded HMAC digest to verify.
    current_secret:
        The active HMAC key.
    previous_secret:
        The key that was active before the most recent rotation.  ``None``
        disables fallback entirely.
    rotated_at:
        Unix timestamp (integer seconds) when the rotation occurred.  ``None``
        disables fallback even if ``previous_secret`` is set.
    now:
        Current Unix timestamp (integer seconds).
    grace_window_seconds:
        How long (in seconds) after ``rotated_at`` the previous key is still
        accepted.  Defaults to 86400 (24 hours).

    Returns
    -------
    bool
        ``True`` only when at least one key produces a valid HMAC **and** the
        timestamp is within its replay window.
    """
    # Always try the current key first.
    if verify(payload=payload, signature=signature, secret=current_secret, now=now):
        return True

    # Fallback to previous key only when rotation metadata is complete and the
    # grace window has not expired.
    if (
        previous_secret is not None
        and rotated_at is not None
        and (now - rotated_at) <= grace_window_seconds
        and verify(payload=payload, signature=signature, secret=previous_secret, now=now)
    ):
        _log.info(
            "Slack signature verified via previous key (rotation grace window active); "
            "rotated_at=%d now=%d age=%ds window=%ds",
            rotated_at,
            now,
            now - rotated_at,
            grace_window_seconds,
        )
        return True

    return False
