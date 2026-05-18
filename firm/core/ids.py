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
