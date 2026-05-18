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
