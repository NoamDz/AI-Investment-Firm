"""Tests for firm.core.random — seeded RNG facade."""
import random

import pytest

from firm.core.random import get_rng


def test_default_seed_is_42(monkeypatch: pytest.MonkeyPatch) -> None:
    """With FIRM_RANDOM_SEED unset, get_rng() behaves like random.Random(42)."""
    monkeypatch.delenv("FIRM_RANDOM_SEED", raising=False)
    assert get_rng().random() == random.Random(42).random()


def test_seed_from_env_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    """When FIRM_RANDOM_SEED=1234, get_rng() behaves like random.Random(1234)."""
    monkeypatch.setenv("FIRM_RANDOM_SEED", "1234")
    assert get_rng().random() == random.Random(1234).random()


def test_invalid_seed_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-integer FIRM_RANDOM_SEED silently falls back to seed 42."""
    monkeypatch.setenv("FIRM_RANDOM_SEED", "not-a-number")
    assert get_rng().random() == random.Random(42).random()


def test_two_calls_produce_identical_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two fresh RNG instances from the same seed produce the identical 100-element sequence.

    This is the cross-process-determinism proxy: a fresh Random with the same
    seed is equivalent to a fresh process with the same FIRM_RANDOM_SEED.
    """
    monkeypatch.setenv("FIRM_RANDOM_SEED", "7")
    rng_a = get_rng()
    rng_b = get_rng()
    seq_a = [rng_a.random() for _ in range(100)]
    seq_b = [rng_b.random() for _ in range(100)]
    assert seq_a == seq_b


def test_shuffle_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two shuffles from the same seed produce the same permutation."""
    monkeypatch.setenv("FIRM_RANDOM_SEED", "42")
    items = list(range(20))
    items2 = list(range(20))
    get_rng().shuffle(items)
    get_rng().shuffle(items2)
    assert items == items2
    assert items != list(range(20))  # shuffle actually permuted
