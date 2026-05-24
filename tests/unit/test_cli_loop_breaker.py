"""Unit tests for the consecutive-identical-failure circuit breaker.

Regression for the silent-heartbeat-swallow bug where the loop wrapped
``_do_heartbeat`` in ``try/except Exception`` with no failure tracking — a
qdrant DNS error produced 76 silent failures over 7 minutes before SIGTERM
without surfacing the underlying problem in CI logs.
"""
from __future__ import annotations

import pytest

from firm.cli import _run_heartbeat_loop


def _stop_after_iterations(max_iters: int):
    """Return a should_stop callable that flips True after ``max_iters`` heartbeat attempts.

    Tied to the test's ``do_heartbeat`` fake via a shared counter dict so the
    "should stop" decision depends on iteration count, not on how many times
    the loop body calls ``should_stop`` per iteration (3 calls per iter today).
    """
    counter = {"iters": 0}

    def should_stop() -> bool:
        return counter["iters"] >= max_iters

    return should_stop, counter


def test_loop_reraises_after_threshold_identical_failures() -> None:
    """3 consecutive identical exceptions trip the breaker and re-raise."""
    attempts: list[int] = []

    def always_fails(seq: int) -> None:
        attempts.append(seq)
        raise RuntimeError("qdrant DNS down")

    with pytest.raises(RuntimeError, match="qdrant DNS down"):
        _run_heartbeat_loop(
            always_fails,
            interval_seconds=0,
            should_stop=lambda: False,
            sleep=lambda _s: None,
            echo=lambda *a, **kw: None,
            failure_threshold=3,
        )

    assert attempts == [1, 2, 3]


def test_loop_does_not_trip_on_different_failure_signatures() -> None:
    """Distinct exception messages reset the counter — only *identical* re-raise."""
    seq_to_exc = {
        1: RuntimeError("dns A"),
        2: RuntimeError("dns B"),
        3: RuntimeError("dns A"),
        4: ValueError("dns A"),
    }
    attempts: list[int] = []
    should_stop, counter = _stop_after_iterations(4)

    def cycle_failures(seq: int) -> None:
        attempts.append(seq)
        counter["iters"] += 1
        raise seq_to_exc[seq]

    seq = _run_heartbeat_loop(
        cycle_failures,
        interval_seconds=0,
        should_stop=should_stop,
        sleep=lambda _s: None,
        echo=lambda *a, **kw: None,
        failure_threshold=3,
    )
    assert seq == 4
    assert attempts == [1, 2, 3, 4]


def test_loop_resets_counter_on_success() -> None:
    """A successful heartbeat between failures resets the consecutive counter."""
    attempts: list[int] = []
    should_stop, counter = _stop_after_iterations(5)

    def fail_succeed_fail(seq: int) -> None:
        attempts.append(seq)
        counter["iters"] += 1
        if seq == 3:
            return
        raise RuntimeError("transient")

    seq = _run_heartbeat_loop(
        fail_succeed_fail,
        interval_seconds=0,
        should_stop=should_stop,
        sleep=lambda _s: None,
        echo=lambda *a, **kw: None,
        failure_threshold=3,
    )
    # 5 attempts: fail, fail, success (resets), fail, fail (counter=2 < 3) → no raise.
    assert seq == 5
    assert attempts == [1, 2, 3, 4, 5]


def test_loop_logs_failure_count_with_threshold() -> None:
    """Operator-facing message must show ``N/THRESHOLD`` and the exc signature."""
    messages: list[str] = []

    def always_fails(_seq: int) -> None:
        raise RuntimeError("qdrant DNS down")

    def capture(msg: str, **_kw: object) -> None:
        messages.append(msg)

    with pytest.raises(RuntimeError):
        _run_heartbeat_loop(
            always_fails,
            interval_seconds=0,
            should_stop=lambda: False,
            sleep=lambda _s: None,
            echo=capture,
            failure_threshold=3,
        )

    failure_lines = [m for m in messages if "heartbeat" in m and "failed" in m]
    assert len(failure_lines) == 3
    assert "1/3" in failure_lines[0]
    assert "2/3" in failure_lines[1]
    assert "3/3" in failure_lines[2]
    for line in failure_lines:
        assert "RuntimeError" in line
        assert "qdrant DNS down" in line
    assert any("aborting loop" in m for m in messages)


def test_loop_exits_cleanly_when_should_stop_flips() -> None:
    """A clean stop via should_stop returns the seq count without raising."""
    calls: list[int] = []
    should_stop, counter = _stop_after_iterations(2)

    def succeeds(seq: int) -> None:
        calls.append(seq)
        counter["iters"] += 1

    seq = _run_heartbeat_loop(
        succeeds,
        interval_seconds=0,
        should_stop=should_stop,
        sleep=lambda _s: None,
        echo=lambda *a, **kw: None,
    )
    assert seq == 2
    assert calls == [1, 2]
