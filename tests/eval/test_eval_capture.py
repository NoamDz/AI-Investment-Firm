"""Tests for ``scripts/eval_capture.py`` (Plan 4 T16).

The script can't be exercised end-to-end without real API access, so these
tests cover what's testable: argument plumbing, dry-run plan emission,
preflight checks, and subprocess dispatch. Every test patches
``subprocess.run`` so NO real subprocess is ever spawned.
"""
from __future__ import annotations

from unittest import mock

import pytest
from click.testing import CliRunner

from scripts import eval_capture


# ---------------------------------------------------------------------------
# Test 1 — --dry-run prints the plan without invoking subprocess.run.
# ---------------------------------------------------------------------------
def test_dry_run_prints_plan_without_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = CliRunner()
    # If subprocess.run is called during dry-run, fail the test loudly.
    with mock.patch(
        "scripts.eval_capture.subprocess.run",
        side_effect=AssertionError("subprocess.run must not be called in dry-run"),
    ):
        result = runner.invoke(
            eval_capture.main,
            ["--dry-run", "--regime", "r1"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    assert "r1_earnings" in result.output
    # Cassette path printed in the plan (path separator is platform-specific).
    assert "tests" in result.output and "eval" in result.output
    assert "cassettes" in result.output


# ---------------------------------------------------------------------------
# Test 2 — preflight fails fast when ANTHROPIC_API_KEY is unset.
# ---------------------------------------------------------------------------
def test_missing_api_key_fails_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    runner = CliRunner()
    with mock.patch(
        "scripts.eval_capture.subprocess.run",
        side_effect=AssertionError("subprocess.run must not be called when API key missing"),
    ):
        result = runner.invoke(
            eval_capture.main,
            ["--regime", "r1", "--yes"],
            catch_exceptions=False,
        )

    assert result.exit_code == 1
    # Click's CliRunner merges stderr into output by default.
    assert "ANTHROPIC_API_KEY" in result.output


# ---------------------------------------------------------------------------
# Test 3 — --regime all dispatches exactly 3 subprocesses with record-mode env.
# ---------------------------------------------------------------------------
def test_dispatches_one_subprocess_per_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    runner = CliRunner()

    fake_completed = mock.Mock(returncode=0)
    with mock.patch(
        "scripts.eval_capture.subprocess.run",
        return_value=fake_completed,
    ) as mock_run:
        result = runner.invoke(
            eval_capture.main,
            ["--regime", "all", "--yes"],
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert mock_run.call_count == 3

    # Collect the args + env from each dispatch and assert they look right.
    seen_regime_flags: list[str] = []
    for call in mock_run.call_args_list:
        # Positional args list (argv) is the first positional kwarg or call.args[0].
        args = call.args[0] if call.args else call.kwargs["args"]
        env = call.kwargs["env"]

        assert env["FIRM_LLM_MODE"] == "record"
        assert env["FIRM_VCR_MODE"] == "record"
        assert env["FIRM_PRICES_MODE"] == "record"
        assert "FIRM_CASSETTE_DIR" in env
        assert env["FIRM_RANDOM_SEED"] == "42"
        assert "FIRM_HMAC_SECRET" in env
        # check=True is passed positionally; subprocess.run will raise on
        # CalledProcessError, which the script translates to sys.exit. We only
        # care that the contract was used — verify by inspecting kwargs.
        assert call.kwargs.get("check") is True

        # Extract the --regime flag from argv (--regime <r1|r2|r3>).
        regime_idx = args.index("--regime")
        seen_regime_flags.append(args[regime_idx + 1])

    assert sorted(seen_regime_flags) == ["r1", "r2", "r3"]


# ---------------------------------------------------------------------------
# Test 4 — cost prompt aborts cleanly when the operator declines.
# ---------------------------------------------------------------------------
def test_cost_prompt_aborts_on_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    runner = CliRunner()

    with (
        mock.patch(
            "scripts.eval_capture.click.confirm",
            return_value=False,
        ) as mock_confirm,
        mock.patch(
            "scripts.eval_capture.subprocess.run",
            side_effect=AssertionError("subprocess.run must not be called when operator aborts"),
        ) as mock_run,
    ):
        result = runner.invoke(
            eval_capture.main,
            ["--regime", "r1"],  # no --yes => prompt fires
            catch_exceptions=False,
        )

    assert result.exit_code == 0, result.output
    assert mock_confirm.called
    assert mock_run.call_count == 0
    assert "Aborted" in result.output
