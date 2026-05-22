"""One-time cassette + price-parquet capture for the eval harness (Plan 4 T16).

Runs each regime in *record* mode against the real Anthropic API + yfinance,
writing YAML cassettes under ``tests/eval/cassettes/<regime_id>/`` and price
parquets under ``data/prices_eval/<TICKER>.parquet`` (path is hard-coded by
``firm.cli.eval_cmd``, not the ``benchmarks._default_prices_dir`` default).
Operator-triggered only — must NEVER be wired into CI or a Makefile target;
doing so would burn API budget on every push.

Surface
-------
``python scripts/eval_capture.py [--regime r1|r2|r3|all] [--dry-run] [--yes]``

* ``--regime``  defaults to ``all``.
* ``--dry-run`` prints the per-regime plan (env vars, target paths) without
  invoking the eval subprocess. ``ANTHROPIC_API_KEY`` is NOT required.
* ``--yes``     skips the interactive cost-confirmation prompt.

TODO (Plan 4 T16.2)
-------------------
A ``--stub`` mode for fully-offline cassette generation was scoped but
NOT implemented. Stubbing the LLM transport alone is insufficient: a real
eval run also depends on a running Qdrant instance, downloaded
sentence-transformers + BGE rerank models, and a populated BM25 corpus —
none of which a stand-alone script can fake without duplicating most of
``firm.cli._build_llm_stack`` infrastructure. The honest engineering
decision was to leave ``--stub`` unimplemented and let T16.1's
loud-fail (a ``click.ClickException`` from ``firm eval`` on a missing
parquet) communicate the gap. A future PR that ships a true ``--stub``
mode should:

  1. Add a ``StubAnthropicTransport`` implementing
     ``AnthropicTransport`` with sha256-deterministic canned responses
     keyed off (prompt_hash, model) — see
     ``firm/llm/anthropic_client.py`` for the protocol shape.
  2. Add a ``FIRM_LLM_MODE=stub`` recognised by
     ``CachedAnthropicClient.from_env`` that substitutes the stub
     transport for ``_AnthropicSdkTransport`` while still letting the
     cassette layer wrap it.
  3. Generate fixture price parquets via
     ``numpy.random`` seeded by ``hash(ticker)`` (~10 OHLCV rows per
     ticker over the regime window) and write them through
     ``firm.eval.benchmarks._write_parquet_series``.
  4. Provide a way to skip Qdrant + sentence-transformers (likely an
     additional ``FIRM_RAG_MODE=stub`` env var that swaps the retriever
     for a fixture-backed one). This is the largest piece of work.

Until that lands, operators must run the recording variant with a real
``ANTHROPIC_API_KEY`` once to populate ``tests/eval/cassettes/`` and
``data/prices_eval/``. The recorded artifacts can then be committed.

Each regime is launched in its OWN subprocess so env-var mutations
(``FIRM_LLM_MODE=record`` / ``FIRM_VCR_MODE=record`` / ``FIRM_PRICES_MODE=
record`` / ``FIRM_CASSETTE_DIR=...``) don't leak across regimes — running
in-process would require monkeypatching the determinism defaults the eval
CLI applies.

The eval reports written by each subprocess land under
``data/captured/<regime_id>/`` — a throwaway directory that should NOT be
committed; only the cassettes + parquets are committable artifacts.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import click

# Repo root resolved from this file's location so the script works regardless
# of the operator's CWD.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_CASSETTE_ROOT = _REPO_ROOT / "tests" / "eval" / "cassettes"
# Mirror the explicit ``prices_dir=Path("data/prices_eval")`` that
# ``firm.cli.eval_cmd`` passes to ``compute_spy_return`` / ``compute_basket_return``
# (overrides the ``benchmarks._default_prices_dir`` default). Dry-run output and
# the operator-facing reminder MUST match where parquets actually land.
_PRICES_DIR = _REPO_ROOT / "data" / "prices_eval"
_CAPTURED_ROOT = _REPO_ROOT / "data" / "captured"

# Map --regime flag values to the canonical regime IDs accepted by
# ``firm.cli eval --regime`` (plus their long-form IDs used for path layout).
_REGIME_MAP: dict[str, str] = {
    "r1": "r1_earnings",
    "r2": "r2_drawdown",
    "r3": "r3_quiet",
}

# Default HMAC secret if the operator hasn't set one — matches T15's
# ``_EVAL_DEFAULT_ENV["FIRM_HMAC_SECRET"]`` (64 hex zeros, fixture-grade,
# never production).
_DEFAULT_HMAC_SECRET = "0" * 64


def _selected_regime_flags(regime_arg: str) -> list[str]:
    """Resolve --regime input ('r1'|'r2'|'r3'|'all') to the CLI flag values."""
    if regime_arg == "all":
        return ["r1", "r2", "r3"]
    return [regime_arg]


def _build_subprocess_env(regime_flag: str) -> dict[str, str]:
    """Build the env vars passed to one ``firm eval`` subprocess.

    Forwards the operator's existing env (notably ``ANTHROPIC_API_KEY`` and
    ``FIRM_HMAC_SECRET`` if set), then overlays the record-mode flags + a
    per-regime ``FIRM_CASSETTE_DIR``. Operator's ``FIRM_HMAC_SECRET`` wins;
    otherwise the same 64-hex-zero fixture default T15 uses is applied so
    HMAC verification doesn't reject during recording.
    """
    env = os.environ.copy()
    regime_id = _REGIME_MAP[regime_flag]
    env["FIRM_LLM_MODE"] = "record"
    env["FIRM_VCR_MODE"] = "record"
    env["FIRM_PRICES_MODE"] = "record"
    env["FIRM_CASSETTE_DIR"] = str(_CASSETTE_ROOT / regime_id)
    env["FIRM_RANDOM_SEED"] = "42"
    env.setdefault("FIRM_HMAC_SECRET", _DEFAULT_HMAC_SECRET)
    # Note: FIRM_REPORTS_ROOT is intentionally NOT set here — the eval CLI
    # owns its own _artifacts dir under --output-dir.
    return env


def _build_subprocess_args(regime_flag: str) -> list[str]:
    """Build the argv for the ``firm eval`` subprocess for one regime."""
    regime_id = _REGIME_MAP[regime_flag]
    return [
        sys.executable,
        "-m",
        "firm.cli",
        "eval",
        "--regime",
        regime_flag,
        "--output-dir",
        str(_CAPTURED_ROOT / regime_id),
    ]


def _print_plan(regime_flags: list[str]) -> None:
    """Emit the dry-run plan: per-regime env + target paths."""
    click.echo("DRY RUN — no API calls will be made.")
    click.echo(f"Will process {len(regime_flags)} regime(s).")
    for flag in regime_flags:
        regime_id = _REGIME_MAP[flag]
        cassette_dir = _CASSETTE_ROOT / regime_id
        output_dir = _CAPTURED_ROOT / regime_id
        click.echo("")
        click.echo(f"  Regime: {regime_id} (--regime {flag})")
        click.echo("    FIRM_LLM_MODE=record")
        click.echo("    FIRM_VCR_MODE=record")
        click.echo("    FIRM_PRICES_MODE=record")
        click.echo(f"    FIRM_CASSETTE_DIR={cassette_dir}")
        click.echo(f"    Cassettes -> {cassette_dir}/<sha256>.yaml")
        click.echo(f"    Prices    -> {_PRICES_DIR}/<TICKER>.parquet")
        click.echo(f"    Reports   -> {output_dir}/ (NOT committed)")
    click.echo("")
    click.echo(
        f"After capture, commit {_CASSETTE_ROOT.relative_to(_REPO_ROOT)} "
        f"and {_PRICES_DIR.relative_to(_REPO_ROOT)} but NOT "
        f"{_CAPTURED_ROOT.relative_to(_REPO_ROOT)}."
    )


def _count_files(directory: Path, pattern: str) -> int:
    """Return the number of files in *directory* matching *pattern* (0 if absent)."""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob(pattern))


def _print_summary(regime_flags: list[str], elapsed: float) -> None:
    """Emit the post-capture summary: file counts, paths, reminder."""
    click.echo("")
    click.echo("=" * 60)
    click.echo("Capture complete.")
    click.echo(f"Regimes processed : {len(regime_flags)}")
    click.echo(f"Elapsed (wall)    : {elapsed:.1f}s")
    click.echo("")
    click.echo("Cassette directories:")
    for flag in regime_flags:
        regime_id = _REGIME_MAP[flag]
        cassette_dir = _CASSETTE_ROOT / regime_id
        count = _count_files(cassette_dir, "*.yaml")
        click.echo(f"  {cassette_dir}: {count} YAML file(s)")
    click.echo("")
    parquet_count = _count_files(_PRICES_DIR, "*.parquet")
    click.echo(f"Price parquets: {_PRICES_DIR}: {parquet_count} file(s)")
    click.echo("")
    click.echo(
        "Reminder: commit `tests/eval/cassettes/` and `data/prices_eval/` "
        "to git; do NOT commit `data/captured/`."
    )


@click.command()
@click.option(
    "--regime",
    "regime_arg",
    type=click.Choice(["r1", "r2", "r3", "all"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Which regime to capture (r1/r2/r3) or 'all' for the full sweep.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the per-regime plan without invoking the eval subprocess.",
)
@click.option(
    "--yes",
    "yes_flag",
    is_flag=True,
    default=False,
    help="Skip the interactive cost-confirmation prompt (non-interactive runs).",
)
@click.option(
    "--stub",
    "stub_flag",
    is_flag=True,
    default=False,
    help=(
        "(NOT IMPLEMENTED — Plan 4 T16.2) Offline cassette generation. "
        "Currently raises ClickException with operator guidance; see "
        "this script's module docstring for the rollout plan."
    ),
)
def main(regime_arg: str, dry_run: bool, yes_flag: bool, stub_flag: bool) -> None:
    """Capture eval cassettes + price parquets in record mode."""
    regime_arg = regime_arg.lower()
    regime_flags = _selected_regime_flags(regime_arg)

    # ------------------------------------------------------------------
    # T16.2 stub mode is intentionally not implemented (see module
    # docstring). The flag exists so the surface is discoverable and a
    # future implementer can plug it in without changing the CLI shape.
    # ------------------------------------------------------------------
    if stub_flag:
        raise click.ClickException(
            "--stub mode is not implemented (Plan 4 T16.2 deferred). "
            "Stubbing the LLM transport alone is insufficient because "
            "the eval graph also requires Qdrant + sentence-transformers "
            "+ BM25 corpus. Run with a real ANTHROPIC_API_KEY (omit "
            "--stub) to populate tests/eval/cassettes/ and "
            "data/prices_eval/, then commit the resulting fixtures. See "
            "scripts/eval_capture.py module docstring for the planned "
            "--stub rollout."
        )

    # ------------------------------------------------------------------
    # Dry-run short-circuits BEFORE the API-key preflight: the operator
    # may legitimately want to preview the plan without exporting a key.
    # ------------------------------------------------------------------
    if dry_run:
        _print_plan(regime_flags)
        return

    # Preflight: refuse to run without ANTHROPIC_API_KEY.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        click.echo(
            "ERROR: ANTHROPIC_API_KEY is not set. Export the key before "
            "running scripts/eval_capture.py (or use --dry-run to preview).",
            err=True,
        )
        sys.exit(1)

    # Cost-warning confirmation (skippable with --yes).
    if not yes_flag:
        proceed = click.confirm(
            "This will make REAL Anthropic API calls (estimated cost: a few "
            "dollars) and download real yfinance prices. Continue?",
            default=False,
        )
        if not proceed:
            click.echo("Aborted by operator.")
            return

    # ------------------------------------------------------------------
    # Per-regime: dispatch a fresh subprocess so env mutations don't leak.
    # ------------------------------------------------------------------
    start = time.monotonic()
    for flag in regime_flags:
        regime_id = _REGIME_MAP[flag]
        click.echo("")
        click.echo(f"=== Capturing regime {regime_id} ===")
        env = _build_subprocess_env(flag)
        args = _build_subprocess_args(flag)
        try:
            # cwd=_REPO_ROOT anchors the subprocess so the relative prices_dir
            # hard-coded in firm.cli.eval_cmd resolves under the repo root,
            # not the operator's invoking shell CWD.
            subprocess.run(args, env=env, check=True, cwd=str(_REPO_ROOT))
        except subprocess.CalledProcessError as exc:
            click.echo(
                f"ERROR: eval subprocess for regime {regime_id} exited "
                f"with code {exc.returncode}. Aborting capture.",
                err=True,
            )
            sys.exit(exc.returncode or 1)
    elapsed = time.monotonic() - start

    _print_summary(regime_flags, elapsed)


if __name__ == "__main__":
    main()
