"""Smoke tests for the --source option added by T21."""
from __future__ import annotations

from click.testing import CliRunner

from firm.cli import cli


def test_cli_ingest_rejects_bad_source() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--source", "notavalue"])
    assert result.exit_code != 0
    assert "notavalue" in (result.output or "")
