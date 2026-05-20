"""Tests for fundamentals.get_ratio MCP-style tool and precompute script.

Uses Option A: a session-scoped pytest fixture that invokes the precompute
script in fixture mode (FIRM_FUNDAMENTALS_FIXTURE=1) and writes the parquet
into a pytest tmp dir. This avoids a pytest-xdist race on a shared
repo-relative path, and ensures no parquet survives between runs.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from firm.tools.fundamentals import FundamentalsTool

SCRIPT = (
    Path(__file__).parent.parent.parent / "scripts" / "precompute_fundamentals.py"
)


@pytest.fixture(scope="session")
def fixture_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run precompute script in fixture mode into a tmp dir; return parquet path."""
    out_dir = tmp_path_factory.mktemp("fundamentals")
    out_path = out_dir / "fundamentals_fixture.parquet"
    env = {
        **os.environ,
        "FIRM_FUNDAMENTALS_FIXTURE": "1",
        "FIRM_FUNDAMENTALS_OUT": str(out_path),
    }
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"precompute script failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert out_path.exists(), (
        f"Expected parquet at {out_path} but it was not created."
    )
    return out_path


@pytest.fixture(scope="session")
def tool(fixture_parquet: Path) -> FundamentalsTool:
    return FundamentalsTool(fixture_parquet)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_ratio_returns_decimal_for_known_pair(tool: FundamentalsTool) -> None:
    """get_ratio returns a Decimal, not a float."""
    result = tool.get_ratio(ticker="AAPL", ratio_name="pe_ratio", as_of=date(2024, 11, 1))
    assert isinstance(result, Decimal), f"Expected Decimal, got {type(result)}"
    assert result == Decimal("28.5")


def test_get_ratio_raises_for_unknown_ticker(tool: FundamentalsTool) -> None:
    """Unknown ticker raises KeyError with a descriptive message."""
    with pytest.raises(KeyError, match="ZZZ"):
        tool.get_ratio(ticker="ZZZ", ratio_name="pe_ratio", as_of=date(2024, 11, 1))


def test_get_ratio_raises_for_unknown_ratio_name(tool: FundamentalsTool) -> None:
    """Unknown ratio_name raises KeyError with a descriptive message."""
    with pytest.raises(KeyError, match="not_a_ratio"):
        tool.get_ratio(ticker="AAPL", ratio_name="not_a_ratio", as_of=date(2024, 11, 1))


def test_get_ratio_uses_as_of_to_select_latest_filing(tool: FundamentalsTool) -> None:
    """PIT lookup: as_of=2024-03-13 selects the 2024-02-01 row (not 2024-05-01)."""
    result = tool.get_ratio(ticker="AAPL", ratio_name="pe_ratio", as_of=date(2024, 3, 13))
    assert result == Decimal("27.0"), (
        f"Expected Decimal('27.0') from 2024-02-01 row, got {result!r}"
    )


def test_get_ratio_raises_when_no_entry_on_or_before_as_of(
    tool: FundamentalsTool,
) -> None:
    """KeyError when as_of is before all filings for the (ticker, ratio_name) pair."""
    with pytest.raises(KeyError):
        tool.get_ratio(ticker="AAPL", ratio_name="pe_ratio", as_of=date(2020, 1, 1))


def test_get_ratio_accepts_iso_date_string_for_as_of(
    tool: FundamentalsTool,
) -> None:
    """get_ratio accepts ISO 8601 date string (Anthropic tool_use JSON payload)."""
    result_date = tool.get_ratio(
        ticker="AAPL", ratio_name="pe_ratio", as_of=date(2024, 11, 1)
    )
    result_str = tool.get_ratio(
        ticker="AAPL", ratio_name="pe_ratio", as_of="2024-11-01"
    )
    assert result_date == result_str == Decimal("28.5")


def test_get_ratio_signature_matches_mcp_tool_schema(
    tool: FundamentalsTool,
) -> None:
    """tool_def.input_schema is a valid JSON-schema-shaped dict."""
    schema = FundamentalsTool.tool_def.input_schema
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "ticker" in props
    assert "ratio_name" in props
    assert "as_of" in props
    required = schema["required"]
    assert set(required) == {"ticker", "ratio_name", "as_of"}
    # ratio_name must enumerate the 5 supported ratios
    ratio_enum = props["ratio_name"]["enum"]
    assert set(ratio_enum) == {
        "pe_ratio",
        "gross_margin",
        "revenue_yoy_growth",
        "debt_to_equity",
        "current_ratio",
    }
    # tool_def fields
    assert FundamentalsTool.tool_def.name == "fundamentals.get_ratio"
    assert len(FundamentalsTool.tool_def.description) > 10
