"""Tests for risk.get_metric MCP-style tool and precompute script.

Uses the same session-fixture pattern as test_fundamentals_tool.py:
a session-scoped fixture invokes the precompute script in fixture mode
(FIRM_RISK_FIXTURE=1) and writes the parquet into a pytest tmp dir.
This avoids a pytest-xdist race on a shared repo-relative path, and
ensures no parquet survives between runs.
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from firm.tools.risk_metrics import RiskMetricsTool

SCRIPT = (
    Path(__file__).parent.parent.parent / "scripts" / "precompute_risk_metrics.py"
)


@pytest.fixture(scope="session")
def fixture_parquet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run precompute script in fixture mode into a tmp dir; return parquet path."""
    out_dir = tmp_path_factory.mktemp("risk_metrics")
    out_path = out_dir / "risk_metrics_fixture.parquet"
    env = {
        **os.environ,
        "FIRM_RISK_FIXTURE": "1",
        "FIRM_RISK_OUT": str(out_path),
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
def tool(fixture_parquet: Path) -> RiskMetricsTool:
    return RiskMetricsTool(fixture_parquet)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_get_metric_returns_volatility_decimal_for_30d(tool: RiskMetricsTool) -> None:
    """get_metric returns a Decimal, not a float, for AAPL volatility_30d."""
    result = tool.get_metric(
        ticker="AAPL", metric="volatility_30d", as_of=date(2024, 11, 1)
    )
    assert isinstance(result, Decimal), f"Expected Decimal, got {type(result)}"
    assert result == Decimal("0.21")


def test_get_metric_raises_for_unknown_metric_name(tool: RiskMetricsTool) -> None:
    """Unknown metric raises KeyError with a descriptive message."""
    with pytest.raises(KeyError, match="not_a_metric"):
        tool.get_metric(
            ticker="AAPL", metric="not_a_metric", as_of=date(2024, 11, 1)
        )


def test_get_metric_raises_for_unknown_ticker(tool: RiskMetricsTool) -> None:
    """Unknown ticker raises KeyError with a descriptive message."""
    with pytest.raises(KeyError, match="ZZZ"):
        tool.get_metric(
            ticker="ZZZ", metric="volatility_30d", as_of=date(2024, 11, 1)
        )


def test_get_metric_uses_as_of_window(tool: RiskMetricsTool) -> None:
    """PIT lookup: as_of=2024-03-13 selects the 2024-02-01 row (not 2024-05-01)."""
    result = tool.get_metric(
        ticker="AAPL", metric="volatility_30d", as_of=date(2024, 3, 13)
    )
    assert result == Decimal("0.18"), (
        f"Expected Decimal('0.18') from 2024-02-01 row, got {result!r}"
    )


def test_get_metric_accepts_iso_date_string_for_as_of(
    tool: RiskMetricsTool,
) -> None:
    """get_metric accepts ISO 8601 date string (Anthropic tool_use JSON payload)."""
    result_date = tool.get_metric(
        ticker="AAPL", metric="volatility_30d", as_of=date(2024, 11, 1)
    )
    result_str = tool.get_metric(
        ticker="AAPL", metric="volatility_30d", as_of="2024-11-01"
    )
    assert result_date == result_str == Decimal("0.21")


def test_get_metric_signature_matches_mcp_tool_schema(
    tool: RiskMetricsTool,
) -> None:
    """tool_def.input_schema is a Mapping (not dict) with correct structure."""
    schema = RiskMetricsTool.tool_def.input_schema
    # Must be a Mapping, not a plain mutable dict
    assert isinstance(schema, Mapping), f"Expected Mapping, got {type(schema)}"
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "ticker" in props
    assert "metric" in props
    assert "as_of" in props
    required = schema["required"]
    assert set(required) == {"ticker", "metric", "as_of"}
    # metric must enumerate all 3 supported metrics
    metric_enum = props["metric"]["enum"]
    assert set(metric_enum) == {"volatility_30d", "beta_180d", "max_drawdown_90d"}
    # tool_def name and description
    assert RiskMetricsTool.tool_def.name == "risk.get_metric"
    assert len(RiskMetricsTool.tool_def.description) > 10
