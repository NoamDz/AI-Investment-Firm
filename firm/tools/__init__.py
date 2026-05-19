"""MCP-style tool definitions for the AI Investment Firm research pipeline."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, ClassVar, Protocol, runtime_checkable

from firm.tools.fundamentals import ToolDef


@runtime_checkable
class Tool(Protocol):
    """Uniform dispatch interface for research extractor tools.

    Both :class:`firm.tools.fundamentals.FundamentalsTool` and
    :class:`firm.tools.risk_metrics.RiskMetricsTool` satisfy this Protocol via
    their ``tool_def`` class variable and ``run(**kwargs)`` method.
    """

    tool_def: ClassVar[ToolDef]

    def run(self, **kwargs: Any) -> Decimal:
        """Execute the tool with the given keyword arguments and return a Decimal."""
        ...


__all__ = ["Tool", "ToolDef"]
