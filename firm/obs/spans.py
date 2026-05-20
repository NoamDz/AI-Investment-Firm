"""Span decorators / context managers for the four agent operation kinds (Plan 3 T02).

Each helper is usable both as a decorator and as a context manager (via
:func:`contextlib.contextmanager`).  On entry the span is opened via
:func:`firm.obs.tracer.get_tracer` and the canonical ``agent`` / ``operation``
attributes are set immediately.  On exception, ``status="error"`` and
``failure_mode=<exception class name>`` are recorded and the exception is
**re-raised** (never swallowed).

Operation-name scheme
---------------------
* ``agent_span("research")``                 -> operation = ``agent.research``
* ``llm_span("anthropic", "claude-sonnet")`` -> operation = ``llm.call``
                                                (``provider`` + ``model`` attrs)
* ``tool_span("fundamentals.get_ratio")``    -> operation = ``tool.fundamentals.get_ratio``
* ``retrieval_span("hybrid")``               -> operation = ``retrieval.hybrid``

This scheme is stable across agent / LLM / retrieval call sites so that the
T05 integration assertion (">= 1 span per agent + 1 per LLM call + 1 per
retrieval stage") is a simple prefix check against ``operation``.

``llm_span`` uses the constant operation name ``"llm.call"`` (with provider /
model on attributes) so cost-ledger queries can group all LLM calls under one
operation; the other three decorators parameterize the operation name so that
per-agent / per-tool / per-retrieval-stage rollups are direct attribute filters.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from opentelemetry.trace import Span

from firm.obs.tracer import get_tracer


def _record_exception(span: Span, exc: Exception) -> None:
    """Tag *span* with ``status=error`` and ``failure_mode=<class name>``."""
    span.set_attribute("status", "error")
    span.set_attribute("failure_mode", type(exc).__name__)


@contextmanager
def agent_span(agent: str) -> Iterator[Span]:
    """Open a span for an agent's top-level operation.

    Operation is ``agent.<agent>`` (e.g. ``agent.research``).  The ``agent``
    attribute is set to *agent* so downstream span-by-agent queries work.

    Usable as either::

        with agent_span("research"):
            ...

    or::

        @agent_span("research")
        def run(...): ...
    """
    tracer = get_tracer(agent)
    operation = f"agent.{agent}"
    with tracer.start_as_current_span(operation) as span:
        span.set_attribute("agent", agent)
        span.set_attribute("operation", operation)
        try:
            yield span
        except Exception as exc:
            _record_exception(span, exc)
            raise


@contextmanager
def llm_span(provider: str, model: str) -> Iterator[Span]:
    """Open a span for a single LLM invocation.

    Operation is the constant string ``llm.call`` (so cost / token-rollup
    queries can filter on a single name regardless of provider).  ``provider``
    and ``model`` are recorded as span attributes; ``agent`` is inherited from
    the enclosing :func:`agent_span` via OTel's current-span context.
    """
    tracer = get_tracer(f"llm.{provider}")
    operation = "llm.call"
    with tracer.start_as_current_span(operation) as span:
        span.set_attribute("operation", operation)
        span.set_attribute("provider", provider)
        span.set_attribute("model", model)
        try:
            yield span
        except Exception as exc:
            _record_exception(span, exc)
            raise


@contextmanager
def tool_span(tool_name: str) -> Iterator[Span]:
    """Open a span for a tool invocation.

    Operation is ``tool.<tool_name>`` (e.g. ``tool.fundamentals.get_ratio``).
    The tool name is recorded verbatim as ``operation`` so dashboards can
    group by tool.
    """
    tracer = get_tracer(f"tool.{tool_name}")
    operation = f"tool.{tool_name}"
    with tracer.start_as_current_span(operation) as span:
        span.set_attribute("operation", operation)
        try:
            yield span
        except Exception as exc:
            _record_exception(span, exc)
            raise


@contextmanager
def retrieval_span(stage: str) -> Iterator[Span]:
    """Open a span for a retrieval-pipeline stage (``hybrid`` | ``rerank`` | ``pit``).

    Operation is ``retrieval.<stage>``.
    """
    tracer = get_tracer(f"retrieval.{stage}")
    operation = f"retrieval.{stage}"
    with tracer.start_as_current_span(operation) as span:
        span.set_attribute("operation", operation)
        try:
            yield span
        except Exception as exc:
            _record_exception(span, exc)
            raise


__all__ = [
    "agent_span",
    "llm_span",
    "retrieval_span",
    "tool_span",
]
