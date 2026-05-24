"""OTel TracerProvider + JSONL file exporter (spec §10.1).

Usage
-----
Production (batch, non-blocking)::

    from firm.obs.tracer import init_tracer, get_tracer
    init_tracer(traces_root=Path("traces"), run_id=ulid_new())
    tracer = get_tracer("research")

Tests / deterministic replay::

    from firm.obs.tracer import use_sync_exporter, get_tracer
    use_sync_exporter(traces_root=tmp_path)   # safe to call repeatedly
    tracer = get_tracer("research")

Environment overrides
---------------------
``OTEL_EXPORTER``   ``file`` (default) | ``otlp``
``FIRM_OTEL_SYNC``  ``1`` → swap in ``SimpleSpanProcessor`` (test mode);
                    consulted on first ``init_tracer``/``get_tracer`` call.

Design note
-----------
OpenTelemetry's global :func:`opentelemetry.trace.set_tracer_provider` may only
be called once per process — subsequent calls log a warning and are silently
ignored.  We therefore build the ``TracerProvider`` exactly once and mutate the
underlying :class:`JsonlFileExporter`'s ``traces_root`` / ``run_id`` in place
when tests rebind them via :func:`use_sync_exporter`.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date
from pathlib import Path
from typing import Any, Optional, Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JSONL file exporter
# ---------------------------------------------------------------------------

_WRITE_LOCK = threading.Lock()


class JsonlFileExporter(SpanExporter):
    """Exports each completed span as one JSON line to ``traces/YYYY-MM-DD/run-<run_id>.jsonl``.

    The exporter is thread-safe: a module-level lock serialises concurrent
    ``export()`` calls so that lines from different threads don't interleave.

    ``traces_root`` and ``run_id`` are public, mutable attributes — callers
    (notably :func:`use_sync_exporter`) rebind them in place to redirect
    output without rebuilding the global :class:`TracerProvider`.  Note:
    rebinding ``traces_root`` / ``run_id`` is intended for test setup only
    and is not safe concurrent with active span emission.  Production code
    should configure both at :func:`init_tracer` time and not mutate them
    afterward.
    """

    def __init__(self, traces_root: Path, run_id: str) -> None:
        self.traces_root = traces_root
        self.run_id = run_id

    def _output_path(self) -> Path:
        today = date.today().isoformat()
        day_dir = self.traces_root / today
        day_dir.mkdir(parents=True, exist_ok=True)
        return day_dir / f"run-{self.run_id}.jsonl"

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            output_path = self._output_path()
            lines: list[str] = []
            for span in spans:
                lines.append(json.dumps(_span_to_dict(span), ensure_ascii=False))
            with _WRITE_LOCK:
                with output_path.open("a", encoding="utf-8") as fh:
                    for line in lines:
                        fh.write(line + "\n")
            return SpanExportResult.SUCCESS
        except Exception:  # noqa: BLE001
            _logger.exception("JsonlFileExporter.export failed")
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass  # no resources to release

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # noqa: ARG002
        return True


def _span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    """Convert a ``ReadableSpan`` to the spec §10.1 dict.

    All 15 fields are always present; missing attribute values are represented
    by their natural zero-value (``None`` for IDs, ``""`` for strings, ``0``
    for integers, ``0.0`` for floats).
    """
    ctx = span.context
    trace_id_hex: str = format(ctx.trace_id, "032x") if ctx else ""
    span_id_hex: str = format(ctx.span_id, "016x") if ctx else ""

    parent_span_id: Optional[str] = None
    if span.parent and span.parent.span_id:
        parent_span_id = format(span.parent.span_id, "016x")

    # Duration in milliseconds
    duration_ms: float = 0.0
    if span.start_time and span.end_time:
        duration_ms = (span.end_time - span.start_time) / 1_000_000.0

    attrs: dict[str, Any] = dict(span.attributes or {})

    def _str(key: str) -> str:
        v = attrs.get(key)
        return str(v) if v is not None else ""

    def _int(key: str) -> int:
        v = attrs.get(key)
        try:
            return int(v) if v is not None else 0
        except (TypeError, ValueError):
            return 0

    def _float(key: str) -> float:
        v = attrs.get(key)
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    return {
        "trace_id": trace_id_hex,
        "span_id": span_id_hex,
        "parent_span_id": parent_span_id,
        "agent": _str("agent"),
        "operation": _str("operation"),
        "decision_id": _str("decision_id"),
        "duration_ms": duration_ms,
        "model": _str("model"),
        "input_tokens": _int("input_tokens"),
        "output_tokens": _int("output_tokens"),
        "cached_tokens": _int("cached_tokens"),
        "cost_usd": _float("cost_usd"),
        "citations": _int("citations"),
        "failure_mode": _str("failure_mode"),
        "status": _str("status"),
    }


# ---------------------------------------------------------------------------
# Provider management
# ---------------------------------------------------------------------------

# Module-level mutable state protected by _PROVIDER_LOCK.  We keep a single
# shared TracerProvider for the lifetime of the process because OTel's global
# ``set_tracer_provider`` is set-once.  Per-test redirection is achieved by
# mutating ``_exporter.traces_root`` / ``_exporter.run_id`` in place.
_PROVIDER_LOCK = threading.Lock()
_provider: Optional[TracerProvider] = None
_exporter: Optional[JsonlFileExporter] = None
_DEFAULT_TRACES_ROOT = Path("data/traces")
_DEFAULT_RUN_ID = "00000000000000000000000000"


def _build_and_register_provider(
    *,
    traces_root: Path,
    run_id: str,
    sync: bool,
) -> None:
    """Build the global TracerProvider exactly once and register it.

    Subsequent calls update ``_exporter`` in place instead (handled by the
    public entrypoints).
    """
    global _provider, _exporter  # noqa: PLW0603

    resource = Resource.create({"service.name": "ai-investment-firm"})
    provider = TracerProvider(resource=resource)
    exporter = JsonlFileExporter(traces_root=traces_root, run_id=run_id)
    if sync:
        processor: BatchSpanProcessor | SimpleSpanProcessor = SimpleSpanProcessor(
            exporter
        )
    else:
        processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _provider = provider
    _exporter = exporter


def use_sync_exporter(
    *,
    traces_root: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> None:
    """Activate synchronous span flushing for deterministic test usage.

    On the *first* call this builds the global :class:`TracerProvider` with a
    :class:`SimpleSpanProcessor` and registers it.  On *subsequent* calls the
    provider is reused (OTel's ``set_tracer_provider`` is set-once); the
    exporter's ``traces_root`` and ``run_id`` are rebound in place so each
    test can redirect output to its own ``tmp_path``.

    Also activated automatically when ``FIRM_OTEL_SYNC=1`` is set in the
    environment and :func:`init_tracer` or :func:`get_tracer` is invoked
    (conftest calls this at session scope).
    """
    with _PROVIDER_LOCK:
        if _provider is None:
            _build_and_register_provider(
                traces_root=traces_root if traces_root is not None else _DEFAULT_TRACES_ROOT,
                run_id=run_id if run_id is not None else _DEFAULT_RUN_ID,
                sync=True,
            )
            _logger.debug(
                "firm.obs.tracer: sync exporter activated (traces_root=%s, run_id=%s)",
                _exporter.traces_root if _exporter else None,
                _exporter.run_id if _exporter else None,
            )
            return

        # Provider already exists — reuse it and rebind exporter in place.
        # NB: we intentionally do NOT call set_tracer_provider again because
        # OTel only honours the first registration.
        if _exporter is not None:
            if traces_root is not None:
                _exporter.traces_root = traces_root
            if run_id is not None:
                _exporter.run_id = run_id
            _logger.debug(
                "firm.obs.tracer: rebound exporter (traces_root=%s, run_id=%s)",
                _exporter.traces_root,
                _exporter.run_id,
            )


def init_tracer(
    *,
    traces_root: Optional[Path] = None,
    run_id: Optional[str] = None,
) -> None:
    """Initialise the global provider with the production-default ``BatchSpanProcessor``.

    Call once at CLI entry, passing the ``run_id`` minted by
    :func:`firm.core.ids.ulid_new`.  Subsequent calls reuse the existing
    provider (per OTel set-once semantics) and rebind the exporter's
    ``traces_root`` / ``run_id`` in place — useful in tests that need a fresh
    run_id per test function.

    Honours ``OTEL_EXPORTER=otlp`` to switch to the OTLP gRPC exporter; the
    optional ``opentelemetry-exporter-otlp-proto-grpc`` dependency is imported
    lazily and raises ``ImportError`` if missing.  ``FIRM_OTEL_SYNC=1`` swaps
    in ``SimpleSpanProcessor`` for deterministic flushing.
    """
    effective_root = traces_root if traces_root is not None else Path(
        os.environ.get("FIRM_TRACES_ROOT", "data/traces")
    )
    effective_run_id = run_id if run_id is not None else _DEFAULT_RUN_ID

    sync = os.environ.get("FIRM_OTEL_SYNC", "0") == "1"

    otel_exporter = os.environ.get("OTEL_EXPORTER", "file")
    if otel_exporter == "otlp":
        # OTLP path: defer import so the optional dependency isn't required in
        # environments that only use the file exporter.  Raises ImportError if
        # the dependency isn't installed — callers may catch + fall back.
        _init_otlp_provider(sync=sync)
        return

    with _PROVIDER_LOCK:
        if _provider is None:
            _build_and_register_provider(
                traces_root=effective_root,
                run_id=effective_run_id,
                sync=sync,
            )
            _logger.info(
                "firm.obs.tracer: initialised (exporter=file, sync=%s, run_id=%s)",
                sync,
                effective_run_id,
            )
            return

        # Provider already exists — reuse and rebind exporter.  This is the
        # intended pattern: OTel's global provider is set-once, so we update
        # mutable exporter state instead.
        if _exporter is not None:
            _exporter.traces_root = effective_root
            _exporter.run_id = effective_run_id
            _logger.info(
                "firm.obs.tracer: rebound exporter (run_id=%s)",
                effective_run_id,
            )


def _init_otlp_provider(*, sync: bool) -> None:
    """Set up an OTLP exporter (production path).

    Idempotent: on subsequent calls the existing provider is reused (no
    rebinding is performed since OTLP exporters don't expose mutable
    redirection knobs in the same way the file exporter does).
    """
    global _provider  # noqa: PLW0603

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
            OTLPSpanExporter,
        )
    except ImportError as exc:
        raise ImportError(
            "opentelemetry-exporter-otlp-proto-grpc is required for OTEL_EXPORTER=otlp. "
            "Install it with: uv pip install opentelemetry-exporter-otlp-proto-grpc"
        ) from exc

    with _PROVIDER_LOCK:
        if _provider is not None:
            _logger.warning(
                "OTEL_EXPORTER=otlp requested, but a TracerProvider is already "
                "active (likely file exporter from an earlier init_tracer call). "
                "The existing provider is kept. Set OTEL_EXPORTER before the "
                "first init_tracer() call to use OTLP."
            )
            return

        resource = Resource.create({"service.name": "ai-investment-firm"})
        provider = TracerProvider(resource=resource)
        otlp_exporter = OTLPSpanExporter()
        if sync:
            processor: BatchSpanProcessor | SimpleSpanProcessor = SimpleSpanProcessor(
                otlp_exporter
            )
        else:
            processor = BatchSpanProcessor(otlp_exporter)
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        _provider = provider
        _logger.info("firm.obs.tracer: initialised (exporter=otlp, sync=%s)", sync)


def get_tracer(agent: str) -> trace.Tracer:
    """Return an OTel :class:`~opentelemetry.trace.Tracer` for *agent*.

    If the global provider has not yet been initialised (e.g. in tests that
    call :func:`use_sync_exporter` and then ``get_tracer`` directly), a
    sync-mode provider is auto-initialised using the module-level defaults.

    The ``run_id`` is owned by :func:`init_tracer` / :func:`use_sync_exporter`;
    callers acquire a tracer *after* one of those has been called at CLI
    startup.  Per-call ``run_id`` rebinding is intentionally out of scope.
    """
    with _PROVIDER_LOCK:
        needs_init = _provider is None

    if needs_init:
        # Auto-init for callers that skip init_tracer (common in unit tests).
        # Honour FIRM_OTEL_SYNC=1 by routing through use_sync_exporter; the
        # env var is consulted lazily here rather than at import time.
        if os.environ.get("FIRM_OTEL_SYNC", "0") == "1":
            use_sync_exporter()
        else:
            init_tracer()

    return trace.get_tracer(f"firm.obs.{agent}")
