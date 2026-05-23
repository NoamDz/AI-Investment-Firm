"""Shared date-formatting helpers for the eval harness (Plan 4 §T15).

Extracted from ``firm.eval.runner`` so the summary aggregator can reuse
``format_header_dates`` without importing a private name. Behavior is
unchanged from the original ``_format_header_dates`` in T13.
"""
from __future__ import annotations

from datetime import date


def format_header_dates(start: date, end: date) -> str:
    """Format ``start..end`` as ``Mon dd–dd, YYYY`` (en-dash U+2013).

    Same-month windows collapse to a single ``Mon`` prefix; cross-month
    windows fall back to ``Mon dd – Mon dd, YYYY``. Days have no leading
    zero. The 3 fixed regimes are all single-month, but the cross-month
    branch is kept for robustness so the template never has to think about
    date math.
    """
    if start.year != end.year:
        raise ValueError("cross-year windows not supported by format_header_dates")
    if start.year == end.year and start.month == end.month:
        return f"{start.strftime('%b')} {start.day}–{end.day}, {start.year}"
    # Cross-month fallback. Year of ``start`` is the report year by
    # convention; cross-year regimes aren't a configured shape and are
    # rejected above.
    return (
        f"{start.strftime('%b')} {start.day} – "
        f"{end.strftime('%b')} {end.day}, {start.year}"
    )


__all__ = ["format_header_dates"]
