"""Finance-aware HTML preprocessing: table-to-prose, ticker-aware tokenization, and
text hygiene.  See design spec §8.3 and T5 implementation notes.

Deviations from spec regex patterns:
- `[A-Z]+\\.[A-Z]+` used instead of `[A-Z]+\\.[A-Z]` so that `BRK.B` is captured in full
  (the single-char variant would only keep `BRK.B` as the match boundary, but Python's
  non-overlapping findall would stop at the dot + one char, dropping multi-char suffixes).
- `\\d+-[A-Z]+` used instead of `\\d+-[A-Z]` so that `10-K`, `10-KA`, `8-K` all tokenize
  as single atoms.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Final

from bs4 import BeautifulSoup, Tag

_ZERO_WIDTH: Final[str] = "​‌‍﻿"
_ZERO_WIDTH_RE: Final[re.Pattern[str]] = re.compile(f"[{re.escape(_ZERO_WIDTH)}]")

_BOILERPLATE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?im)^[ \t]*(?:Table of Contents|Forward[‐‑‒–—\-]Looking Statements)[ \t]*$"
)

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

_FINANCE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    \$[A-Z]+
    | [A-Z]+\.[A-Z]+
    | \d+-[A-Z]+
    | [A-Za-z]+
    | \d+
    """,
    re.VERBOSE,
)

_MILLIONS_CONTEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"\bmillions\b", re.IGNORECASE
)


def normalize_text(text: str) -> str:
    """NFKC-normalize, strip zero-width chars, remove SEC boilerplate, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = re.sub(_BOILERPLATE_RE, "", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _format_value(value: str, millions_context: bool) -> str:
    """Return a formatted value string for prose output.

    - If value already contains ``$`` or ``%``: return as-is.
    - If *millions_context* is True (column header says "$ in millions"): prefix ``$`` and
      append `` million``.
    - Otherwise: return as-is (let surrounding text carry unit context).
    """
    if "$" in value or "%" in value:
        return value
    if millions_context:
        return f"${value} million"
    return value


def tables_to_prose(html: str) -> str:
    """Replace each HTML <table> with deterministic prose sentences.

    Expected table shape: a <thead> row whose first cell is a label header and
    remaining cells are period labels (e.g. "Q3 2024"), followed by <tbody> rows
    whose first <td> is a metric label and remaining <td>s are numeric values.

    One prose sentence per (metric, period, value) triple:
        "In <period>, <metric_label_lower> was <formatted_value>."

    Robustness:
    - If no ``<thead>`` is present, the first ``<tr>`` of the table is used as the
      header row.
    - If no ``<tbody>`` is present, all ``<tr>`` rows after the first are treated as
      body rows.
    - If zero prose lines are produced (single-column, header-only, or unrecognised
      shape), the table is replaced with its plain text (via ``get_text``) rather than
      an empty string, preserving data for downstream retrieval.
    """
    soup = BeautifulSoup(html, "lxml")

    for table in soup.find_all("table"):
        assert isinstance(table, Tag)
        prose_lines: list[str] = []

        thead = table.find("thead")
        tbody = table.find("tbody")

        if isinstance(thead, Tag):
            header_cells = thead.find_all(["th", "td"])
        else:
            # Fall back: treat first <tr> anywhere in the table as the header row.
            all_rows = table.find_all("tr")
            if not all_rows:
                table.replace_with(soup.new_string(table.get_text(separator=" ", strip=True)))
                continue
            header_cells = all_rows[0].find_all(["th", "td"])

        period_labels = [c.get_text(strip=True) for c in header_cells[1:]]

        # Detect "$ in millions" context: check column headers AND adjacent siblings.
        header_text = " ".join(c.get_text(strip=True) for c in header_cells)
        sibling_text = ""
        for sibling in list(table.previous_siblings) + list(table.next_siblings):
            if isinstance(sibling, Tag):
                sibling_text += " " + sibling.get_text(separator=" ", strip=True)
        millions_context = bool(
            _MILLIONS_CONTEXT_RE.search(header_text + " " + sibling_text)
        )

        body_rows: list[Tag]
        if isinstance(tbody, Tag):
            body_rows = list(tbody.find_all("tr"))
        else:
            # Fall back: all <tr> elements after the first.
            all_rows = table.find_all("tr")
            body_rows = list(all_rows[1:])

        for row in body_rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            metric_label = cells[0].get_text(strip=True).lower()
            for i, period in enumerate(period_labels):
                if i + 1 >= len(cells):
                    break
                value = cells[i + 1].get_text(strip=True)
                if value:
                    formatted = _format_value(value, millions_context)
                    prose_lines.append(
                        f"In {period}, {metric_label} was {formatted}."
                    )

        if prose_lines:
            prose_text = " ".join(prose_lines)
        else:
            # Fall back to plain text so no data is silently lost.
            prose_text = table.get_text(separator=" ", strip=True)

        table.replace_with(soup.new_string(prose_text))

    return str(soup)


def ticker_aware_tokens(text: str) -> list[str]:
    """Tokenize *text* preserving finance-specific tokens as single atoms.

    Finance tokens matched first (in order of the alternation):
    - Dollar-prefixed tickers: $AAPL
    - Dotted tickers: BRK.B
    - Filing form names: 10-K, 8-K

    Remaining alphabetic and numeric runs are returned with their original case
    (no lowercasing is applied).
    Punctuation and whitespace that is not part of a recognised token is dropped.

    Note: decimals and comma-grouped numbers are split (e.g. ``1.25`` → ``['1', '25']``,
    ``18,120`` → ``['18', '120']``); this is acceptable for BM25 sparse retrieval where
    preserving them is not needed for token-overlap scoring.
    """
    return _FINANCE_TOKEN_RE.findall(text)
