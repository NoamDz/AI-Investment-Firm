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
    r"(?:Table\s+of\s+Contents|Forward[‐-]Looking\s+Statements)[^\n]*",
    re.IGNORECASE,
)

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

_FINANCE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(
    r"""
    \$[A-Z]+          # dollar-prefixed ticker: $AAPL
    | [A-Z]+\.[A-Z]+  # dotted ticker: BRK.B
    | \d+-[A-Z]+      # filing form: 10-K, 8-K, 10-KA
    | [A-Za-z]+       # ordinary word
    | \d+             # number (kept for context)
    """,
    re.VERBOSE,
)


def normalize_text(text: str) -> str:
    """NFKC-normalize, strip zero-width chars, remove SEC boilerplate, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _BOILERPLATE_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def tables_to_prose(html: str) -> str:
    """Replace each HTML <table> with deterministic prose sentences.

    Expected table shape: a <thead> row whose first cell is a label header and
    remaining cells are period labels (e.g. "Q3 2024"), followed by <tbody> rows
    whose first <td> is a metric label and remaining <td>s are numeric values.

    One prose sentence per (metric, period, value) triple:
        "In <period>, <metric_label_lower> was $<value> million."
    """
    soup = BeautifulSoup(html, "lxml")

    for table in soup.find_all("table"):
        prose_lines: list[str] = []

        thead = table.find("thead")
        if not isinstance(thead, Tag):
            table.replace_with(soup.new_string(""))
            continue

        header_cells = thead.find_all(["th", "td"])
        period_labels = [c.get_text(strip=True) for c in header_cells[1:]]

        tbody = table.find("tbody")
        if not isinstance(tbody, Tag):
            table.replace_with(soup.new_string(""))
            continue

        for row in tbody.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            metric_label = cells[0].get_text(strip=True).lower()
            for i, period in enumerate(period_labels):
                if i + 1 >= len(cells):
                    break
                value = cells[i + 1].get_text(strip=True)
                if value:
                    prose_lines.append(
                        f"In {period}, {metric_label} was ${value} million."
                    )

        prose_text = " ".join(prose_lines)
        table.replace_with(soup.new_string(prose_text))

    return str(soup)


def ticker_aware_tokens(text: str) -> list[str]:
    """Tokenize *text* preserving finance-specific tokens as single atoms.

    Finance tokens matched first (in order of the alternation):
    - Dollar-prefixed tickers: $AAPL
    - Dotted tickers: BRK.B
    - Filing form names: 10-K, 8-K

    Remaining alphabetic and numeric runs are returned as-is (case preserved).
    Punctuation and whitespace that is not part of a recognised token is dropped.
    """
    return _FINANCE_TOKEN_RE.findall(text)
