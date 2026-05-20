"""Tests for firm.rag.preprocess — T5 spec compliance."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from firm.rag.preprocess import normalize_text, tables_to_prose, ticker_aware_tokens

_FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "financebench_two_docs.json"


@pytest.fixture(scope="module")
def two_docs() -> list[dict[str, str]]:
    with _FIXTURE_PATH.open() as fh:
        return json.load(fh)["docs"]  # type: ignore[no-any-return]


def test_html_table_converted_to_prose(two_docs: list[dict[str, str]]) -> None:
    aapl_doc = next(d for d in two_docs if d["ticker"] == "AAPL")
    result = tables_to_prose(aapl_doc["html"])
    assert "In Q3 2024, total revenue was $18,120 million" in result


def test_ticker_tokens_preserved() -> None:
    text = "We hold $AAPL and BRK.B; the 10-K was filed last quarter."
    tokens = ticker_aware_tokens(text)
    assert "$AAPL" in tokens
    assert "BRK.B" in tokens
    assert "10-K" in tokens


def test_strips_boilerplate_and_normalizes_whitespace() -> None:
    raw = (
        "Table of Contents\n\n"
        "Some​ text   with‌ zero‍ width﻿ chars.\n"
        "Forward-Looking Statements\n"
        "Normal   content   here."
    )
    cleaned = normalize_text(raw)

    assert "Table of Contents" not in cleaned
    assert "Forward-Looking Statements" not in cleaned
    assert "​" not in cleaned
    assert "‌" not in cleaned
    assert "‍" not in cleaned
    assert "﻿" not in cleaned
    assert "  " not in cleaned
    assert "Normal content here." in cleaned


def test_tables_to_prose_nvda(two_docs: list[dict[str, str]]) -> None:
    nvda_doc = next(d for d in two_docs if d["ticker"] == "NVDA")
    result = tables_to_prose(nvda_doc["html"])
    assert "In Q3 2024, total revenue was $35,082 million" in result
    assert "In Q3 2023, total revenue was $18,120 million" in result


def test_tables_to_prose_no_table_unchanged() -> None:
    html = "<html><body><p>No tables here.</p></body></html>"
    result = tables_to_prose(html)
    assert "No tables here." in result
    assert "<table" not in result


def test_ticker_aware_tokens_dollar_prefix() -> None:
    tokens = ticker_aware_tokens("$NVDA reported earnings.")
    assert "$NVDA" in tokens
    assert "reported" in tokens


def test_ticker_aware_tokens_filing_forms() -> None:
    tokens = ticker_aware_tokens("The 8-K and 10-KA filings were submitted.")
    assert "8-K" in tokens
    assert "10-KA" in tokens


def test_normalize_text_nfkc() -> None:
    text = "ﬁnance"
    cleaned = normalize_text(text)
    assert cleaned == "finance"


def test_normalize_text_multiple_spaces() -> None:
    cleaned = normalize_text("too   many     spaces")
    assert cleaned == "too many spaces"


# ── IMPORTANT #4 new tests ────────────────────────────────────────────────────


def test_table_without_thead_still_converts_or_preserves_text() -> None:
    """A <table> with no <thead>/<tbody> must not silently drop cell data."""
    html = (
        "<html><body>"
        "<table>"
        "<tr><td>Metric</td><td>Q1 2024</td></tr>"
        "<tr><td>Total revenue</td><td>18,120</td></tr>"
        "</table>"
        "</body></html>"
    )
    result = tables_to_prose(html)
    assert "<table" not in result.lower()
    assert "Total revenue" in result or "total revenue" in result
    assert "18,120" in result


def test_boilerplate_does_not_eat_legitimate_prose() -> None:
    """Boilerplate pattern must not consume text beyond the matched line."""
    text = "The Forward-Looking Statements section is on page 5."
    cleaned = normalize_text(text)
    assert "section is on page 5" in cleaned


def test_tables_to_prose_output_has_no_table_tags() -> None:
    """tables_to_prose must remove all <table> tags — the chunker invariant."""
    html = (
        "<html><body>"
        "<table><thead><tr><th>Metric</th><th>Q3 2024</th></tr></thead>"
        "<tbody><tr><td>Revenue</td><td>100</td></tr></tbody></table>"
        "</body></html>"
    )
    result = tables_to_prose(html)
    assert "<table" not in result.lower()
    assert "</table>" not in result.lower()


def test_ticker_tokens_adjacent_punctuation() -> None:
    """Trailing comma/period must not break dollar-ticker capture."""
    tokens = ticker_aware_tokens("Buy $AAPL, sell $MSFT.")
    assert "$AAPL" in tokens
    assert "$MSFT" in tokens
