"""TranscriptsCorpusSource — local JSONL adapter for earnings-call transcripts.

Each JSONL line must be a JSON object with required fields:
  ticker        : str    (symbol, e.g. "AAPL")
  quarter       : int    (1, 2, 3, or 4)
  fiscal_year   : int    (e.g. 2024)
  published_at  : str    (ISO 8601, tz-aware)
  body          : str    (transcript text)

Missing or empty values for any required field raise ValueError per
spec §6.3 "no-NULL" data discipline — the corpus is rejected, not
silently coerced. The Pydantic validator on FilingDoc enforces
tz-awareness on published_at as a second line of defence.

The adapter implements the CorpusSource protocol and slots directly into
IngestPipeline; the existing chunker and ContextualAugmenter are reused
unchanged once iter_docs() yields FilingDoc instances.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime
from pathlib import Path

from firm.rag.source import FilingDoc


def _load_jsonl(path: Path) -> Iterable[dict[str, object]]:
    """Yield dicts from a UTF-8 JSONL file, skipping blank lines.

    Opens with ``utf-8-sig`` so a leading BOM (common in Windows/Excel
    exports) is transparently stripped instead of surfacing as a
    misleading JSON-decode error on line 1.
    """
    with path.open("r", encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip blank lines silently
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"transcripts JSONL line {lineno} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"transcripts JSONL line {lineno} is not a JSON object"
                )
            yield row


def _require_str(row: dict[str, object], field: str) -> str:
    """Return row[field] as a non-empty str or raise ValueError."""
    val = row.get(field)
    if val is None:
        raise ValueError(f"transcript row missing required field: {field}")
    s = str(val).strip() if not isinstance(val, str) else val.strip()
    if not s:
        raise ValueError(f"transcript row missing required field: {field}")
    return s


def _require_int(row: dict[str, object], field: str) -> int:
    """Return row[field] coerced to int or raise ValueError."""
    val = row.get(field)
    if val is None:
        raise ValueError(f"transcript row missing required field: {field}")
    if not isinstance(val, (int, float, str)):
        raise ValueError(
            f"transcript row field {field!r} must be an integer, got {val!r}"
        )
    try:
        return int(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"transcript row field {field!r} must be an integer, got {val!r}"
        ) from exc


def _row_to_filing_doc(row: dict[str, object]) -> FilingDoc:
    """Convert one JSONL row dict to a FilingDoc, enforcing §6.3 no-NULL rules."""
    ticker = _require_str(row, "ticker")

    quarter = _require_int(row, "quarter")
    if quarter not in {1, 2, 3, 4}:
        raise ValueError(
            f"transcript row field 'quarter' must be 1–4, got {quarter!r}"
        )

    fiscal_year = _require_int(row, "fiscal_year")
    if fiscal_year <= 0:
        raise ValueError(
            f"transcript row field 'fiscal_year' must be > 0, got {fiscal_year!r}"
        )

    # published_at: must be present, non-empty, and tz-aware
    published_at_raw = row.get("published_at")
    if not published_at_raw:
        raise ValueError("transcript row missing required field: published_at")
    published_at_str = str(published_at_raw).strip()
    if not published_at_str:
        raise ValueError("transcript row missing required field: published_at")
    published_at: datetime = datetime.fromisoformat(published_at_str)
    # FilingDoc's Pydantic validator will catch naive timestamps as a second
    # line of defence, but we let it propagate naturally as ValidationError.

    body = _require_str(row, "body")

    doc_id = f"transcript-{ticker}-FY{fiscal_year}-Q{quarter}"
    title = f"{ticker} Q{quarter} FY{fiscal_year} earnings call"

    return FilingDoc(
        doc_id=doc_id,
        ticker=ticker,
        filing_type="transcript",
        published_at=published_at,
        title=title,
        html=body,
        metadata={"quarter": quarter, "fiscal_year": fiscal_year},
    )


class TranscriptsCorpusSource:
    """CorpusSource backed by a local JSONL file of earnings-call transcripts.

    Parameters
    ----------
    path:
        Path to the JSONL file (production mode).  Mutually exclusive with
        *dataset_loader*.
    dataset_loader:
        Callable returning an iterable of raw row dicts (test mode).
        Mutually exclusive with *path*.

    Exactly one of *path* or *dataset_loader* must be provided; supplying
    both or neither raises :class:`ValueError`.
    """

    name = "transcripts"

    def __init__(
        self,
        *,
        path: Path | None = None,
        dataset_loader: Callable[[], Iterable[dict[str, object]]] | None = None,
    ) -> None:
        if path is not None and dataset_loader is not None:
            raise ValueError(
                "TranscriptsCorpusSource: supply path or dataset_loader, not both"
            )
        if path is None and dataset_loader is None:
            raise ValueError(
                "TranscriptsCorpusSource: supply exactly one of path or dataset_loader"
            )
        self._path = path
        self._loader = dataset_loader

    def _load(self) -> Iterable[dict[str, object]]:
        if self._loader is not None:
            return self._loader()
        assert self._path is not None  # guaranteed by __init__
        return _load_jsonl(self._path)

    def iter_docs(self) -> Iterator[FilingDoc]:
        for row in self._load():
            yield _row_to_filing_doc(row)
