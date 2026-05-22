"""Sufficiency-gate dev-set loader + precision/recall computation (Plan 4 §T12).

Replaces the ``(1.0, 1.0)`` stub previously hard-coded in ``runner.py`` with a
real measurement against ``tests/fixtures/sufficiency_dev_set.jsonl``.

Definitions
-----------
* **Precision** — of the citations the system actually produced (over all dev
  claims), what fraction are present in the union of ``gold_citations``.
* **Recall** — of the union of ``gold_citations`` in the dev set, what fraction
  did the system produce.

The metric is computed over ``(claim_id, chunk_id)`` pairs: a chunk_id that
appears in the dev set under a *different* claim_id does NOT count as a true
positive. This avoids cross-claim leakage from passing the gate spuriously.

Empty-input semantics (deliberate; see ``runner._wire_sufficiency_metrics``):

  * ``dev_set`` empty   → ``(0.0, 0.0)`` — there is nothing to measure;
    surfacing zero forces the gate to fail rather than vacuously pass.
  * ``produced`` empty  → precision is undefined; we report ``0.0`` so the
    gate fails. The runner protects against the "no decisions had citations
    in this regime" case by falling back to ``(1.0, 1.0)`` BEFORE calling
    here (see runner docstring).
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SufficiencyDevCase:
    """One labeled dev-set claim with its expected citations.

    ``gold_citations`` is the set of chunk IDs that a correct system would
    cite for this claim. The order does not matter — comparisons are
    set-based.
    """

    claim_id: str
    text: str
    gold_citations: tuple[str, ...]


def load_dev_set(path: Path) -> list[SufficiencyDevCase]:
    """Load the JSONL dev set at *path*.

    Each line must be a JSON object with keys ``claim_id`` (str), ``text``
    (str), and ``gold_citations`` (list[str]). Blank lines are skipped.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If any line is missing one of the required keys.
    json.JSONDecodeError
        If any non-blank line is not valid JSON.
    """
    cases: list[SufficiencyDevCase] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            cases.append(
                SufficiencyDevCase(
                    claim_id=str(obj["claim_id"]),
                    text=str(obj["text"]),
                    gold_citations=tuple(str(c) for c in obj["gold_citations"]),
                )
            )
    return cases


def compute_sufficiency_metrics(
    dev_set: Sequence[SufficiencyDevCase],
    produced_citations: Mapping[str, Iterable[str]],
) -> tuple[float, float]:
    """Return ``(precision, recall)`` over *dev_set* given the produced
    citations.

    Parameters
    ----------
    dev_set : labeled claims with gold citations.
    produced_citations : mapping of ``claim_id → iterable of chunk_ids the
        system actually produced for that claim``. Missing claim_ids are
        treated as "produced nothing".

    Returns
    -------
    (precision, recall) : both in ``[0.0, 1.0]``. See module docstring for
        empty-input semantics.
    """
    if not dev_set:
        return 0.0, 0.0

    all_produced: set[tuple[str, str]] = set()
    all_gold: set[tuple[str, str]] = set()
    for case in dev_set:
        for chunk_id in case.gold_citations:
            all_gold.add((case.claim_id, chunk_id))
        for chunk_id in produced_citations.get(case.claim_id, ()):
            all_produced.add((case.claim_id, chunk_id))

    true_pos = len(all_produced & all_gold)
    precision = true_pos / len(all_produced) if all_produced else 0.0
    recall = true_pos / len(all_gold) if all_gold else 0.0
    return precision, recall


# Default path for the canonical dev set fixture. Resolved relative to repo
# root (this file lives at ``firm/eval/sufficiency.py``; ``parents[2]`` is
# the repo root).
DEFAULT_DEV_SET_PATH: Path = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "sufficiency_dev_set.jsonl"
)


__all__ = [
    "DEFAULT_DEV_SET_PATH",
    "SufficiencyDevCase",
    "compute_sufficiency_metrics",
    "load_dev_set",
]
