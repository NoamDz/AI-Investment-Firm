"""Link-checker for ``docs/threat_model.md`` (Plan 4 T43).

Parses every markdown link target ``[text](target)`` in the doc and asserts
that the target either:

* points at an existing path on disk (relative to the doc), OR
* is in :data:`KNOWN_FORWARD_REFS` — paths that are valid forward references
  even when not yet on disk, OR
* is an http(s) URL (skipped — we don't reach out to the network in unit
  tests), OR
* is a same-doc anchor link (starts with ``#``).

Also asserts the doc is non-trivially long (>50 lines).
"""
from __future__ import annotations

import re
from pathlib import Path

# No forward references expected for threat_model.md — every linked path
# should exist on disk at the time the doc is written.
KNOWN_FORWARD_REFS: frozenset[str] = frozenset()

_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")


def _repo_root() -> Path:
    # tests/unit/test_docs_threat_model_links.py  →  tests/unit  →  tests  →  repo root
    return Path(__file__).resolve().parents[2]


def _doc_path() -> Path:
    return _repo_root() / "docs" / "threat_model.md"


def _strip_anchor(target: str) -> str:
    """Strip ``#anchor`` from a link target — we only verify the path part."""
    return target.split("#", 1)[0]


def test_docs_threat_model_md_is_non_trivially_long() -> None:
    doc = _doc_path()
    assert doc.exists(), f"{doc} does not exist"
    line_count = sum(1 for _ in doc.read_text(encoding="utf-8").splitlines())
    assert line_count > 50, f"docs/threat_model.md too short: {line_count} lines"


def test_docs_threat_model_md_links_resolve() -> None:
    doc = _doc_path()
    text = doc.read_text(encoding="utf-8")
    doc_dir = doc.parent

    missing: list[tuple[str, str]] = []
    for match in _LINK_RE.finditer(text):
        target = match.group("target").strip()

        # Skip URLs and pure anchor links.
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue

        # Known forward references — paths not yet on disk.
        if target in KNOWN_FORWARD_REFS:
            continue

        path_part = _strip_anchor(target)
        if not path_part:
            continue

        resolved = (doc_dir / path_part).resolve()
        if not resolved.exists():
            missing.append((match.group("text"), target))

    assert not missing, (
        "docs/threat_model.md has unresolved link targets:\n"
        + "\n".join(f"  [{text}]({target})" for text, target in missing)
    )
