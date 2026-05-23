from pathlib import Path
from tests.red_team._corpus import load_corpus

_CORPUS_PATH = Path(__file__).parent / "corpus.jsonl"


def test_corpus_schema() -> None:
    """Validates the red-team corpus loads + every entry conforms to RedTeamCase.

    Per Plan 4 Section B T07.h (Bundle C), the baseline 50 = 5 cases per
    class x 10 classes is extended by one PM-targeted ``citation_forgery_6``
    case so the UNCITED_CLAIM end-to-end assertion has a real triggering
    fixture.  The citation_forgery class therefore has 6 cases; every
    other class still has exactly 5.
    """
    cases = load_corpus(_CORPUS_PATH)
    assert len(cases) == 51, f"Expected 51 cases, got {len(cases)}"
    # 10 classes, 5 each except citation_forgery which has 6 (T07.h).
    from collections import Counter
    class_counts = Counter(c.injection_class for c in cases)
    assert len(class_counts) == 10, f"Expected 10 classes, got {len(class_counts)}"
    assert class_counts["citation_forgery"] == 6, (
        f"citation_forgery should have 6 cases (5 baseline + T07.h "
        f"PM-targeted UNCITED_CLAIM case); got {class_counts['citation_forgery']}"
    )
    for cls, count in class_counts.items():
        if cls == "citation_forgery":
            continue
        assert count == 5, f"Class {cls!r} should have 5 cases, got {count}"
    # All case_ids unique
    assert len({c.case_id for c in cases}) == 51, "Duplicate case_ids"
