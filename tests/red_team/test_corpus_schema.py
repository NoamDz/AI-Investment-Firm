from pathlib import Path
from tests.red_team._corpus import load_corpus

_CORPUS_PATH = Path(__file__).parent / "corpus.jsonl"


def test_corpus_schema() -> None:
    """Validates the red-team corpus loads + every entry conforms to RedTeamCase."""
    cases = load_corpus(_CORPUS_PATH)
    assert len(cases) == 50, f"Expected 50 cases, got {len(cases)}"
    # 5 cases per class × 10 classes
    from collections import Counter
    class_counts = Counter(c.injection_class for c in cases)
    assert all(v == 5 for v in class_counts.values()), f"Uneven per-class counts: {class_counts}"
    assert len(class_counts) == 10, f"Expected 10 classes, got {len(class_counts)}"
    # All case_ids unique
    assert len({c.case_id for c in cases}) == 50, "Duplicate case_ids"
