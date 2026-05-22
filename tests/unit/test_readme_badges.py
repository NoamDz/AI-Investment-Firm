"""Plan 4 T30.2 — Validate README CI badges resolve to real workflow files.

Each ![...badge.svg](.../workflows/<NAME>/badge.svg) embedded in README.md must
point to a workflow file that actually exists at .github/workflows/<NAME>.
Catches accidental badge removal (count gate) and accidental workflow rename
(existence gate) before they reach main.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BADGE_RE = re.compile(
    r"!\[[^\]]*\]\(https://github\.com/[^/]+/[^/]+/actions/workflows/([^/]+)/badge\.svg\)"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"


def _badges() -> list[str]:
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    return _BADGE_RE.findall(readme)


def test_readme_has_at_least_three_workflow_badges() -> None:
    badges = _badges()
    assert len(badges) >= 3, (
        f"Expected at least 3 CI workflow badges in README.md, found {len(badges)}: {badges}"
    )


@pytest.mark.parametrize("workflow_name", _badges())
def test_badge_workflow_exists(workflow_name: str) -> None:
    path = _WORKFLOWS_DIR / workflow_name
    assert path.exists(), (
        f"README badge points to {workflow_name} but {path} does not exist"
    )
