"""
Integration tests for scripts/check_reports_clean.sh.

The script discovers the repo root by walking up from BASH_SOURCE[0] looking
for the directory that contains the Makefile.  To exercise it in isolation we:
  - Copy the real script into tmp_path/scripts/
  - Drop a stub Makefile in tmp_path/ (so the repo-root detection works)
  - Write a small wrapper shell script that implements the fake eval command,
    then point FIRM_EVAL_CMD at it (avoids MSYS2/Git-Bash env-var mangling of
    values that contain shell metacharacters like && on Windows).
  - Run `bash scripts/check_reports_clean.sh` with cwd=tmp_path

On Windows with WSL2, the `bash` on PATH may be the WSL2 bash which does NOT
inherit arbitrary Windows env vars unless they appear in WSLENV.  We always
add FIRM_EVAL_CMD to WSLENV so it crosses the WSL boundary.  On Linux, WSLENV
is ignored, so this is harmless.

On Windows, bash (Git-Bash / WSL) cannot accept Windows-style absolute paths
as a script argument.  We work around this by running with cwd=tmp_path and
passing the script as a relative POSIX path ("scripts/check_reports_clean.sh").
"""

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent  # worktree root
SCRIPT_SRC = REPO_ROOT / "scripts" / "check_reports_clean.sh"


def _to_posix_bash(p: Path) -> str:
    """Convert a Windows absolute path to a form WSL/Git-Bash can accept.

    E.g. C:\\Users\\foo\\bar → /mnt/c/Users/foo/bar  (WSL convention)
    Relative paths are left untouched.
    """
    s = str(p)
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return s.replace("\\", "/")


def _setup_sandbox(tmp_path: Path) -> None:
    """Copy the script into tmp_path/scripts/ and create a stub Makefile."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copy2(SCRIPT_SRC, scripts_dir / "check_reports_clean.sh")
    # Stub Makefile so the repo-root detection in the script finds it.
    (tmp_path / "Makefile").write_text("# stub\n")


def _write_fake_eval_script(tmp_path: Path, body: str) -> str:
    """Write a bash script that acts as the fake FIRM_EVAL_CMD.

    Returns the path in a form that bash can use as a command
    (i.e. a POSIX path that Git-Bash/WSL understands on Windows).
    The file is written with Unix (LF-only) line endings so that WSL bash
    does not choke on CRLF when running on Windows.
    """
    fake = tmp_path / "fake_eval.sh"
    content = f"#!/usr/bin/env bash\nset -euo pipefail\n{body}\n"
    # Use binary mode to force LF-only line endings (Python text mode writes
    # CRLF on Windows, which WSL bash rejects with "invalid option name").
    fake.write_bytes(content.encode("utf-8").replace(b"\r\n", b"\n"))
    return _to_posix_bash(fake)


def _make_env(eval_cmd: str) -> dict:  # type: ignore[type-arg]
    """Build an env dict that ensures FIRM_EVAL_CMD crosses the WSL boundary."""
    existing_wslenv = os.environ.get("WSLENV", "")
    if existing_wslenv:
        new_wslenv = existing_wslenv + ":FIRM_EVAL_CMD"
    else:
        new_wslenv = "FIRM_EVAL_CMD"
    return {**os.environ, "FIRM_EVAL_CMD": eval_cmd, "WSLENV": new_wslenv}


def _run(eval_cmd: str, cwd: Path) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    """Run check_reports_clean.sh via a relative path so bash can find it."""
    return subprocess.run(
        ["bash", "scripts/check_reports_clean.sh"],
        env=_make_env(eval_cmd),
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


# ---------------------------------------------------------------------------
# Test 1: deterministic eval → exit 0
# ---------------------------------------------------------------------------

def test_deterministic_eval_passes(tmp_path: Path) -> None:
    """A FIRM_EVAL_CMD that always writes the same content must exit 0."""
    _setup_sandbox(tmp_path)

    # Write fake eval script: always writes the same file content.
    fake_path = _write_fake_eval_script(
        tmp_path,
        "mkdir -p reports/eval\necho hello > reports/eval/run.txt",
    )
    eval_cmd = f"bash {fake_path}"

    result = _run(eval_cmd, tmp_path)

    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"Expected exit 0 for deterministic eval, got {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "deterministic" in combined.lower(), (
        f"Expected 'deterministic' in output.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 2: non-deterministic eval → exit 1 with diff snippet
# ---------------------------------------------------------------------------

def test_nondeterministic_eval_fails_with_diff(tmp_path: Path) -> None:
    """A FIRM_EVAL_CMD that writes different content each run must exit 1."""
    _setup_sandbox(tmp_path)

    # Write fake eval script: $RANDOM + $BASHPID changes every invocation.
    fake_path = _write_fake_eval_script(
        tmp_path,
        "mkdir -p reports/eval\n"
        'printf \'%s\\n\' "${RANDOM:-$$}_${BASHPID}" > reports/eval/run.txt',
    )
    eval_cmd = f"bash {fake_path}"

    result = _run(eval_cmd, tmp_path)

    combined = result.stdout + result.stderr
    assert result.returncode == 1, (
        f"Expected exit 1 for non-deterministic eval, got {result.returncode}.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "non-deterministic" in combined.lower(), (
        f"Expected 'non-deterministic' in output.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # The diff snippet should be present (at minimum the "---" / "+++" lines).
    assert "---" in combined or "+++" in combined or "@@" in combined, (
        f"Expected diff output in stdout.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
