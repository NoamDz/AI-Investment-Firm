#!/usr/bin/env bash
# check_reports_clean.sh
#
# WHAT: Verifies that running the eval pipeline twice produces identical output,
#       i.e. that report generation is fully deterministic (no timestamps, random
#       seeds, or ordering hazards leaking into reports/eval/).
# WHY:  Non-deterministic outputs break reproducible research, make CI diffs
#       meaningless, and signal hidden state (time, randomness, ordering) that
#       can mask regressions.
# HOW:  Run FIRM_EVAL_CMD twice into separate temp snapshots, diff them.  A
#       non-empty diff prints the first 50 lines and exits 1.  A missing
#       reports/eval/ after the first run exits 2 (misconfigured pipeline).
#
# FIRM_EVAL_CMD (env var, optional): command to run instead of "make eval".
#   Example: FIRM_EVAL_CMD='mkdir -p reports/eval && echo hi > reports/eval/out.txt'
#   This lets tests substitute a lightweight fake without requiring T15 to exist.

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root: the directory that contains the Makefile.
# BASH_SOURCE[0] is this script; go two levels up (script → scripts/ → root).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

EVAL_CMD="${FIRM_EVAL_CMD:-make eval}"
REPORTS_DIR="reports/eval"
DIFF_FILE="/tmp/firm-determinism-diff.txt"

# ---------------------------------------------------------------------------
# Temp dirs — cleaned up on exit regardless of outcome.
# ---------------------------------------------------------------------------
T1="$(mktemp -d)"
T2="$(mktemp -d)"
trap 'rm -rf "$T1" "$T2"' EXIT

# ---------------------------------------------------------------------------
# First run
# ---------------------------------------------------------------------------
echo "==> Run 1: ${EVAL_CMD}"
eval "${EVAL_CMD}"

if [[ ! -d "${REPORTS_DIR}" ]]; then
    echo "ERROR: FIRM_EVAL_CMD did not produce ${REPORTS_DIR}/ — is \`make eval\` defined?" >&2
    exit 2
fi

cp -a "${REPORTS_DIR}/." "${T1}/"

# ---------------------------------------------------------------------------
# Second run
# ---------------------------------------------------------------------------
echo "==> Run 2: ${EVAL_CMD}"
eval "${EVAL_CMD}"

if [[ ! -d "${REPORTS_DIR}" ]]; then
    echo "ERROR: FIRM_EVAL_CMD did not produce ${REPORTS_DIR}/ on second run." >&2
    exit 2
fi

cp -a "${REPORTS_DIR}/." "${T2}/"

# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------
if ! diff -ruN "${T1}" "${T2}" > "${DIFF_FILE}"; then
    echo ""
    echo "Eval is non-deterministic. First 50 lines of diff:"
    echo "----------------------------------------------------"
    head -n 50 "${DIFF_FILE}"
    exit 1
fi

echo ""
echo "✓ reports/eval/ is deterministic across two runs."
exit 0
