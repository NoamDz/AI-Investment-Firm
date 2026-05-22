#!/usr/bin/env bash
# scripts/sanitise_plan.sh — strip volatile content from `terraform plan` output.
#
# Usage:
#   terraform -chdir=infra/terraform plan -no-color -var-file=envs/dev.tfvars \
#     | bash scripts/sanitise_plan.sh > infra/terraform/PLAN.md
#
# Transforms (in order):
#   1. AWS account IDs inside ARNs  → <account-id>
#   2. Region tokens inside ARNs    → <region>
#   3. (known after apply)          → <computed>
#   4. ANSI color codes stripped    (defensive — -no-color above should suffice)
#   5. Refresh / read-complete chatter dropped (grep -v)
#   6. ISO-8601 timestamps in warning brackets → [<timestamp>]
#
# Requirements: GNU sed (gawk/sed with -E and \b), available on GitHub Actions
# Ubuntu runners. Not portable to BSD sed (macOS) without modification.
#
# This script ONLY sanitises stdin → stdout. It does NOT prepend a header — the
# PLAN.md header is part of the hand-curated (or freshly-generated) file that
# wraps this pipeline's output.

set -euo pipefail

# ---------------------------------------------------------------------------
# Validate that we are running with GNU sed (GitHub Actions Ubuntu).
# On macOS, brew install gnu-sed and set PATH to include gbrew's bin.
# ---------------------------------------------------------------------------
if ! sed --version 2>/dev/null | grep -q 'GNU'; then
    echo "WARNING: GNU sed not detected. BSD sed may not support -E with \\b." >&2
    echo "         Install gnu-sed or run on a Linux host (e.g. GitHub Actions)." >&2
fi

# ---------------------------------------------------------------------------
# Transform pipeline
#
# Each sed -E expression is kept on its own -e flag for readability and to
# make per-transform disabling trivial during debugging.
#
# Transform notes:
#
#   T1 — Account IDs: match ONLY when the 12-digit block sits at the account
#        position in an ARN: "arn:aws:<service>:<region>:<account>:"
#        Capture group 1 preserves everything up to (but not including) the
#        account digits, so service and region context survive.
#
#   T2 — Region tokens: match ONLY when the region token sits at the region
#        position in an ARN: "arn:aws:<service>:<region-slug>:"
#        Pattern [a-z]{2}-[a-z]+-[0-9]+ covers us-east-1, eu-west-2, etc.
#        Intentionally does NOT touch region tokens outside ARNs (variables,
#        outputs) because those are meaningful plan context.
#
#   T3 — Normalize "(known after apply)" to the shorter "<computed>" so that
#        PLAN.md diffs only show semantic changes, not Terraform verbosity shifts.
#
#   T4 — Strip ANSI escape sequences (color codes). Defensive belt-and-suspenders
#        since -no-color is passed in the usage line above.
#        \x1b (ESC) followed by [ then optional digit/semicolon run then letter.
#        GNU sed supports \x1b in the replacement string; match side uses [].
#
#   T5 — Refresh/read-complete chatter: these lines are timing-dependent and
#        add no semantic content to the plan diff. Dropped via grep -v before
#        the sed pass so sed doesn't need to handle multi-line anchoring.
#
#   T6 — ISO-8601 timestamps appearing inside square brackets in the Terraform
#        warning/error block (e.g. [2026-05-22T14:03:01Z]). Replaced with
#        [<timestamp>] so the plan is deterministic across runs.
# ---------------------------------------------------------------------------

grep -v -E \
    'Refreshing state\.\.\.|Read complete after [0-9]+s' \
| sed -E \
    -e 's|(arn:aws:[a-z0-9-]+:[a-z0-9-]*:)[0-9]{12}|\1<account-id>|g' \
    -e 's|(arn:aws:[a-z0-9-]+:)[a-z]{2}-[a-z]+-[0-9]+|\1<region>|g' \
    -e 's/\(known after apply\)/<computed>/g' \
    -e 's/(\x1b|\033)\[[0-9;]*[A-Za-z]//g' \
    -e 's/\[[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[A-Z]\]/[<timestamp>]/g'
