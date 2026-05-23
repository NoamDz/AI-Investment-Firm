#!/usr/bin/env bash
# scripts/sanitise_plan.sh — strip volatile content from `terraform plan` output.
#
# Usage:
#   terraform -chdir=infra/terraform plan -no-color -var-file=envs/dev.tfvars \
#     | bash scripts/sanitise_plan.sh > infra/terraform/PLAN.md
#
# Transforms (in order, all in ONE sed -E call — no grep stage):
#   1. Refresh / read-complete chatter dropped (sed /pattern/d)
#   2. AWS account IDs inside ARNs  → <account-id>
#   3. Region tokens inside ARNs    → <region>
#   4. (known after apply)          → <computed>
#   5. ANSI color codes stripped    (defensive — -no-color above should suffice)
#   6. ISO-8601 timestamps in warning brackets → [<timestamp>]
#
# Why one sed call instead of `grep -v | sed`: grep returns exit 1 when zero
# lines match, and `set -o pipefail` propagates that. A no-op `terraform plan`
# (everything up-to-date) sometimes produces only refresh-chatter lines after
# all other filtering — `grep -v` then drops every line and exits 1, killing
# the pipeline silently. Collapsing into a single sed call sidesteps that.
#
# Requirements: GNU sed (supports -E, \x1b in regex, /pattern/d). Available on
# GitHub Actions Ubuntu runners. Not portable to BSD sed (macOS) without
# modification — the GNU-sed check below hard-fails on mismatch.
#
# This script ONLY sanitises stdin → stdout. It does NOT prepend a header — the
# PLAN.md header is part of the hand-curated (or freshly-generated) file that
# wraps this pipeline's output.

set -euo pipefail

# ---------------------------------------------------------------------------
# Hard-fail if not running on GNU sed. A silently-malformed PLAN.md (e.g.
# under BSD sed where \x1b in the regex matches literal `x1b`, not ESC) is
# worse than a clear CI error pointing operators at the right install step.
# ---------------------------------------------------------------------------
sed_version="$(sed --version 2>/dev/null | head -n 1 || true)"
if ! printf '%s' "$sed_version" | grep -q 'GNU'; then
    printf 'ERROR: GNU sed required. Got: %s\n' "${sed_version:-<unknown / BSD sed>}" >&2
    printf '       On macOS: brew install gnu-sed && export PATH="/opt/homebrew/opt/gnu-sed/libexec/gnubin:$PATH"\n' >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Transform pipeline — single sed -E invocation
#
# Each transform is on its own -e flag for readability and to make per-
# transform disabling trivial during debugging.
#
# Transform notes:
#
#   T1 — Refresh/read-complete chatter: timing-dependent lines that add no
#        semantic content. /pattern/d deletes matching lines in-place. Lives
#        in the same sed call (not a separate grep) so an all-chatter input
#        produces empty output with exit 0 — see header comment for why.
#
#   T2 — Account IDs: match ONLY when the 12-digit block sits at the account
#        position in an ARN: "arn:aws:<service>:<region>:<account>:"
#        Capture group 1 preserves everything up to (but not including) the
#        account digits, so service and region context survive.
#
#   T3 — Region tokens: match ONLY when the region token sits at the region
#        position in an ARN: "arn:aws:<service>:<region-slug>:"
#        Pattern [a-z]{2}-[a-z]+-[0-9]+ covers us-east-1, eu-west-2, etc.
#        Intentionally does NOT touch region tokens outside ARNs (variables,
#        outputs) because those are meaningful plan context.
#
#   T4 — Normalize "(known after apply)" to the shorter "<computed>" so that
#        PLAN.md diffs only show semantic changes, not Terraform verbosity shifts.
#
#   T5 — Strip ANSI escape sequences (color codes). Defensive belt-and-suspenders
#        since -no-color is passed in the usage line above.
#        \x1b (ESC) followed by [ then optional digit/semicolon run then letter.
#        GNU sed supports \x1b in the regex; \033 is the octal equivalent.
#
#   T6 — ISO-8601 timestamps appearing inside square brackets in the Terraform
#        warning/error block (e.g. [2026-05-22T14:03:01Z]). Replaced with
#        [<timestamp>] so the plan is deterministic across runs.
# ---------------------------------------------------------------------------

sed -E \
    -e '/Refreshing state\.\.\.|Read complete after [0-9]+s/d' \
    -e 's|(arn:aws:[a-z0-9-]+:[a-z0-9-]*:)[0-9]{12}|\1<account-id>|g' \
    -e 's|(arn:aws:[a-z0-9-]+:)[a-z]{2}-[a-z]+-[0-9]+|\1<region>|g' \
    -e 's/\(known after apply\)/<computed>/g' \
    -e 's/(\x1b|\033)\[[0-9;]*[A-Za-z]//g' \
    -e 's/\[[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[A-Z]\]/[<timestamp>]/g'
