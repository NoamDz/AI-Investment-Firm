# ---------------------------------------------------------------------------
# modules/secrets (Plan 4 T35)
#
# Secrets layer for the firm:
#   * 1 customer-managed KMS key (CMK) with annual rotation enabled — used
#     to encrypt every Secrets Manager entry at rest.
#   * 1 KMS alias for human-readable references in the console / CLI.
#   * 6 AWS Secrets Manager secrets (placeholder entries only — actual values
#     are written out-of-band by operators via aws-cli or the console):
#       - firm/anthropic_api_key
#       - firm/slack_signing_secret
#       - firm/slack_bot_token
#       - firm/firm_hmac_secret
#       - firm/firm_hmac_secret_prev
#       - firm/firm_hmac_rotated_at
#
# IAM grant contract:
#   modules/compute task IAM policy already grants
#   secretsmanager:GetSecretValue on arn:aws:secretsmanager:*:*:secret:firm/*
#   All six names below match that wildcard — no compute changes needed.
#
# Out of scope for T35:
#   * Cross-module KMS grant wiring (storage, compute) — T37 / T44.
#   * Secret rotation Lambdas — values are rotated out-of-band.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# KMS — customer-managed key for encryption-at-rest of Secrets Manager entries
#
# No custom key_policy block: the default key policy grants the root account
# full key access, and IAM policies on consumer principals control the rest.
# Adding a custom policy without a complete list of consumers risks lockout.
# ---------------------------------------------------------------------------

resource "aws_kms_key" "secrets" {
  description             = "${var.project_name}-${var.env} secrets encryption key"
  deletion_window_in_days = var.kms_key_deletion_window_days

  # CIS benchmark requirement: enable automatic annual key-material rotation.
  # AWS KMS rotates the backing key material every year; existing ciphertext
  # remains decryptable with the old material, so no re-encryption is needed.
  enable_key_rotation = true
}

# Alias provides a stable, human-readable name for the key in the AWS console
# and CLI without exposing the raw key ID in operator runbooks.
resource "aws_kms_alias" "secrets" {
  name          = "alias/${var.project_name}-${var.env}-secrets"
  target_key_id = aws_kms_key.secrets.key_id
}

# ---------------------------------------------------------------------------
# Secrets Manager — placeholder entries
#
# WHY for_each over a local set: all six secrets share identical configuration
# (CMK, description template, no version). A single resource with for_each is
# cleaner than six identical blocks and makes adding secrets a one-liner.
#
# WHY no aws_secretsmanager_secret_version resources:
#   The spec says "placeholders only; actual values rotated out-of-band."
#   Writing a version resource with dummy ciphertext would:
#     1. Commit a fake value to Terraform state, visible in `terraform show`
#        and plan diffs (mild OPSEC leak).
#     2. Persist a sentinel that operators might forget to replace before
#        going live.
#     3. Require `ignore_changes = [secret_string]` to avoid drift when the
#        real value is written via aws-cli / secrets rotation Lambda.
#   Operators write the initial secret value once after the first `apply`:
#     aws secretsmanager put-secret-value \
#       --secret-id firm/anthropic_api_key \
#       --secret-string '{"value":"sk-ant-..."}'
# ---------------------------------------------------------------------------

locals {
  secret_names = toset([
    "firm/anthropic_api_key",
    "firm/slack_signing_secret",
    "firm/slack_bot_token",
    "firm/firm_hmac_secret",
    "firm/firm_hmac_secret_prev",
    "firm/firm_hmac_rotated_at",
  ])
}

# Each entry is encrypted with the CMK above. recovery_window_in_days is left
# at the AWS default (30 days) — do NOT set to 0, which would enable immediate
# destroy and eliminate the recovery window for accidental deletions.
resource "aws_secretsmanager_secret" "this" {
  for_each    = local.secret_names
  name        = each.key
  description = "${var.project_name}-${var.env} secret — value written out-of-band by operators"
  kms_key_id  = aws_kms_key.secrets.arn
}
