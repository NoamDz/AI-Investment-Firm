# ---------------------------------------------------------------------------
# modules/secrets — outputs (Plan 4 T35)
#
# Consumed by:
#   * modules/compute IAM policy — already grants firm/* via wildcard ARN;
#     secret_arns is exposed here for future tightening (T44 prod hardening).
#   * modules/storage / modules/observability — kms_key_arn for future grants
#     allowing those modules to encrypt/decrypt with this CMK.
#   * Operator runbooks — secret_names for deterministic iteration in scripts.
# ---------------------------------------------------------------------------

output "secret_arns" {
  description = "Map of secret name to ARN for all six Secrets Manager entries (consumed by compute IAM policy tightening in T44)."
  value       = { for k, s in aws_secretsmanager_secret.this : k => s.arn }
}

output "secret_names" {
  description = "Sorted list of Secrets Manager secret names — deterministic ordering for operator scripts and downstream iteration."
  value       = sort([for s in aws_secretsmanager_secret.this : s.name])
}

output "kms_key_arn" {
  description = "ARN of the customer-managed KMS key used for secrets encryption-at-rest (for future cross-module kms:Decrypt grants)."
  value       = aws_kms_key.secrets.arn
}

output "kms_key_id" {
  description = "Key ID of the customer-managed KMS key (short form, for use in policies that accept either ARN or key ID)."
  value       = aws_kms_key.secrets.key_id
}

output "kms_alias_name" {
  description = "KMS alias name (alias/<project>-<env>-secrets) for human-readable references in the console and runbooks."
  value       = aws_kms_alias.secrets.name
}
