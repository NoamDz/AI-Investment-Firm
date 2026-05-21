# ---------------------------------------------------------------------------
# modules/storage — outputs (Plan 4 T34)
#
# Consumed by downstream modules / docs:
#   * modules/secrets (T35)       — db_master_user_secret_arn (to grant the
#                                    task role kms:Decrypt on the auto-
#                                    rotated secret's KMS key)
#   * modules/observability (T37) — bucket ARNs (for traces export targets)
#   * docs/path-to-production.md  — db_endpoint for manual smoke-test
# ---------------------------------------------------------------------------

output "reports_bucket_name" {
  description = "Name of the reports S3 bucket (matches the IAM grant in modules/compute task policy)."
  value       = aws_s3_bucket.this["reports"].id
}

output "traces_bucket_name" {
  description = "Name of the traces S3 bucket (90-day lifecycle expiration)."
  value       = aws_s3_bucket.this["traces"].id
}

output "cassettes_bucket_name" {
  description = "Name of the eval cassettes S3 bucket (no lifecycle — cassettes are reproducibility artifacts)."
  value       = aws_s3_bucket.this["cassettes"].id
}

output "reports_bucket_arn" {
  description = "ARN of the reports S3 bucket."
  value       = aws_s3_bucket.this["reports"].arn
}

output "traces_bucket_arn" {
  description = "ARN of the traces S3 bucket."
  value       = aws_s3_bucket.this["traces"].arn
}

output "cassettes_bucket_arn" {
  description = "ARN of the eval cassettes S3 bucket."
  value       = aws_s3_bucket.this["cassettes"].arn
}

output "db_endpoint" {
  description = "RDS Postgres endpoint (hostname:port) for application connection strings."
  value       = aws_db_instance.this.endpoint
}

output "db_name" {
  description = "Postgres database name created on the instance (hyphens in project_name are replaced with underscores)."
  value       = aws_db_instance.this.db_name
}

output "db_master_user_secret_arn" {
  description = "ARN of the Secrets Manager secret holding the auto-rotated master password (only valid because manage_master_user_password = true)."
  value       = aws_db_instance.this.master_user_secret[0].secret_arn
}
