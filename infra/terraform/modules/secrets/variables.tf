# ---------------------------------------------------------------------------
# modules/secrets — variables (Plan 4 T35)
#
# All values passed down from the root orchestrator (infra/terraform/main.tf).
# No region variable — region is inherited from the provider, not the module.
# ---------------------------------------------------------------------------

variable "project_name" {
  description = "Short project slug used to prefix every resource name (KMS alias, secret description)."
  type        = string
}

variable "env" {
  description = "Deployment environment ('dev' or 'prod'). Included in KMS alias and secret descriptions for at-a-glance identification."
  type        = string
}

variable "kms_key_deletion_window_days" {
  description = "KMS key deletion window in days (7–30). Default 30 is the AWS maximum — longest recovery window for accidental destroys. Operators can lower this for test environments."
  type        = number
  default     = 30
}
