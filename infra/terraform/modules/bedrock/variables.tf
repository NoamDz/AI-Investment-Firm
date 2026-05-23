# ---------------------------------------------------------------------------
# modules/bedrock — input variables (Plan 4 T36)
#
# All variables except log_retention_days are required (no defaults): the
# orchestrator (../../main.tf) is the single source of truth and forwards
# values from module.secrets outputs. Defaulting here would risk silent
# divergence when the root wiring is updated.
# ---------------------------------------------------------------------------

variable "project_name" {
  description = "Short project slug used to prefix resource names (IAM role, CW log group)."
  type        = string
}

variable "env" {
  description = "Deployment environment ('dev' or 'prod'); already validated by the orchestrator."
  type        = string
}

variable "hmac_secret_arn" {
  description = "ARN of firm/firm_hmac_secret in Secrets Manager (T35). Granted to the AgentCore runtime IAM role for Identity provider signature verification."
  type        = string
}

variable "hmac_secret_prev_arn" {
  description = "ARN of firm/firm_hmac_secret_prev in Secrets Manager (T35). Required by the Identity provider during HMAC key rotation to validate requests signed with the previous key."
  type        = string
}

variable "hmac_rotated_at_arn" {
  description = "ARN of firm/firm_hmac_rotated_at in Secrets Manager (T35). Timestamp sentinel used by the Identity provider to determine which HMAC key version to apply."
  type        = string
}

variable "secrets_kms_key_arn" {
  description = "ARN of the customer-managed KMS key (T35) that encrypts the firm/* secrets. The AgentCore runtime needs kms:Decrypt to read the HMAC secrets."
  type        = string
}

variable "log_retention_days" {
  description = "CloudWatch log group retention in days. Defaults to 90 to match the ECS task log group in modules/compute (T33)."
  type        = number
  default     = 90
}
