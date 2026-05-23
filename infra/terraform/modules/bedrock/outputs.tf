# ---------------------------------------------------------------------------
# modules/bedrock — outputs (Plan 4 T36)
#
# Consumed by:
#   * T39 AgentCore CLI — agentcore_runtime_role_arn, agentcore_runtime_name,
#     agentcore_memory_namespace, agentcore_identity_secret_arn are passed as
#     arguments to `aws bedrock-agentcore create-runtime` and related commands.
#   * Operator runbooks — log group name for `aws logs tail` during debug.
# ---------------------------------------------------------------------------

output "agentcore_runtime_role_arn" {
  description = "ARN of the IAM role the AgentCore Runtime service assumes. Pass to `aws bedrock-agentcore create-runtime --execution-role-arn` in T39."
  value       = aws_iam_role.agentcore_runtime.arn
}

output "agentcore_runtime_role_name" {
  description = "Name of the AgentCore Runtime IAM role (short form, for policy attachment and console navigation)."
  value       = aws_iam_role.agentcore_runtime.name
}

output "agentcore_runtime_name" {
  description = "Symbolic AgentCore runtime name ('firm-reporter'). Consumed by T39 CLI commands and docs/agentcore_mapping.md."
  value       = local.agentcore_runtime_name
}

output "agentcore_memory_namespace" {
  description = "AgentCore Memory namespace ('firm-desk-state'). Consumed by T39 CLI commands to attach the shared desk state store to the reporter runtime."
  value       = local.agentcore_memory_namespace
}

output "agentcore_log_group_name" {
  description = "CloudWatch log group name for AgentCore Reporter stdout/stderr. Use with `aws logs tail` or the CW console."
  value       = aws_cloudwatch_log_group.agentcore_reporter.name
}

output "agentcore_identity_secret_arn" {
  description = "ARN of firm/firm_hmac_secret — the primary HMAC signing secret linked to the AgentCore Identity provider. Passed through from the hmac_secret_arn variable so consumers can read it from the bedrock module output rather than re-plumbing from the secrets module."
  value       = var.hmac_secret_arn
}
