# ---------------------------------------------------------------------------
# modules/observability — outputs (Plan 4 T37)
#
# Consumed by downstream tasks / operators:
#   * firm_log_group_name / _arn — the Reporter agent (T40) may write OTLP
#     telemetry here; the T38 PLAN.md references the ARN for policy docs.
#   * otelcol_log_group_name — operator reference for `aws logs tail` on
#     collector startup errors.
#   * otelcol_service_name — operator reference for `aws ecs update-service`.
#   * otelcol_task_role_arn — future cross-account telemetry forwarding (T44).
#   * dashboard_arn — operators can bookmark the console URL from this ARN.
# ---------------------------------------------------------------------------

output "firm_log_group_name" {
  description = "Name of the /firm/<env> CloudWatch log group — application telemetry export target for the OTLP awscloudwatchlogs exporter."
  value       = aws_cloudwatch_log_group.firm.name
}

output "firm_log_group_arn" {
  description = "ARN of the /firm/<env> CloudWatch log group. Used in IAM policy scoping for any service that writes telemetry here (e.g. Reporter agent T40)."
  value       = aws_cloudwatch_log_group.firm.arn
}

output "otelcol_log_group_name" {
  description = "Name of the /ecs/<project>-<env>-otelcol CloudWatch log group — collector container stdout/stderr (boot messages, exporter errors)."
  value       = aws_cloudwatch_log_group.otelcol.name
}

output "otelcol_service_name" {
  description = "Name of the ECS service running the OTLP collector. Operators use this with `aws ecs update-service` to force a new deployment."
  value       = aws_ecs_service.otelcol.name
}

output "otelcol_task_role_arn" {
  description = "ARN of the IAM role assumed by the collector process. Exposed for future cross-account telemetry forwarding (path-to-prod T44)."
  value       = aws_iam_role.otelcol_task.arn
}

output "dashboard_arn" {
  description = "ARN of the CloudWatch dashboard. Operators can construct the console URL as: https://<region>.console.aws.amazon.com/cloudwatch/home#dashboards:name=<dashboard_name>."
  value       = aws_cloudwatch_dashboard.firm.dashboard_arn
}
