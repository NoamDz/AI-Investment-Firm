# ---------------------------------------------------------------------------
# modules/compute — outputs (Plan 4 T33)
#
# Consumed by downstream modules / docs:
#   * modules/observability (T37) — log_group_name for dashboards + alarms;
#                                    cluster_name + service_name for CW
#                                    Container Insights metric dimensions.
#   * modules/secrets (T35) — task_role_arn may appear in KMS key policies
#                              so only this role can decrypt secret values.
#   * docs/path-to-production.md (T44) — task_execution_role_arn for
#                                         operators wiring image-pull policies.
# ---------------------------------------------------------------------------

output "cluster_name" {
  description = "Name of the ECS Fargate cluster."
  value       = aws_ecs_cluster.this.name
}

output "service_name" {
  description = "Name of the ECS service running the firm task."
  value       = aws_ecs_service.this.name
}

output "task_role_arn" {
  description = "ARN of the IAM role assumed by the running container (application-level permissions: Secrets Manager, Bedrock, S3, CW Logs)."
  value       = aws_iam_role.task.arn
}

output "task_execution_role_arn" {
  description = "ARN of the IAM role assumed by ECS itself to pull the image and ship task-stdout logs."
  value       = aws_iam_role.task_execution.arn
}

output "log_group_name" {
  description = "CloudWatch log group name receiving task-stdout streams (one stream per task, prefixed 'ecs')."
  value       = aws_cloudwatch_log_group.ecs.name
}
