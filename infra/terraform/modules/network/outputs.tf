# ---------------------------------------------------------------------------
# modules/network — outputs (Plan 4 T32)
#
# Consumed by downstream modules:
#   * modules/compute (T33) — needs vpc_id, private_subnet_ids,
#                              ecs_task_security_group_id
#   * modules/storage (T34) — needs private_subnet_ids,
#                              rds_security_group_id
#   * modules/observability (T37) — needs otlp_security_group_id
# ---------------------------------------------------------------------------

output "vpc_id" {
  description = "ID of the VPC created by this module."
  value       = aws_vpc.this.id
}

output "public_subnet_ids" {
  description = "IDs of the two public subnets (AZ-0, AZ-1) — for the NAT and any future public-facing load balancers."
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the two private subnets (AZ-0, AZ-1) — for ECS tasks, RDS, and the OTLP collector."
  value       = aws_subnet.private[*].id
}

output "ecs_task_security_group_id" {
  description = "Security group ID to attach to ECS Fargate tasks (egress-only)."
  value       = aws_security_group.ecs_task.id
}

output "rds_security_group_id" {
  description = "Security group ID to attach to the RDS Postgres instance (ingress 5432 from ECS only)."
  value       = aws_security_group.rds.id
}

output "otlp_security_group_id" {
  description = "Security group ID to attach to the OTLP collector (ingress 4317 + 4318 from ECS only)."
  value       = aws_security_group.otlp_collector.id
}
