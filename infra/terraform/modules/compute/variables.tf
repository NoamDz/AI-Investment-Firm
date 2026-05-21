# ---------------------------------------------------------------------------
# modules/compute — input variables (Plan 4 T33)
#
# Most variables are required (no defaults): the orchestrator (../../main.tf)
# is the single source of truth and forwards values from var.* in
# variables.tf. Defaulting here would risk silent divergence between the
# module and its caller.
#
# `container_image` is the one exception: it defaults to the GHCR tag that
# .github/workflows/main.yml's docker-build-and-push job publishes, so a
# bare `terraform plan` resolves a working image. Per-env tfvars may pin to
# an immutable :<sha> tag later.
# ---------------------------------------------------------------------------

variable "project_name" {
  description = "Short project slug used to prefix resource names and Name tags."
  type        = string
}

variable "env" {
  description = "Deployment environment ('dev' or 'prod'); already validated by the orchestrator."
  type        = string
}

variable "ecs_task_cpu" {
  description = "ECS Fargate task CPU units (1024 = 1 vCPU). Forwarded from the orchestrator's var.ecs_task_cpu."
  type        = number
}

variable "ecs_task_memory" {
  description = "ECS Fargate task memory in MiB (2048 = 2 GB). Forwarded from the orchestrator's var.ecs_task_memory."
  type        = number
}

variable "private_subnet_ids" {
  description = "Private subnet IDs the ECS service places tasks into. Forwarded from module.network.private_subnet_ids."
  type        = list(string)
}

variable "ecs_task_security_group_id" {
  description = "Security group ID attached to ECS tasks (egress-only). Forwarded from module.network.ecs_task_security_group_id."
  type        = string
}

variable "container_image" {
  description = "OCI image reference for the firm container. Default matches the :latest tag published by .github/workflows/main.yml's docker-build-and-push job; override per-env to pin an immutable :<sha> tag."
  type        = string
  default     = "ghcr.io/noamdz/ai-investment-firm:latest"
}
