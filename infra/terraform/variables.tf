# ---------------------------------------------------------------------------
# Input variables (Plan 4 T31).
#
# Order: env-required first, then naming, then network, then storage, then
# compute. Every block has description -> type -> default (if any) ->
# validation (if any). Values are tuned for the take-home: defaults match
# `envs/dev.tfvars` so a bare `terraform plan -var env=dev` works; prod sizes
# come from `envs/prod.tfvars` overrides.
# ---------------------------------------------------------------------------

variable "env" {
  description = "Deployment environment. Must be 'dev' or 'prod' — gates module sizing and tagging."
  type        = string

  validation {
    condition     = contains(["dev", "prod"], var.env)
    error_message = "env must be one of: dev, prod."
  }
}

variable "project_name" {
  description = "Short project slug used to prefix resource names and as the `project` tag value."
  type        = string
  default     = "firm"
}

variable "region" {
  description = "AWS region for every module's resources. Single-region by design for the take-home."
  type        = string
  default     = "us-east-1"
}

variable "vpc_cidr" {
  description = "Top-level VPC CIDR consumed by modules/network (T32) for subnet carving."
  type        = string
  default     = "10.0.0.0/16"
}

variable "db_instance_class" {
  description = "RDS Postgres instance class consumed by modules/storage (T34); prod tfvars upgrades to db.r6g.large."
  type        = string
  default     = "db.t4g.micro"
}

variable "ecs_task_cpu" {
  description = "ECS Fargate task CPU units consumed by modules/compute (T33); 1024 = 1 vCPU."
  type        = number
  default     = 1024
}

variable "ecs_task_memory" {
  description = "ECS Fargate task memory (MiB) consumed by modules/compute (T33); 2048 = 2 GB."
  type        = number
  default     = 2048
}
