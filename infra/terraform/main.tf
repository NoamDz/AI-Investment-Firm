# ---------------------------------------------------------------------------
# Top-level Terraform orchestrator (Plan 4 T31).
#
# This file is intentionally near-empty today: it declares the `locals` map
# used by every downstream module and reserves space (as a TODO outline) for
# the six module composition blocks that Plan 4 T32–T37 will add. No `module`
# blocks live here yet — referencing a `source = "./modules/<name>"` before
# that directory exists would break `terraform init -backend=false` in CI's
# terraform-validate job. Each module task (T32–T37) appends its own block.
#
# When all six modules have landed, this file becomes the single place a
# reviewer reads to see how the system wires together at the infra level.
# ---------------------------------------------------------------------------

locals {
  # Stamped onto every taggable AWS resource via provider.aws.default_tags
  # in providers.tf — modules need not re-declare these tags individually.
  common_tags = {
    project    = var.project_name
    env        = var.env
    managed_by = "terraform"
    repo       = "NoamDz/AI-Investment-Firm"
  }
}

# ---------------------------------------------------------------------------
# Module composition (Plan 4 T32–T37). Each task appends its own block.
#
#   1. module "network"        (T32, DONE)  — VPC, public/private subnets,
#                                              NAT, SGs.
#   2. module "storage"        (T34, DONE)  — RDS Postgres + S3 buckets
#                                              (reports, traces, eval
#                                              cassettes).
#   3. module "secrets"        (T35, TODO)  — Secrets Manager entries (HMAC,
#                                              API keys, broker creds).
#   4. module "bedrock"        (T36, TODO)  — AgentCore runtime config + IAM
#                                              role for the Reporter agent
#                                              (§11.1).
#   5. module "compute"        (T33, DONE)  — ECS Fargate cluster + task
#                                              definition + service
#                                              (consumes network + secrets).
#   6. module "observability"  (T37, TODO)  — CloudWatch log groups + OTLP
#                                              collector sidecar wiring +
#                                              alarms.
#
# Composition order above reflects dependency direction: network and storage
# stand alone; secrets feeds compute; bedrock is self-contained; compute pulls
# from network + secrets; observability wraps compute.
# ---------------------------------------------------------------------------

module "network" {
  source = "./modules/network"

  vpc_cidr     = var.vpc_cidr
  project_name = var.project_name
  env          = var.env
}

module "compute" {
  source = "./modules/compute"

  project_name               = var.project_name
  env                        = var.env
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  private_subnet_ids         = module.network.private_subnet_ids
  ecs_task_security_group_id = module.network.ecs_task_security_group_id
}

module "storage" {
  source = "./modules/storage"

  project_name          = var.project_name
  env                   = var.env
  db_instance_class     = var.db_instance_class
  private_subnet_ids    = module.network.private_subnet_ids
  rds_security_group_id = module.network.rds_security_group_id
}

module "secrets" {
  source = "./modules/secrets"

  project_name = var.project_name
  env          = var.env
  # kms_key_deletion_window_days uses the module default (30) — maximum AWS
  # recovery window; override via root tfvars only if tests need faster cleanup.
}
