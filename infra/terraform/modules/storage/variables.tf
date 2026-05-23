# ---------------------------------------------------------------------------
# modules/storage — variables (Plan 4 T34)
#
# All values are passed down from the orchestrator in infra/terraform/main.tf.
# No defaults: the module is intentionally not standalone — it is only
# composed via the root module so naming and sizing decisions live in one
# place (variables.tf + envs/*.tfvars at the root).
# ---------------------------------------------------------------------------

variable "project_name" {
  description = "Short project slug used to prefix every resource name (bucket, DB identifier, parameter group)."
  type        = string
}

variable "env" {
  description = "Deployment environment ('dev' or 'prod'). Gates RDS skip_final_snapshot, deletion_protection, backup retention, and multi-AZ."
  type        = string
}

variable "db_instance_class" {
  description = "RDS Postgres instance class. Orchestrator passes db.t4g.micro for dev, db.r6g.large for prod (via envs/prod.tfvars)."
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs (two AZs) for the RDS subnet group. Sourced from module.network.private_subnet_ids."
  type        = list(string)
}

variable "rds_security_group_id" {
  description = "Security group ID for the RDS instance — already configured for ingress 5432 from the ECS task SG. Sourced from module.network.rds_security_group_id."
  type        = string
}
