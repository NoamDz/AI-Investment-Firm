# ---------------------------------------------------------------------------
# modules/observability — input variables (Plan 4 T37)
#
# All cross-module inputs are required (no defaults) — the root orchestrator
# is the single source of truth. Module-internal tuning knobs carry defaults.
# ---------------------------------------------------------------------------

variable "project_name" {
  description = "Short project slug used to prefix resource names."
  type        = string
}

variable "env" {
  description = "Deployment environment ('dev' or 'prod'); validated by the orchestrator."
  type        = string
}

variable "vpc_id" {
  description = "VPC ID from modules/network (T32). Not directly used in resources today but available for future VPC-endpoint configuration."
  type        = string
}

variable "private_subnet_ids" {
  description = "Private subnet IDs from modules/network (T32). The OTLP collector task runs here — no public IP, internet-bound traffic via NAT."
  type        = list(string)
}

variable "otlp_security_group_id" {
  description = "Security group ID from modules/network (T32) that permits ingress 4317 + 4318 from the ECS task SG. Attached to the collector service."
  type        = string
}

variable "ecs_cluster_name" {
  description = "Name of the ECS cluster from modules/compute (T33). Observability reuses the app cluster — no separate cluster needed at take-home scale."
  type        = string
}

variable "log_retention_days" {
  description = "CloudWatch log group retention in days. Defaults to 90 to match the 90-day evidence window used in compute (T33) and bedrock (T36)."
  type        = number
  default     = 90
}

variable "otelcol_image" {
  description = <<-EOT
    Container image for the OTLP collector. Pinned to 0.95.0 so dashboard
    metric names don't drift between releases. Updating this version requires
    a follow-up review to confirm collector config compatibility (pipeline
    component names and exporter config keys change between minor releases).
  EOT
  type        = string
  default     = "otel/opentelemetry-collector-contrib:0.95.0"
}

variable "desired_count" {
  description = <<-EOT
    Number of OTLP collector tasks. Default 1 is sufficient for take-home
    scale — a single task can handle the telemetry volume of one concurrent
    research workflow. Multi-AZ HA is a path-to-prod (T44) item.
  EOT
  type        = number
  default     = 1
}
