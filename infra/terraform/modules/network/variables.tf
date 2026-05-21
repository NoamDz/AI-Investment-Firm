# ---------------------------------------------------------------------------
# modules/network — input variables (Plan 4 T32)
#
# All three are required (no defaults): the orchestrator (../../main.tf) is
# the single source of truth for these values, so defaulting here would risk
# silent divergence between the module and its caller.
# ---------------------------------------------------------------------------

variable "vpc_cidr" {
  description = "CIDR block for the VPC; subnets are carved as /24s via cidrsubnet(). A /16 yields 256 candidate /24s — plenty of headroom."
  type        = string
}

variable "project_name" {
  description = "Short project slug used to prefix Name tags on every resource."
  type        = string
}

variable "env" {
  description = "Deployment environment ('dev' or 'prod'); already validated by the orchestrator."
  type        = string
}
