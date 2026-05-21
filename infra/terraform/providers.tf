# ---------------------------------------------------------------------------
# Provider + backend configuration (Plan 4 T31).
#
# `required_version` matches the floor pinned by hashicorp/setup-terraform@v3
# (>= 1.6.0) in .github/workflows/{pr,main}.yml so local + CI agree on syntax.
#
# AWS provider is `~> 5.0` — the 5.x line covers every resource the T32–T37
# modules will declare (VPC, ECS Fargate, RDS, S3, Secrets Manager, Bedrock
# AgentCore data sources, CloudWatch). default_tags propagates the locals-
# defined common_tags onto every taggable AWS resource, so individual modules
# do not have to re-stamp project/env/managed_by tags themselves.
# ---------------------------------------------------------------------------

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # -------------------------------------------------------------------------
  # Take-home: state lives locally. Production would use S3 + DynamoDB lock.
  # See `docs/path-to-production.md` (T44) for the uncomment-and-bootstrap
  # recipe (create the bucket + table first, then `terraform init -migrate-
  # state`). Left commented so `terraform init -backend=false` in CI does
  # not try to reach a non-existent bucket.
  # -------------------------------------------------------------------------
  # backend "s3" {
  #   bucket         = "firm-tfstate-<account-id>"
  #   key            = "firm/terraform.tfstate"
  #   region         = "us-east-1"
  #   dynamodb_table = "firm-tfstate-lock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.region

  default_tags {
    tags = local.common_tags
  }
}
