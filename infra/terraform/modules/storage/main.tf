# ---------------------------------------------------------------------------
# modules/storage (Plan 4 T34)
#
# Persistence layer for the firm:
#   * 3 S3 buckets — reports, traces, cassettes — all versioned, AES256
#     server-side encrypted, with public access fully blocked. Only the
#     traces bucket has a 90-day lifecycle expiration (per the spec —
#     reports and recorded eval cassettes are retained indefinitely as
#     they are evidence artifacts).
#   * 1 RDS Postgres 15 instance in private subnets, accessed only via the
#     SG provisioned in modules/network (T32). Master credentials are
#     managed by AWS (manage_master_user_password) and auto-rotated via
#     Secrets Manager rather than hand-managed in tfvars.
#   * RDS subnet group spanning the two private subnets + parameter group
#     pinning max_connections=200 (per spec).
#
# Forward-reference contract with T33 (compute):
#   modules/compute's task IAM policy grants R/W on the bucket named
#   `${var.project_name}-${var.env}-reports` (line ~181 of compute/main.tf).
#   The bucket name template below MUST match exactly. A `bucket_name_suffix`
#   variable is tempting for global-uniqueness collisions but would break
#   that grant — operators hitting a collision in production should resolve
#   it by changing var.project_name, not by suffixing here. S3 bucket names
#   are GLOBALLY unique across AWS; for the take-home this is unlikely to
#   collide, but a real deploy may need to bump the project slug.
#
# Out of scope for T34 (deferred):
#   * KMS-CMK encryption on the buckets and RDS — T35 (secrets) handles
#     CMK provisioning. AES256 (SSE-S3) is the safe baseline for now.
#   * RDS Performance Insights / Enhanced Monitoring — T37 (observability).
#   * Bucket replication, intelligent tiering, CRR — not needed for the
#     take-home's single-region scope.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# S3 buckets — for_each over a set keeps the three bucket-attached resources
# (versioning, encryption, public access block) from triplicating boilerplate.
# ---------------------------------------------------------------------------

locals {
  buckets = toset(["reports", "traces", "cassettes"])
}

resource "aws_s3_bucket" "this" {
  for_each = local.buckets
  bucket   = "${var.project_name}-${var.env}-${each.key}"

  tags = {
    Name = "${var.project_name}-${var.env}-${each.key}"
  }
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.this[each.key].id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = local.buckets
  bucket   = aws_s3_bucket.this[each.key].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Critical hardening: block every flavor of public access. AWS defaults
# these to false on bucket creation, so an explicit `true` on all four is
# the only safe posture for a fresh bucket.
resource "aws_s3_bucket_public_access_block" "this" {
  for_each                = local.buckets
  bucket                  = aws_s3_bucket.this[each.key].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle rule applies ONLY to traces — reports and cassettes are retained
# indefinitely (reports are user-facing evidence; cassettes are reproducibility
# artifacts for the eval harness). Spec: "lifecycle rule expiring traces
# after 90 days."
resource "aws_s3_bucket_lifecycle_configuration" "traces" {
  bucket = aws_s3_bucket.this["traces"].id

  rule {
    id     = "expire-traces-after-90d"
    status = "Enabled"

    # Empty filter = apply to all objects in the bucket. AWS provider v4.x+
    # warns (will become an error) if neither `filter` nor the deprecated
    # `prefix` is specified.
    filter {}

    expiration {
      days = 90
    }
  }
}

# ---------------------------------------------------------------------------
# RDS Postgres 15
# ---------------------------------------------------------------------------

resource "aws_db_subnet_group" "this" {
  name       = "${var.project_name}-${var.env}-rds-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name = "${var.project_name}-${var.env}-rds-subnet-group"
  }
}

# Parameter group pins max_connections=200 per spec. Postgres' default
# (~100) is fine for steady-state but the firm's bursty per-research-run
# pattern (PM + 4 analysts + tools, each with pooled clients) can transiently
# saturate at default.
resource "aws_db_parameter_group" "this" {
  name   = "${var.project_name}-${var.env}-pg15"
  family = "postgres15"

  parameter {
    name  = "max_connections"
    value = "200"
  }

  tags = {
    Name = "${var.project_name}-${var.env}-pg15"
  }
}

resource "aws_db_instance" "this" {
  identifier        = "${var.project_name}-${var.env}"
  engine            = "postgres"
  engine_version    = "15"
  instance_class    = var.db_instance_class
  allocated_storage = 20
  storage_encrypted = true

  # Postgres database names cannot contain hyphens. project_name = "firm"
  # has none today, but the replace() guards against future renames like
  # "firm-staging".
  db_name  = replace(var.project_name, "-", "_")
  username = "firm_admin"

  # Let AWS generate and rotate the master password via Secrets Manager.
  # Avoids any plaintext password in tfvars or state. The created secret
  # ARN is surfaced as `db_master_user_secret_arn` so future modules
  # (compute, secrets) can grant read access without hand-wiring values.
  manage_master_user_password = true

  db_subnet_group_name   = aws_db_subnet_group.this.name
  parameter_group_name   = aws_db_parameter_group.this.name
  vpc_security_group_ids = [var.rds_security_group_id]

  # Env-gated lifecycle posture:
  #   dev  — destroy-friendly: skip final snapshot, no delete protection,
  #          1-day backups, single-AZ.
  #   prod — durable: take a final snapshot on destroy, enable delete
  #          protection, 7-day backups, multi-AZ for failover.
  skip_final_snapshot     = var.env != "prod"
  deletion_protection     = var.env == "prod"
  backup_retention_period = var.env == "prod" ? 7 : 1
  multi_az                = var.env == "prod"

  # Hard requirement: the instance lives in private subnets and is only
  # reachable via the RDS SG (ingress 5432 from ECS tasks only). Flipping
  # this to true would assign a public IP and expose 5432 to the internet.
  publicly_accessible = false

  tags = {
    Name = "${var.project_name}-${var.env}-postgres"
  }
}
