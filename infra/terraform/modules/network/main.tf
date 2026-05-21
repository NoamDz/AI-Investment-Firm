# ---------------------------------------------------------------------------
# modules/network (Plan 4 T32)
#
# Lays down the foundational AWS networking the rest of the system rides on:
#   * 1 VPC (var.vpc_cidr; default /16)
#   * 2 public subnets + 2 private subnets across the first 2 AZs of the
#     active region (deterministic ordering via `aws_availability_zones`)
#   * 1 Internet Gateway (public egress) + 1 NAT Gateway (private egress)
#   * Public + private route tables with the corresponding default routes
#   * 3 Security Groups: ECS task (egress only), RDS Postgres (5432 from
#     ECS only), OTLP collector (4317 from ECS only)
#
# Tagging note: providers.tf in the orchestrator declares default_tags with
# project / env / managed_by / repo, so each resource here only needs a Name
# tag to be uniquely identifiable in the console.
#
# Cost-conscious dev choice: a single NAT Gateway in AZ-0 services both
# private subnets. Production would deploy one NAT per AZ to remove the
# cross-AZ SPOF and avoid cross-AZ data charges; see docs/path-to-production.md
# (T44) for the upgrade recipe.
# ---------------------------------------------------------------------------

# Deterministic AZ selection — `state = "available"` filters out AZs in
# maintenance so plans stay reproducible across runs.
data "aws_availability_zones" "available" {
  state = "available"
}

# ---------------------------------------------------------------------------
# VPC
# ---------------------------------------------------------------------------

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-${var.env}-vpc"
  }
}

# ---------------------------------------------------------------------------
# Subnets — 2 public + 2 private across 2 AZs
#
# CIDR carving (assuming /16 VPC):
#   public[0]  = 10.0.0.0/24   (AZ-0)
#   public[1]  = 10.0.1.0/24   (AZ-1)
#   private[0] = 10.0.10.0/24  (AZ-0)
#   private[1] = 10.0.11.0/24  (AZ-1)
# Private subnets are offset by +10 so the CIDR ranges read distinctly when
# debugging route tables / flow logs.
# ---------------------------------------------------------------------------

resource "aws_subnet" "public" {
  count = 2

  vpc_id                  = aws_vpc.this.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-${var.env}-public-${count.index}"
  }
}

resource "aws_subnet" "private" {
  count = 2

  vpc_id                  = aws_vpc.this.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = false

  tags = {
    Name = "${var.project_name}-${var.env}-private-${count.index}"
  }
}

# ---------------------------------------------------------------------------
# Internet Gateway + NAT Gateway
# ---------------------------------------------------------------------------

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id

  tags = {
    Name = "${var.project_name}-${var.env}-igw"
  }
}

# `domain = "vpc"` is the AWS provider 4.x+ replacement for `vpc = true`.
resource "aws_eip" "nat" {
  domain = "vpc"

  tags = {
    Name = "${var.project_name}-${var.env}-nat-eip"
  }
}

resource "aws_nat_gateway" "this" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name = "${var.project_name}-${var.env}-nat"
  }

  # Per AWS provider docs: an IGW must exist on the VPC before a NAT Gateway
  # is created in it, otherwise the NAT creation can race the IGW attach.
  depends_on = [aws_internet_gateway.this]
}

# ---------------------------------------------------------------------------
# Route tables
# ---------------------------------------------------------------------------

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }

  tags = {
    Name = "${var.project_name}-${var.env}-public-rt"
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.this.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.this.id
  }

  tags = {
    Name = "${var.project_name}-${var.env}-private-rt"
  }
}

resource "aws_route_table_association" "public" {
  count = 2

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count = 2

  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

# ---------------------------------------------------------------------------
# Security Groups
#
# Rules are declared as standalone `aws_security_group_rule` resources rather
# than inline `ingress { }` / `egress { }` blocks. The standalone form is the
# modern HashiCorp recommendation: it avoids the "ghost rule" drift that
# happens when rules are added/removed dynamically alongside inline blocks.
# ---------------------------------------------------------------------------

# ECS task SG — egress only; no service listens inside the task container
# directly, so no ingress rule is ever required.
resource "aws_security_group" "ecs_task" {
  name        = "${var.project_name}-${var.env}-ecs-task-sg"
  description = "ECS task: egress only"
  vpc_id      = aws_vpc.this.id

  tags = {
    Name = "${var.project_name}-${var.env}-ecs-task-sg"
  }
}

resource "aws_security_group_rule" "ecs_task_egress_all" {
  type              = "egress"
  security_group_id = aws_security_group.ecs_task.id
  from_port         = 0
  to_port           = 0
  protocol          = "-1"
  cidr_blocks       = ["0.0.0.0/0"]
  description       = "Allow all egress from ECS tasks (NAT-routed for private subnets)."
}

# RDS Postgres SG — only the ECS task SG may reach 5432/tcp. AWS's implicit
# allow-all egress is left in place; RDS does not initiate outbound traffic
# in normal operation, and clamping it down would add no security value.
resource "aws_security_group" "rds" {
  name        = "${var.project_name}-${var.env}-rds-sg"
  description = "RDS Postgres: 5432 from ECS only"
  vpc_id      = aws_vpc.this.id

  tags = {
    Name = "${var.project_name}-${var.env}-rds-sg"
  }
}

resource "aws_security_group_rule" "rds_ingress_from_ecs" {
  type                     = "ingress"
  security_group_id        = aws_security_group.rds.id
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.ecs_task.id
  description              = "Allow Postgres ingress from ECS tasks only."
}

# OTLP collector SG — only the ECS task SG may reach 4317/tcp (gRPC).
# Default allow-all egress is sufficient: the collector pushes spans/metrics
# to CloudWatch via the IGW (or future VPC endpoint), no explicit rule needed.
resource "aws_security_group" "otlp_collector" {
  name        = "${var.project_name}-${var.env}-otlp-sg"
  description = "OTLP collector: 4317 from ECS only"
  vpc_id      = aws_vpc.this.id

  tags = {
    Name = "${var.project_name}-${var.env}-otlp-sg"
  }
}

resource "aws_security_group_rule" "otlp_ingress_from_ecs" {
  type                     = "ingress"
  security_group_id        = aws_security_group.otlp_collector.id
  from_port                = 4317
  to_port                  = 4317
  protocol                 = "tcp"
  source_security_group_id = aws_security_group.ecs_task.id
  description              = "Allow OTLP/gRPC ingress from ECS tasks only."
}
