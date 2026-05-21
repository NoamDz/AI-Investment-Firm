# ---------------------------------------------------------------------------
# modules/observability (Plan 4 T37)
#
# Telemetry plane for the AI Investment Firm: a dedicated CloudWatch log group
# (/firm/<env>) for OTLP-exported spans/logs, a separate log group for the
# collector's own stdout, an otelcol-contrib Fargate service listening on
# 4317/4318 inside the VPC, and a 4-widget CloudWatch dashboard.
#
# This module reuses the app ECS cluster from modules/compute (T33) — no
# separate cluster is provisioned here.
# ---------------------------------------------------------------------------

# Region consumed by the awslogs driver and the awscloudwatchlogs exporter
# config; kept at module scope so both log groups and task def share the value.
data "aws_region" "current" {}

# ---------------------------------------------------------------------------
# CloudWatch log groups
#
# Two groups are intentional:
#   /firm/<env>                    — application telemetry the otelcol
#                                    EXPORTS into via awscloudwatchlogs.
#                                    NOT /ecs/... because this is the
#                                    telemetry namespace, not container stdout.
#   /ecs/<project>-<env>-otelcol   — collector container's own stdout/stderr
#                                    (boot messages, errors). Operational noise
#                                    that must not pollute the telemetry group.
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "firm" {
  name              = "/firm/${var.env}"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "otelcol" {
  name              = "/ecs/${var.project_name}-${var.env}-otelcol"
  retention_in_days = var.log_retention_days
}

# ---------------------------------------------------------------------------
# IAM — task execution role (assumed by ECS itself)
#
# Covers ECR image pull and CloudWatch log writes for the collector's own
# stdout (/ecs/<project>-<env>-otelcol). The AWS-managed policy is the
# canonical grant for this split; it never touches the telemetry log group.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "otelcol_task_execution" {
  name = "${var.project_name}-${var.env}-otelcol-task-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "otelcol_task_execution_managed" {
  role       = aws_iam_role.otelcol_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---------------------------------------------------------------------------
# IAM — task role (assumed by the collector process inside the container)
#
# Three statement groups:
#   1. CloudWatchLogsWriteTelemetry — log writes scoped to the /firm/<env>
#      group (the awscloudwatchlogs exporter target). Wildcard on :* for log
#      streams because the exporter creates stream names at runtime.
#   2. CloudWatchPutMetricData — cloudwatch:PutMetricData has no resource ARN
#      support; AWS requires Resource = ["*"] for this action (IAM docs §CW).
#   3. XRayPutTraceSegments — same situation: xray:Put* actions do not support
#      resource-level restrictions; Resource = ["*"] is required by the API.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "otelcol_task" {
  name = "${var.project_name}-${var.env}-otelcol-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "otelcol_task_policy" {
  name = "${var.project_name}-${var.env}-otelcol-task-policy"
  role = aws_iam_role.otelcol_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogsWriteTelemetry"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        # Scoped to the /firm/<env> group (awscloudwatchlogs exporter target).
        # The :* suffix covers log streams the exporter creates at runtime.
        Resource = [
          "${aws_cloudwatch_log_group.firm.arn}:*"
        ]
      },
      {
        Sid    = "CloudWatchPutMetricData"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        # PutMetricData does not support resource ARN scoping — AWS requires
        # Resource = ["*"] for this action (see IAM CW action reference).
        Resource = ["*"]
      },
      {
        Sid    = "XRayPutTraceSegments"
        Effect = "Allow"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords"
        ]
        # X-Ray Put* actions have no resource-level restriction support —
        # Resource = ["*"] is required by the API (see IAM X-Ray reference).
        Resource = ["*"]
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# OTLP collector configuration (inline via environment variable)
#
# otelcol-contrib supports --config=env:VARNAME; we embed the YAML as an env
# var so there is no config-file volume to mount and no S3 fetch at startup.
# The awscloudwatchlogs exporter targets /firm/<env> (telemetry namespace).
# The logging exporter provides collector-internal stdout visibility.
# ---------------------------------------------------------------------------

locals {
  otelcol_config_yaml = <<-YAML
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318

    processors:
      batch:
        send_batch_size: 1024
        timeout: 5s

    exporters:
      awscloudwatchlogs:
        log_group_name: ${aws_cloudwatch_log_group.firm.name}
        log_stream_name: otlp-export
        region: ${data.aws_region.current.name}
      logging:
        verbosity: normal

    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [batch]
          exporters: [awscloudwatchlogs, logging]
        metrics:
          receivers: [otlp]
          processors: [batch]
          exporters: [logging]
        logs:
          receivers: [otlp]
          processors: [batch]
          exporters: [awscloudwatchlogs, logging]
  YAML
}

# ---------------------------------------------------------------------------
# ECS task definition — otelcol-contrib
#
# 256 cpu / 512 mb: collector is sidecar-grade (JSON parsing + HTTP/gRPC
# proxying). Using the app cluster's task definition slot keeps IAM audit
# trails consolidated under one ECS cluster per environment.
# ---------------------------------------------------------------------------

resource "aws_ecs_task_definition" "otelcol" {
  family                   = "${var.project_name}-${var.env}-otelcol"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 256
  memory                   = 512
  execution_role_arn       = aws_iam_role.otelcol_task_execution.arn
  task_role_arn            = aws_iam_role.otelcol_task.arn

  container_definitions = jsonencode([
    {
      name      = "otelcol"
      image     = var.otelcol_image
      essential = true

      # --config=env:OTEL_CONFIG_CONTENTS: collector reads YAML from env var.
      # No config file volume needed — keeps the task definition self-contained.
      command = ["--config=env:OTEL_CONFIG_CONTENTS"]

      environment = [
        {
          name  = "OTEL_CONFIG_CONTENTS"
          value = local.otelcol_config_yaml
        }
      ]

      portMappings = [
        {
          containerPort = 4317
          hostPort      = 4317
          protocol      = "tcp"
        },
        {
          containerPort = 4318
          hostPort      = 4318
          protocol      = "tcp"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.otelcol.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "otelcol"
        }
      }
    }
  ])
}

# ---------------------------------------------------------------------------
# ECS service — otelcol
#
# Runs on the shared app cluster (var.ecs_cluster_name from modules/compute).
# Private subnets only — the app task reaches the collector via the SG-
# permitted private IPs; no load balancer or public IP needed.
#
# No autoscaling: telemetry volume at take-home scale is well within a single
# 256-cpu task's capacity. Multi-AZ HA is a path-to-prod item (T44).
# lifecycle.ignore_changes = [desired_count] matches the T33 pattern so any
# future autoscaling addition doesn't conflict with Terraform state.
# ---------------------------------------------------------------------------

resource "aws_ecs_service" "otelcol" {
  name            = "${var.project_name}-${var.env}-otelcol"
  cluster         = var.ecs_cluster_name
  task_definition = aws_ecs_task_definition.otelcol.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.otlp_security_group_id]
    assign_public_ip = false
  }

  lifecycle {
    ignore_changes = [desired_count]
  }
}

# ---------------------------------------------------------------------------
# CloudWatch Dashboard
#
# Dashboard body lives in dashboard.json.tftpl (separate file) so the JSON
# is editable without HCL escaping. Four widgets in a 2×2 grid at 1200px.
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_dashboard" "firm" {
  dashboard_name = "${var.project_name}-${var.env}"
  dashboard_body = templatefile("${path.module}/dashboard.json.tftpl", {
    project_name = var.project_name
    env          = var.env
    region       = data.aws_region.current.name
    log_group    = aws_cloudwatch_log_group.firm.name
  })
}
