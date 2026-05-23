# ---------------------------------------------------------------------------
# modules/compute (Plan 4 T33)
#
# ECS Fargate runtime for the firm container:
#   * 1 ECS cluster with Container Insights enabled
#   * 1 CloudWatch log group (/ecs/<project>-<env>, 90-day retention) for
#     task-stdout — full observability (OTLP collector, dashboards, alarms)
#     lands in modules/observability (T37).
#   * 2 IAM roles per the Fargate split:
#       - task_execution: assumed by ECS itself to pull the image and
#         publish logs (AWS-managed AmazonECSTaskExecutionRolePolicy)
#       - task:           assumed by application code; inline policy grants
#         Secrets Manager read (firm/*), Bedrock InvokeAgent, S3 reports
#         bucket RW, and CloudWatch Logs write on this log group.
#   * 1 task definition (FARGATE, awsvpc) with a single `firm` container
#     listening on :8080 for the Slack webhook (future ALB wiring).
#   * 1 ECS service (desired_count=1, ignored on subsequent applies so
#     autoscaling decisions stick) in the private subnets.
#   * Target-tracking autoscaling on ECSServiceAverageCPUUtilization @ 70%,
#     bounds [1, 3] tasks.
#
# Forward references to T34 (storage) and T35 (secrets):
#   The IAM policy grants access to ARNs that do NOT exist yet. Resources
#   are referenced by naming-convention ARNs (e.g. arn:aws:s3:::firm-dev-
#   reports/*) so compute is self-contained and applies cleanly today. T34
#   and T35 create the actual resources under those names; no cross-module
#   wiring needs to change later.
# ---------------------------------------------------------------------------

# Region needed for the awslogs driver in container_definitions; account ID
# kept for future tightening (Bedrock IAM is coarse-grained today, so we
# leave Resource = "*" on bedrock:* — see T44 prod hardening for the recipe).
data "aws_region" "current" {}

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# ECS cluster
# ---------------------------------------------------------------------------

resource "aws_ecs_cluster" "this" {
  name = "${var.project_name}-${var.env}"

  # Container Insights is ~$0.005/metric/hr but gives per-task CPU/mem/network
  # series and the autoscaling policy below depends on the same CW namespace,
  # so the enable is effectively free relative to the value for a take-home.
  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "${var.project_name}-${var.env}-cluster"
  }
}

# ---------------------------------------------------------------------------
# CloudWatch log group for task stdout/stderr
#
# Scope: task-stdout only. Application traces / metrics flow via the OTLP
# collector deployed in modules/observability (T37), not through this group.
# 90-day retention matches the take-home's evidence-window requirement
# without paying for indefinite Standard-tier storage.
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project_name}-${var.env}"
  retention_in_days = 90

  tags = {
    Name = "/ecs/${var.project_name}-${var.env}"
  }
}

# ---------------------------------------------------------------------------
# IAM — task execution role (used by ECS itself)
#
# This role lets the ECS agent pull the container image from GHCR and push
# task-stdout to CloudWatch on the application's behalf. The AWS-managed
# policy AmazonECSTaskExecutionRolePolicy is the canonical grant; it covers
# ECR pulls (for future ECR migration) and logs:CreateLogStream +
# logs:PutLogEvents on any /ecs/* group.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "task_execution" {
  name = "${var.project_name}-${var.env}-task-execution"

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

  tags = {
    Name = "${var.project_name}-${var.env}-task-execution"
  }
}

resource "aws_iam_role_policy_attachment" "task_execution_managed" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ---------------------------------------------------------------------------
# IAM — task role (used by the application code inside the container)
#
# Inline policy covers the four grants Plan 4 T33 requires. Resource ARNs
# use naming-convention strings rather than cross-module references so this
# module is self-contained today (T34 storage + T35 secrets don't exist yet
# but their resources will land under these names).
# ---------------------------------------------------------------------------

resource "aws_iam_role" "task" {
  name = "${var.project_name}-${var.env}-task"

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

  tags = {
    Name = "${var.project_name}-${var.env}-task"
  }
}

resource "aws_iam_role_policy" "task_policy" {
  name = "${var.project_name}-${var.env}-task-policy"
  role = aws_iam_role.task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecretsManagerReadFirmPrefix"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = [
          "arn:aws:secretsmanager:*:*:secret:firm/*"
        ]
      },
      {
        Sid    = "BedrockInvokeAgent"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeAgent",
          "bedrock:InvokeAgentRuntime",
          "bedrock:GetAgent"
        ]
        # Bedrock IAM is coarse today — agent ARNs require the agent to
        # already exist, and the agent is provisioned by modules/bedrock
        # (T36). T44 prod hardening tightens this to a specific agent ARN.
        Resource = ["*"]
      },
      {
        Sid    = "S3ReportsBucketRW"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.project_name}-${var.env}-reports",
          "arn:aws:s3:::${var.project_name}-${var.env}-reports/*"
        ]
      },
      {
        Sid    = "CloudWatchLogsWriteOwnGroup"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = [
          "arn:aws:logs:*:*:log-group:/ecs/${var.project_name}-${var.env}:*"
        ]
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# ECS task definition
#
# Single container — the firm orchestrator + agents all live in one process.
# Splitting into per-agent tasks would multiply Fargate baseline cost ~6x
# for a take-home that runs ~1 research workflow at a time.
#
# secrets = [...] is intentionally NOT set here: wiring real Secrets Manager
# ARN-to-env mappings is T35's job. The task role above already permits
# secretsmanager:GetSecretValue, so the application reads via boto3 at
# runtime today.
# ---------------------------------------------------------------------------

resource "aws_ecs_task_definition" "this" {
  family                   = "${var.project_name}-${var.env}"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = var.project_name
      image     = var.container_image
      essential = true

      portMappings = [
        {
          # Port 8080: Slack webhook listener inside the firm container.
          # Exposed in awsvpc mode so a future ALB target group can attach
          # without redeploying the task definition.
          containerPort = 8080
          protocol      = "tcp"
        }
      ]

      environment = [
        {
          name  = "FIRM_ENV"
          value = var.env
        },
        {
          name  = "FIRM_LOG_LEVEL"
          value = "info"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.ecs.name
          awslogs-region        = data.aws_region.current.name
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])

  tags = {
    Name = "${var.project_name}-${var.env}-task"
  }
}

# ---------------------------------------------------------------------------
# ECS service
#
# `desired_count = 1` seeds the service; the autoscaling target below owns
# steady-state count. `lifecycle.ignore_changes = [desired_count]` keeps
# Terraform from yanking the count back to 1 on every subsequent apply.
#
# `assign_public_ip = false` because tasks live in private subnets and reach
# the internet via the NAT Gateway provisioned in modules/network (T32).
# ---------------------------------------------------------------------------

resource "aws_ecs_service" "this" {
  name            = "${var.project_name}-${var.env}"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.this.arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [var.ecs_task_security_group_id]
    assign_public_ip = false
  }

  lifecycle {
    ignore_changes = [desired_count]
  }

  tags = {
    Name = "${var.project_name}-${var.env}-service"
  }
}

# ---------------------------------------------------------------------------
# Service-level autoscaling — 1 to 3 tasks on CPU > 70%
#
# Target-tracking is the modern AWS-recommended pattern: simpler than step
# scaling, and the App Auto Scaling service picks scale-in/scale-out cool-
# downs automatically. CPU is the natural signal here because the LLM agent
# workers are compute-bound on JSON parsing + tool-call orchestration.
# ---------------------------------------------------------------------------

resource "aws_appautoscaling_target" "ecs" {
  max_capacity       = 3
  min_capacity       = 1
  resource_id        = "service/${aws_ecs_cluster.this.name}/${aws_ecs_service.this.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "ecs_cpu" {
  name               = "${var.project_name}-${var.env}-cpu-autoscale"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs.service_namespace

  target_tracking_scaling_policy_configuration {
    target_value = 70.0

    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
  }
}
