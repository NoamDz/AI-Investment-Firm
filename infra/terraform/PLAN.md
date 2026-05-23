# Terraform Plan — Dev Environment

> **This file is hand-curated.** The CI sandbox running this take-home does
> not have AWS credentials to execute `terraform plan` against a real account.
> To regenerate against a real account, run:
>
> ```bash
> cd infra/terraform
> terraform init -backend=false
> terraform plan -no-color -var-file=envs/dev.tfvars \
>   | bash ../../scripts/sanitise_plan.sh \
>   > PLAN.md
> ```
>
> The `main.yml` CI workflow runs `terraform plan` on every push to main and
> uploads the (sanitised) output as a workflow artifact (`tfplan.txt`); see
> `.github/workflows/main.yml` → `terraform-plan` job. Operators compare that
> artifact against this committed PLAN.md to detect semantic drift.
>
> Last refresh: hand-curated 2026-05-22 (Plan 4 T38; branch
> `worktree-plan4-eval-redteam-cicd-deploy`). AWS account IDs and ARN region
> tokens have been replaced with `<account-id>` / `<region>` placeholders by
> `scripts/sanitise_plan.sh`.

---

## Inputs (`envs/dev.tfvars` + `variables.tf` defaults)

| Variable             | Value            |
|----------------------|------------------|
| `region`             | `us-east-1`      |
| `env`                | `dev`            |
| `project_name`       | `firm`           |
| `vpc_cidr`           | `10.0.0.0/16`    |
| `db_instance_class`  | `db.t4g.micro`   |
| `ecs_task_cpu`       | `1024`           |
| `ecs_task_memory`    | `2048`           |

---

## Summary

Terraform will perform the following actions:

```
Plan: 66 to add, 0 to change, 0 to destroy.
```

Resource counts are derived from the HCL across the six modules, with
`count` / `for_each` expansions accounted for (e.g. `aws_subnet.public`
has `count = 2`, `aws_s3_bucket.this` has `for_each` over a 3-element set).
Per-block annotations like `(×2 count)` or `(×3 for_each)` appear inline
next to multi-instance blocks below so the math is reader-checkable.
Counts may still drift ±2 as the AWS provider evolves attribute defaults;
the `main.yml` CI artifact is the authoritative source.

Per-module subtotals (sum to grand total 66):

| Module          | Resources |
|-----------------|-----------|
| network         | 20        |
| compute         | 10        |
| storage         | 16        |
| secrets         | 8         |
| bedrock         | 3         |
| observability   | 9         |
| **Total**       | **66**    |

---

## module.network (`infra/terraform/modules/network`)

```
  # aws_vpc.this will be created
  + resource "aws_vpc" "this" {
      + cidr_block                       = "10.0.0.0/16"
      + enable_dns_hostnames             = true
      + enable_dns_support               = true
      + id                               = <computed>
      + tags                             = {
          + "Name"       = "firm-dev-vpc"
          + "env"        = "dev"
          + "managed_by" = "terraform"
          + "project"    = "firm"
        }
    }

  # aws_internet_gateway.this will be created
  + resource "aws_internet_gateway" "this" {
      + id     = <computed>
      + vpc_id = <computed>
    }

  # aws_eip.nat will be created
  + resource "aws_eip" "nat" {
      + domain     = "vpc"
      + id         = <computed>
      + public_ip  = <computed>
    }

  # aws_nat_gateway.this will be created
  + resource "aws_nat_gateway" "this" {
      + allocation_id = <computed>
      + id            = <computed>
      + subnet_id     = <computed>
    }

  # aws_subnet.public[*] will be created  (×2 count: 10.0.0.0/24 AZ-a, 10.0.1.0/24 AZ-b)
  # aws_subnet.private[*] will be created (×2 count: 10.0.10.0/24 AZ-a, 10.0.11.0/24 AZ-b)

  # aws_route_table.public will be created  (default route → IGW)
  # aws_route_table.private will be created (default route → NAT)
  # aws_route_table_association.public[*]  will be created (×2 count)
  # aws_route_table_association.private[*] will be created (×2 count)

  # aws_security_group.ecs_task will be created
  + resource "aws_security_group" "ecs_task" {
      + description = "ECS task: egress only"
      + id          = <computed>
      + name        = "firm-dev-ecs-task-sg"
      + vpc_id      = <computed>
    }
  # aws_security_group_rule.ecs_task_egress_all will be created (all/0.0.0.0/0)

  # aws_security_group.rds will be created
  + resource "aws_security_group" "rds" {
      + description = "RDS Postgres: 5432 from ECS only"
      + id          = <computed>
      + name        = "firm-dev-rds-sg"
      + vpc_id      = <computed>
    }
  # aws_security_group_rule.rds_ingress_from_ecs will be created (tcp/5432)

  # aws_security_group.otlp_collector will be created
  + resource "aws_security_group" "otlp_collector" {
      + description = "OTLP collector: 4317 from ECS only"
      + id          = <computed>
      + name        = "firm-dev-otlp-sg"
      + vpc_id      = <computed>
    }
  # aws_security_group_rule.otlp_ingress_from_ecs will be created (tcp/4317)
```

**module.network subtotal: 20 resources to add**

> Math: 12 single-instance blocks (vpc, igw, eip, nat, 2×route_table,
> 3×security_group, 3×security_group_rule) + 4 count=2 blocks expanded to
> 8 instances (subnet.public×2, subnet.private×2, route_table_association.public×2,
> route_table_association.private×2) = 12 + 8 = 20.

---

## module.compute (`infra/terraform/modules/compute`)

```
  # aws_ecs_cluster.this will be created
  + resource "aws_ecs_cluster" "this" {
      + id   = <computed>
      + name = "firm-dev"
      + setting {
          + name  = "containerInsights"
          + value = "enabled"
        }
    }

  # aws_cloudwatch_log_group.ecs will be created
  + resource "aws_cloudwatch_log_group" "ecs" {
      + id                = <computed>
      + name              = "/ecs/firm-dev"
      + retention_in_days = 90
    }

  # aws_iam_role.task_execution will be created
  + resource "aws_iam_role" "task_execution" {
      + arn  = <computed>
      + id   = <computed>
      + name = "firm-dev-task-execution"
    }

  # aws_iam_role_policy_attachment.task_execution_managed will be created
  + resource "aws_iam_role_policy_attachment" "task_execution_managed" {
      + policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
      + role       = "firm-dev-task-execution"
    }

  # aws_iam_role.task will be created
  + resource "aws_iam_role" "task" {
      + arn  = <computed>
      + id   = <computed>
      + name = "firm-dev-task"
    }

  # aws_iam_role_policy.task_policy will be created
  + resource "aws_iam_role_policy" "task_policy" {
      + id   = <computed>
      + name = "firm-dev-task-policy"
      + role = <computed>
      # Grants: secretsmanager:GetSecretValue firm/*, bedrock:InvokeAgent *,
      #         s3 RW firm-dev-reports, logs:PutLogEvents /ecs/firm-dev
    }

  # aws_ecs_task_definition.this will be created
  + resource "aws_ecs_task_definition" "this" {
      + arn    = <computed>
      + cpu    = "1024"
      + family = "firm-dev"
      + memory = "2048"
      + requires_compatibilities = ["FARGATE"]
      + network_mode             = "awsvpc"
    }

  # aws_ecs_service.this will be created
  + resource "aws_ecs_service" "this" {
      + cluster         = <computed>
      + desired_count   = 1
      + id              = <computed>
      + launch_type     = "FARGATE"
      + name            = "firm-dev"
      + task_definition = <computed>
    }

  # aws_appautoscaling_target.ecs will be created
  + resource "aws_appautoscaling_target" "ecs" {
      + max_capacity       = 3
      + min_capacity       = 1
      + scalable_dimension = "ecs:service:DesiredCount"
      + service_namespace  = "ecs"
    }

  # aws_appautoscaling_policy.ecs_cpu will be created
  + resource "aws_appautoscaling_policy" "ecs_cpu" {
      + name        = "firm-dev-cpu-autoscale"
      + policy_type = "TargetTrackingScaling"
      # target_value = 70.0, predefined_metric = ECSServiceAverageCPUUtilization
    }
```

**module.compute subtotal: 10 resources to add**

> Math: 10 single-instance blocks (ecs_cluster, cloudwatch_log_group.ecs,
> iam_role.task_execution, iam_role_policy_attachment.task_execution_managed,
> iam_role.task, iam_role_policy.task_policy, ecs_task_definition,
> ecs_service, appautoscaling_target, appautoscaling_policy) = 10. No
> count/for_each in this module.

---

## module.storage (`infra/terraform/modules/storage`)

```
  # aws_s3_bucket.this[*] will be created (×3 for_each: reports, traces, cassettes)
  + resource "aws_s3_bucket" "this" {
      + bucket        = "firm-dev-<each.key>"
      + id            = <computed>
    }

  # aws_s3_bucket_versioning.this[*] will be created (×3 for_each, Enabled on all)
  # aws_s3_bucket_server_side_encryption_configuration.this[*] will be created (×3 for_each, AES256)
  # aws_s3_bucket_public_access_block.this[*] will be created (×3 for_each)
  #   block_public_acls=true, block_public_policy=true,
  #   ignore_public_acls=true, restrict_public_buckets=true

  # aws_s3_bucket_lifecycle_configuration.traces will be created
  + resource "aws_s3_bucket_lifecycle_configuration" "traces" {
      + bucket = "firm-dev-traces"
      + rule {
          + id     = "expire-traces-after-90d"
          + status = "Enabled"
          + expiration { days = 90 }
        }
    }

  # aws_db_subnet_group.this will be created
  + resource "aws_db_subnet_group" "this" {
      + id         = <computed>
      + name       = "firm-dev-rds-subnet-group"
      + subnet_ids = <computed>
    }

  # aws_db_parameter_group.this will be created
  + resource "aws_db_parameter_group" "this" {
      + family = "postgres15"
      + id     = <computed>
      + name   = "firm-dev-pg15"
      + parameter { name = "max_connections", value = "200" }
    }

  # aws_db_instance.this will be created
  + resource "aws_db_instance" "this" {
      + allocated_storage            = 20
      + backup_retention_period      = 1
      + db_name                      = "firm"
      + engine                       = "postgres"
      + engine_version               = "15"
      + id                           = <computed>
      + identifier                   = "firm-dev"
      + instance_class               = "db.t4g.micro"
      + manage_master_user_password  = true
      + multi_az                     = false
      + publicly_accessible          = false
      + skip_final_snapshot          = true
      + storage_encrypted            = true
      + username                     = "firm_admin"
      + vpc_security_group_ids       = <computed>
    }
```

**module.storage subtotal: 16 resources to add**

> Math: 4 for_each=3-set blocks expanded to 12 instances (aws_s3_bucket×3,
> aws_s3_bucket_versioning×3, aws_s3_bucket_server_side_encryption_configuration×3,
> aws_s3_bucket_public_access_block×3) + 4 single-instance blocks (lifecycle
> on traces, db_subnet_group, db_parameter_group, db_instance) = 12 + 4 = 16.

---

## module.secrets (`infra/terraform/modules/secrets`)

```
  # aws_kms_key.secrets will be created
  + resource "aws_kms_key" "secrets" {
      + arn                     = <computed>
      + deletion_window_in_days = 30
      + description             = "firm-dev secrets encryption key"
      + enable_key_rotation     = true
      + id                      = <computed>
      + key_id                  = <computed>
    }

  # aws_kms_alias.secrets will be created
  + resource "aws_kms_alias" "secrets" {
      + name          = "alias/firm-dev-secrets"
      + target_key_id = <computed>
    }

  # aws_secretsmanager_secret.this[*] will be created (×6 for_each):
  #   firm/anthropic_api_key, firm/slack_signing_secret, firm/slack_bot_token,
  #   firm/firm_hmac_secret, firm/firm_hmac_secret_prev, firm/firm_hmac_rotated_at
  + resource "aws_secretsmanager_secret" "this" {
      + arn                     = <computed>
      + description             = "firm-dev secret — value written out-of-band by operators"
      + id                      = <computed>
      + kms_key_id              = <computed>
      + name                    = <each.key>
      + recovery_window_in_days = 30
    }
    # No aws_secretsmanager_secret_version — values are written out-of-band.
```

**module.secrets subtotal: 8 resources to add**

> Math: 2 single-instance blocks (kms_key.secrets, kms_alias.secrets) +
> 1 for_each=6-set block expanded to 6 instances (secretsmanager_secret×6)
> = 2 + 6 = 8.

---

## module.bedrock (`infra/terraform/modules/bedrock`)

```
  # aws_iam_role.agentcore_runtime will be created
  + resource "aws_iam_role" "agentcore_runtime" {
      + arn  = <computed>
      + id   = <computed>
      + name = "firm-dev-agentcore-runtime"
      # Trust: bedrock-agentcore.amazonaws.com
    }

  # aws_iam_role_policy.agentcore_runtime will be created
  + resource "aws_iam_role_policy" "agentcore_runtime" {
      + name = "firm-dev-agentcore-runtime-policy"
      + role = <computed>
      # Grants:
      #   SecretsManagerReadHmac  — GetSecretValue on 3 HMAC secret ARNs
      #   KmsDecryptSecrets       — kms:Decrypt on secrets CMK ARN
      #   CloudWatchLogsWrite     — logs:CreateLogStream + PutLogEvents on
      #     arn:aws:logs:*:*:log-group:/aws/bedrock-agentcore/firm-dev-reporter:*
    }

  # aws_cloudwatch_log_group.agentcore_reporter will be created
  + resource "aws_cloudwatch_log_group" "agentcore_reporter" {
      + id                = <computed>
      + name              = "/aws/bedrock-agentcore/firm-dev-reporter"
      + retention_in_days = 90
    }

  # (Symbolic) AgentCore Layer B entities — managed by AgentCore CLI, not Terraform:
  #   agentcore_runtime_name     = "firm-reporter"
  #   agentcore_memory_namespace = "firm-desk-state"
```

**module.bedrock subtotal: 3 resources to add**

> Math: 3 single-instance blocks (iam_role.agentcore_runtime,
> iam_role_policy.agentcore_runtime, cloudwatch_log_group.agentcore_reporter)
> = 3. Symbolic AgentCore Layer B entities are CLI-managed (T39), not
> Terraform resources, so they do not contribute to the count.

---

## module.observability (`infra/terraform/modules/observability`)

```
  # aws_cloudwatch_log_group.firm will be created
  + resource "aws_cloudwatch_log_group" "firm" {
      + id                = <computed>
      + name              = "/firm/dev"
      + retention_in_days = 90
    }

  # aws_cloudwatch_log_group.otelcol will be created
  + resource "aws_cloudwatch_log_group" "otelcol" {
      + id                = <computed>
      + name              = "/ecs/firm-dev-otelcol"
      + retention_in_days = 90
    }

  # aws_iam_role.otelcol_task_execution will be created
  + resource "aws_iam_role" "otelcol_task_execution" {
      + arn  = <computed>
      + id   = <computed>
      + name = "firm-dev-otelcol-task-execution"
    }

  # aws_iam_role_policy_attachment.otelcol_task_execution_managed will be created
  + resource "aws_iam_role_policy_attachment" "otelcol_task_execution_managed" {
      + policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
      + role       = "firm-dev-otelcol-task-execution"
    }

  # aws_iam_role.otelcol_task will be created
  + resource "aws_iam_role" "otelcol_task" {
      + arn  = <computed>
      + id   = <computed>
      + name = "firm-dev-otelcol-task"
    }

  # aws_iam_role_policy.otelcol_task_policy will be created
  + resource "aws_iam_role_policy" "otelcol_task_policy" {
      + name = "firm-dev-otelcol-task-policy"
      + role = <computed>
      # Grants:
      #   CloudWatchLogsWriteTelemetry — logs:CreateLogStream + PutLogEvents on /firm/dev
      #   CloudWatchPutMetricData      — cloudwatch:PutMetricData (Resource: *)
      #   XRayPutTraceSegments         — xray:PutTraceSegments + PutTelemetryRecords (Resource: *)
    }

  # aws_ecs_task_definition.otelcol will be created
  + resource "aws_ecs_task_definition" "otelcol" {
      + arn    = <computed>
      + cpu    = "256"
      + family = "firm-dev-otelcol"
      + memory = "512"
      + requires_compatibilities = ["FARGATE"]
      + network_mode             = "awsvpc"
      # Container: otel/opentelemetry-collector-contrib:0.95.0
      # Ports: 4317/tcp (gRPC OTLP), 4318/tcp (HTTP OTLP)
      # Config via env var OTEL_CONFIG_CONTENTS (awscloudwatchlogs + logging exporters)
    }

  # aws_ecs_service.otelcol will be created
  + resource "aws_ecs_service" "otelcol" {
      + cluster         = "firm-dev"
      + desired_count   = 1
      + id              = <computed>
      + launch_type     = "FARGATE"
      + name            = "firm-dev-otelcol"
      + task_definition = <computed>
      # Private subnets, otlp SG, assign_public_ip=false
      # lifecycle.ignore_changes = [desired_count]
    }

  # aws_cloudwatch_dashboard.firm will be created
  + resource "aws_cloudwatch_dashboard" "firm" {
      + dashboard_arn  = <computed>
      + dashboard_name = "firm-dev"
      # 4 widgets (2×2): heartbeat p50/p95, cost-by-model metric,
      #   failure_mode log table, decision-action histogram.
    }
```

**module.observability subtotal: 9 resources to add**

> Math: 9 single-instance blocks (2× cloudwatch_log_group {firm, otelcol},
> iam_role.otelcol_task_execution, iam_role_policy_attachment.otelcol_task_execution_managed,
> iam_role.otelcol_task, iam_role_policy.otelcol_task_policy,
> ecs_task_definition.otelcol, ecs_service.otelcol, cloudwatch_dashboard.firm)
> = 9. No count/for_each in this module.

---

## Outputs (all `<computed>` at plan-time)

```
  vpc_id                       = <computed>
  private_subnet_ids           = <computed>
  public_subnet_ids            = <computed>
  ecs_task_security_group_id   = <computed>
  rds_security_group_id        = <computed>
  otlp_security_group_id       = <computed>

  ecs_cluster_name             = "firm-dev"
  ecs_cluster_arn              = <computed>

  reports_bucket_name          = "firm-dev-reports"
  traces_bucket_name           = "firm-dev-traces"
  cassettes_bucket_name        = "firm-dev-cassettes"
  db_endpoint                  = <computed>
  db_master_user_secret_arn    = <computed>

  secrets_kms_key_arn          = <computed>
  secret_arns                  = <computed>

  agentcore_runtime_name       = "firm-reporter"
  agentcore_memory_namespace   = "firm-desk-state"
  agentcore_runtime_role_arn   = <computed>

  dashboard_arn                = <computed>
  firm_log_group_name          = "/firm/dev"
  otelcol_log_group_name       = "/ecs/firm-dev-otelcol"
```

---

```
------------------------------------------------------------------------

Plan: 66 to add, 0 to change, 0 to destroy.
      (network 20 + compute 10 + storage 16 + secrets 8 + bedrock 3 +
       observability 9 = 66)

Note: Exact resource count may drift ±2 as the AWS provider evolves
      attribute defaults. The main.yml CI artifact (tfplan.txt) is the
      authoritative source once AWS credentials are available.
------------------------------------------------------------------------
```
