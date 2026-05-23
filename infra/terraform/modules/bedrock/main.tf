# ---------------------------------------------------------------------------
# modules/bedrock (Plan 4 T36)
#
# TWO-LAYER DESIGN — read this before editing:
#
# Layer A: Real, Terraform-shippable resources (validate-clean, plan-safe).
#   * aws_iam_role.agentcore_runtime — trusted by bedrock-agentcore.amazonaws.com
#     (the documented AgentCore service principal; see AWS docs §Identity).
#     Inline policy grants the three HMAC secret ARNs + KMS decrypt + CW log
#     writes on the AgentCore-managed log group path.
#   * aws_cloudwatch_log_group.agentcore_reporter — 90-day retention at
#     /aws/bedrock-agentcore/<project>-<env>-reporter.
#
# Layer B: Symbolic AgentCore configuration (documented intent, not provisioned).
#   The locals block below captures the three AgentCore entity names:
#     agentcore_runtime_name     = "firm-reporter"
#     agentcore_memory_namespace = "firm-desk-state"
#   These are emitted as outputs so T39 AgentCore CLI commands can consume
#   them without re-plumbing values.
#
#   IMPORTANT: Terraform tracks Layer A (IAM + CW); the AgentCore CLI tracks
#   Layer B (runtime / memory / identity entities). The two planes are
#   intentionally separate because the aws ~> 5.0 provider does NOT yet have
#   first-class resources for AgentCore Runtime, Memory, or Identity. T36 is
#   validate-only; T39 runs the AgentCore CLI to confirm the Reporter executes.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Layer B: Symbolic AgentCore entity names
#
# These locals are the canonical source of truth for T39 CLI invocations.
# agentcore_identity_secret_arn is exposed via output (not a local) because
# it comes from an input variable rather than being a literal constant.
# ---------------------------------------------------------------------------

locals {
  # AgentCore runtime name — matches the entity name T39 passes to:
  #   aws bedrock-agentcore create-runtime --name <agentcore_runtime_name>
  agentcore_runtime_name = "firm-reporter"

  # AgentCore memory namespace — the shared state namespace for the Reporter's
  # desk memory (conversation history + scratchpad) across invocations.
  agentcore_memory_namespace = "firm-desk-state"
}

# ---------------------------------------------------------------------------
# Layer A: IAM role for AgentCore Runtime
#
# bedrock-agentcore.amazonaws.com is the documented service principal for
# the AgentCore Runtime service (analogous to ecs-tasks.amazonaws.com for
# ECS). The role name follows the <project>-<env>-<component> convention
# used by T33 and T35.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "agentcore_runtime" {
  name = "${var.project_name}-${var.env}-agentcore-runtime"

  # Trust policy: the AgentCore service assumes this role on behalf of the
  # firm-reporter runtime. "bedrock-agentcore.amazonaws.com" is the documented
  # service principal name per AWS AgentCore Identity documentation.
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock-agentcore.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# IAM inline policy — four grants for the AgentCore runtime
#
# Grouped into three statements:
#   1. SecretsManagerReadHmac — GetSecretValue on the three HMAC secrets.
#      All three ARNs in a single statement (cleaner diffs, single audit line).
#   2. KmsDecryptSecrets — kms:Decrypt on the CMK that encrypts those secrets.
#      Required because the secrets use a customer-managed key (T35 default).
#   3. CloudWatchLogsWriteAgentCore — log writes on the AgentCore-managed log
#      group path. The AgentCore service creates the log group out-of-band;
#      the IAM grant uses a wildcard ARN so it pre-authorises the path without
#      requiring the group to exist at plan time.
# ---------------------------------------------------------------------------

resource "aws_iam_role_policy" "agentcore_runtime" {
  name = "${var.project_name}-${var.env}-agentcore-runtime-policy"
  role = aws_iam_role.agentcore_runtime.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SecretsManagerReadHmac"
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        # All three HMAC-related secrets in one statement — the Identity
        # provider needs current, previous, and rotation-timestamp values
        # to verify request signatures across key rotations.
        Resource = [
          var.hmac_secret_arn,
          var.hmac_secret_prev_arn,
          var.hmac_rotated_at_arn,
        ]
      },
      {
        Sid    = "KmsDecryptSecrets"
        Effect = "Allow"
        Action = [
          "kms:Decrypt"
        ]
        # The three HMAC secrets are encrypted with the T35 CMK; AgentCore
        # must decrypt them when the Identity provider validates signatures.
        Resource = [
          var.secrets_kms_key_arn
        ]
      },
      {
        Sid    = "CloudWatchLogsWriteAgentCore"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        # Wildcard on the log group ARN: the AgentCore service creates and
        # manages /aws/bedrock-agentcore/<runtime> groups out-of-band.
        # Using a naming-convention ARN pre-authorises the write without
        # requiring the group to exist in Terraform state.
        Resource = [
          "arn:aws:logs:*:*:log-group:/aws/bedrock-agentcore/${var.project_name}-${var.env}-reporter:*"
        ]
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# CloudWatch Log Group — AgentCore Reporter stdout/stderr
#
# Path mirrors the AgentCore convention: /aws/bedrock-agentcore/<runtime>.
# The AgentCore service also creates this group automatically on first run,
# but pre-creating it gives Terraform ownership of the retention policy —
# otherwise AWS defaults to "Never Expire" and the group escapes IaC control.
# 90-day retention matches the ECS task log group in modules/compute (T33).
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "agentcore_reporter" {
  name              = "/aws/bedrock-agentcore/${var.project_name}-${var.env}-reporter"
  retention_in_days = var.log_retention_days
}
