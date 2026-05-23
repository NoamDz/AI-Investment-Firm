# `infra/` — Terraform stack

AWS infrastructure for the firm. Six modules: VPC, ECS Fargate, RDS Postgres,
S3 (replicas + reports), Secrets Manager, CloudWatch.

A captured dry-run plan lives at [`terraform/PLAN.md`](terraform/PLAN.md) — it
shows exactly what would be created without touching AWS.

## Regenerate the plan

```powershell
terraform -chdir=infra/terraform init -backend=false
terraform -chdir=infra/terraform plan -var-file=envs/dev.tfvars
```

## Provision dev stack (HUMAN-GATED)

```powershell
make deploy-dev
```

Prompts for `DEPLOY` confirmation and creates real resources:

| Resource | Idle cost |
|----------|-----------|
| ECS Fargate cluster | ~$15/mo |
| RDS Postgres `db.t4g.micro` | ~$15/mo |
| NAT Gateway | ~$32/mo + per-GB egress |
| S3 / CloudWatch / Secrets Manager | negligible |

Tear down when done:

```powershell
terraform -chdir=infra/terraform destroy -var-file=envs/dev.tfvars
```

See [`../docs/path-to-production.md`](../docs/path-to-production.md) for the
take-home → prod delta.
